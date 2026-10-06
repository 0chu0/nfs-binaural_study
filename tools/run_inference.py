"""
Windows-safe NFS benchmark inference (reusable CLI).

Why this exists
---------------
The official `inference.py` cannot run on Windows as-is: it derives the subset
name with `dp.split('/')[-2]`, but `glob.glob('dataset/.../*/mono.wav')` returns
MIXED separators on Windows, e.g. `dataset/benchmark/testset\\subject1\\mono.wav`.
So `subset` becomes `'benchmark'` and the position path resolves to
`.../testset/benchmark/tx_positions.txt` -> FileNotFoundError.

This launcher reuses every official building block unchanged
(NFS, get_inverse_window, unfold, filter_dict, load_txt) and only replaces the
path parsing with pathlib, so the rendered audio is numerically equivalent.

Usage
-----
    python run_inference.py                          # defaults: lens=1.0, all subsets
    python run_inference.py --lens_sec 0.2           # 200 ms window / 100 ms hop (training config)
    python run_inference.py --subsets subject1 --save_dir ./probe/lens0.2
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

REPO = Path(os.environ.get("NFS_REPO", str(Path(__file__).resolve().parent.parent / "nfs-binaural"))).resolve()
if not (REPO / "networks" / "nfs.py").is_file():
    raise SystemExit(f"[error] 找不到上游代码目录（应含 networks/nfs.py）: {REPO}\n"
                     f"        请用环境变量 NFS_REPO 指定，或把上游代码放在仓库根的 nfs-binaural/ 下。")
SR = 48000
POS_RATE = 120

# The official modules are only importable with the repo as cwd/on sys.path.
os.chdir(str(REPO))
sys.path.insert(0, str(REPO))
from dataset.loader import load_txt            # noqa: E402
from networks.nfs import get_inverse_window    # noqa: E402
from utils import filter_dict, unfold          # noqa: E402


def unfold_batch(x: torch.Tensor, window: torch.Tensor, n_ch: int = 1) -> torch.Tensor:
    """Copy of inference.py:17-21 (unchanged)."""
    taps = window.size(-1)
    x = F.pad(x, (taps // 2, taps // 2), mode="reflect")
    x = unfold(x, taps // 2, n_ch=n_ch)          # (batch*frames, 1, taps)
    return x * window


def fold_batch(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Copy of inference.py:23-31 (unchanged)."""
    taps = window.size(-1)
    n_frames = x.size(0) + 1
    x = rearrange(x, "t c p -> c t p")
    x = x * window
    x = F.fold(x, (n_frames, taps // 2), (n_frames - 1, 1))
    x = x.narrow(2, 1, x.size(2) - 2)
    x = rearrange(x, "c b t p -> c (b t p)")
    return x.transpose(0, 1)


def pad_to_lens(x, p, lens_sec: float):
    """Copy of inference.py:33-42; identical arithmetic, `lens_sec` injectable."""
    x = torch.from_numpy(x).view(1, 1, -1)
    p = p.transpose(0, 1).unsqueeze(0)
    x_lens = int(lens_sec * SR)
    p_lens = int(lens_sec * POS_RATE)
    x_res = x_lens - x.size(-1) % x_lens
    p_res = p_lens - p.size(-1) % p_lens
    x = F.pad(x, (0, x_res))
    p = F.pad(p, (0, p_res))
    return x.float().cuda(), p.float().cuda()


def build_model(ckpt: Path, model_window_ms: float, nch: int, cdim: int):
    import networks.nfs as gen

    nfs = gen.NFS(window_ms=model_window_ms, nch=nch, cdim=cdim)
    n_params = sum(p.view(-1).size()[0] for p in nfs.parameters())
    print(f"num. params: {n_params}")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)["nfs"]
    nfs.load_state_dict(filter_dict(state))
    return nfs.to("cuda:0").eval()


def render(
    root_dir: Path,
    save_dir: Path,
    *,
    lens_sec: float = 1.0,
    model_window_ms: float = 200,
    nch: int = 128,
    cdim: int = 128,
    ckpt: Path | None = None,
    only: list[str] | None = None,
) -> int:
    """Render every `<root_dir>/<subset>/mono.wav` using `<root_dir>/<subset>/tx_positions.txt`."""
    ckpt = ckpt or (REPO / "ckpt" / "nfs_1353.pt")

    nfs = build_model(ckpt, model_window_ms, nch, cdim)

    taps = int(lens_sec * SR)
    a_window = torch.hann_window(taps, periodic=True).cuda().view(1, 1, -1)
    s_window = get_inverse_window(a_window, taps, taps // 2).cuda().view(1, 1, -1)
    p_taps = int(taps / SR * POS_RATE)

    paths = sorted(glob.glob(str(root_dir / "*" / "mono.wav")))
    if only:
        paths = [p for p in paths if Path(p).parent.name in set(only)]
    print(f"lens_sec={lens_sec}  window={taps} samples  hop={taps // 2}  files={len(paths)}")

    for dp_str in tqdm(paths):
        dp = Path(dp_str)
        subset = dp.parent.name                      # <-- pathlib instead of dp.split('/')[-2]
        x, _ = sf.read(str(dp))
        p = load_txt(str(dp.parent / "tx_positions.txt"))   # (time, channel)
        x, p = pad_to_lens(x, p, lens_sec)

        z = unfold_batch(x, a_window)
        p = unfold_batch(p, torch.ones_like(a_window.narrow(-1, 0, p_taps)), n_ch=7)
        out = []
        for b in range(z.size(0)):
            with torch.no_grad():
                out.append(nfs(p.narrow(0, b, 1), z.narrow(0, b, 1))[0])
        y = fold_batch(torch.cat(out, dim=0), s_window).cpu().numpy()

        data_dir = save_dir / subset
        data_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(data_dir / "binauralized.wav"), y, samplerate=SR, subtype="PCM_16")

    print("DONE ->", save_dir)
    return len(paths)


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Windows-safe NFS benchmark inference")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--root_dir", type=str, default=str(REPO / "dataset" / "benchmark" / "testset"))
    ap.add_argument("--save_dir", type=str, default=str(REPO / "benchmark_eval"))
    ap.add_argument("--lens_sec", type=float, default=1.0)
    ap.add_argument("--model_window_ms", type=float, default=200)
    ap.add_argument("--channel", type=int, default=128)
    ap.add_argument("--cdim", type=int, default=128)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--subsets", type=str, default=None, help="comma separated subset names")
    return ap.parse_args()


if __name__ == "__main__":
    args = _cli()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    n = render(
        Path(args.root_dir),
        Path(args.save_dir),
        lens_sec=args.lens_sec,
        model_window_ms=args.model_window_ms,
        nch=args.channel,
        cdim=args.cdim,
        ckpt=Path(args.ckpt) if args.ckpt else None,
        only=args.subsets.split(",") if args.subsets else None,
    )
    print(f"rendered {n} file(s)")
