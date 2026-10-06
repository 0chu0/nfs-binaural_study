"""
Official-protocol NFS benchmark evaluation (Table 1 reproduction).

Mirrors `solver.Solver.test(mode='test')` exactly, minus the DDP boilerplate
(`test.py` spawns processes via `mp.spawn` + nccl and imports the network from a
copied `results.<result_dir>.codes` package, which is not runnable as-is here).

What the official test path actually does — and why it differs from inference.py:
  * dataset  : `Testset(mode='test', lens_sec=args.test_lens_sec)` with
               `test_lens_sec` defaulting to 6.0  -> 6 s chunks
  * forward  : `est = self.nfs(pos, dry)[0]`  -> NFS is fed each raw 6 s chunk
               directly (its own internal 200 ms windowing / overlap-add)
  * metrics  : `compute_metrics(est, wet)` -> l_2 = 1e3 * scr["l2"]
  * aggregate: L_2_EVAL.append(l_2 * batch_size) ... / num_samples
               (batch_size == 1, so a plain per-chunk mean)

`inference.py` instead renders with an outer 1 s Hann analysis window plus the
custom COLA inverse window (`get_inverse_window`). That is the path used for the
released demo audio, and it is what `run_inference.py` reproduces.

Usage:
    python E:/NFS-study/run_official_eval.py
    python E:/NFS-study/run_official_eval.py --test_lens_sec 6.0 --max_chunks 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO = Path(os.environ.get("NFS_REPO", str(Path(__file__).resolve().parent.parent / "nfs-binaural"))).resolve()
if not (REPO / "networks" / "nfs.py").is_file():
    raise SystemExit(f"[error] 找不到上游代码目录（应含 networks/nfs.py）: {REPO}\n"
                     f"        请用环境变量 NFS_REPO 指定，或把上游代码放在仓库根的 nfs-binaural/ 下。")
OUT_JSON = Path(__file__).resolve().parent.parent / "results" / "official_eval.json"
PAPER = {"l2": 0.172e-3, "amplitude": 0.035, "phase": 0.999}

os.chdir(str(REPO))
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(REPO / "ckpt" / "nfs_1353.pt"))
    ap.add_argument("--test_lens_sec", type=float, default=6.0)
    ap.add_argument("--max_chunks", type=int, default=0, help="0 = all chunks")
    args = ap.parse_args()

    import dataset.loader as module
    from evaluate import compute_metrics
    from utils import filter_dict

    print(f"building Testset(mode='test', lens_sec={args.test_lens_sec}) ...")
    testset = module.Testset(mode="test", lens_sec=args.test_lens_sec)
    print(f"  chunks = {len(testset)}   (dry {tuple(testset.dry.shape)}, "
          f"wet {tuple(testset.wet.shape)}, pos {tuple(testset.pos.shape)})")

    import networks.nfs as gen

    nfs = gen.NFS(window_ms=200, nch=128, cdim=128,
                  wo_ni=False, wo_lff=False, wo_geowarp=False, wo_shifter=False)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["nfs"]
    nfs.load_state_dict(filter_dict(state))
    nfs = nfs.to("cuda:0").eval()
    n_params = sum(p.view(-1).size()[0] for p in nfs.parameters())
    print(f"  num. params: {n_params}")

    n = len(testset) if args.max_chunks <= 0 else min(args.max_chunks, len(testset))
    sums = {"l2": 0.0, "amplitude": 0.0, "phase": 0.0}
    for i in tqdm(range(n)):
        ts = testset[i]
        dry = ts["dry"].cuda(non_blocking=True).float().unsqueeze(0)   # (1,1,W)
        wet = ts["wet"].cuda(non_blocking=True).float().unsqueeze(0)   # (1,2,W)
        pos = ts["pos"].cuda(non_blocking=True).float().unsqueeze(0)   # (1,7,T)
        with torch.no_grad():
            est = nfs(pos, dry)[0]
        scr = compute_metrics(est, wet)
        for k in sums:
            sums[k] += float(scr[k])

    mean = {k: v / n for k, v in sums.items()}
    print()
    print(f"{'metric':>12s} {'ours':>12s} {'paper':>12s} {'ratio':>8s}")
    print("-" * 48)
    print(f"{'L2 x1e3':>12s} {mean['l2'] * 1e3:12.4f} {PAPER['l2'] * 1e3:12.4f} "
          f"{mean['l2'] / PAPER['l2']:7.2f}x")
    print(f"{'Amplitude':>12s} {mean['amplitude']:12.4f} {PAPER['amplitude']:12.4f} "
          f"{mean['amplitude'] / PAPER['amplitude']:7.2f}x")
    print(f"{'Phase':>12s} {mean['phase']:12.4f} {PAPER['phase']:12.4f} "
          f"{mean['phase'] / PAPER['phase']:7.2f}x")
    print(f"\n(chunks={n}, test_lens_sec={args.test_lens_sec})")

    OUT_JSON.write_text(json.dumps(
        {"protocol": "solver.Solver.test(mode='test') without DDP",
         "chunks": n, "test_lens_sec": args.test_lens_sec,
         "ours": mean, "paper_table1_nfs": PAPER}, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
