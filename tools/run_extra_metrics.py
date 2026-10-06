#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NFS 复现 —— 论文 Table 1 另外两列（L_STFT / PESQ）的补充评测。

为什么需要这个脚本：
    论文 Table 1 共 5 列（L2 / Amp / L_phs / PESQ / L_STFT），官方仓库自带的
    `evaluate.compute_metrics` 只覆盖前 3 列。本脚本沿用**完全相同的 6 s 分块协议**
    与同一份公开权重，把后 2 列也测出来，使对比表口径完整。

口径声明（务必随结果一起引用）：
    * L_STFT：auraloss.freq.MultiResolutionSTFTLoss（多分辨率 STFT 损失）。
      论文未公布 MRSTFT 的 fft/hop/win 超参，故此处使用 auraloss 默认参数，
      属"同族指标、自建口径"，只用于量级对照，不等于论文原始数值。
    * PESQ：论文未说明声道处理方式；此处按 ITU-T P.862 宽带模式（16 kHz）对
      L/R 下混的单声道计算。若环境未安装 `pesq`，该项自动跳过。

用法:
    "<venv>/python.exe" tools/run_extra_metrics.py
    "<venv>/python.exe" tools/run_extra_metrics.py --max_chunks 20   # 快速自检
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CODE = REPO_ROOT / "nfs-binaural"
OUT_JSON = REPO_ROOT / "results" / "extra_metrics.json"
PAPER = {"l2": 0.172e-3, "amplitude": 0.035, "phase": 0.999, "pesq": 1.656, "stft": 1.241}


def try_import_pesq() -> Any | None:
    """PESQ 为可选依赖；缺失时返回 None 而不是让整个脚本失败。"""
    try:
        from pesq import pesq as pesq_fn  # type: ignore
    except ImportError:
        print("[warn] 未安装 `pesq`，跳过 PESQ 列（pip install pesq 后可补齐）")
        return None
    return pesq_fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="补充评测 L_STFT / PESQ 两列")
    ap.add_argument("--test_lens_sec", type=float, default=6.0)
    ap.add_argument("--max_chunks", type=int, default=0, help="0 = 全部 154 块")
    ap.add_argument("--code-root", type=Path, default=REPO_CODE,
                    help="上游代码所在目录（需含 dataset/、networks/、ckpt/）")
    ap.add_argument("--ckpt", type=str, default=None, help="默认 <code-root>/ckpt/nfs_1353.pt")
    args = ap.parse_args(argv)

    code_root = args.code_root.resolve()
    if not (code_root / "networks" / "nfs.py").is_file():
        print(f"[error] --code-root 下找不到上游代码: {code_root}")
        return 2
    ckpt = Path(args.ckpt) if args.ckpt else code_root / "ckpt" / "nfs_1353.pt"
    if not ckpt.is_file():
        print(f"[error] 找不到权重: {ckpt}（请先下载官方 nfs_1353.pt）")
        return 2

    os.chdir(str(code_root))
    sys.path.insert(0, str(code_root))

    import dataset.loader as module
    import networks.nfs as gen
    from evaluate import compute_metrics
    from utils import filter_dict

    from auraloss.freq import MultiResolutionSTFTLoss

    testset = module.Testset(mode="test", lens_sec=args.test_lens_sec)
    n = len(testset) if args.max_chunks <= 0 else min(args.max_chunks, len(testset))
    print(f"chunks = {n} (out of {len(testset)}), lens_sec = {args.test_lens_sec}")

    nfs = gen.NFS(window_ms=200, nch=128, cdim=128,
                  wo_ni=False, wo_lff=False, wo_geowarp=False, wo_shifter=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["nfs"]
    nfs.load_state_dict(filter_dict(state))
    nfs = nfs.to("cuda:0").eval()
    print(f"num. params = {sum(p.view(-1).size()[0] for p in nfs.parameters())}")

    mr_stft = MultiResolutionSTFTLoss().to("cuda:0")
    stft_cfg = {
        "fft_sizes": list(getattr(mr_stft, "fft_sizes", [])),
        "hop_sizes": list(getattr(mr_stft, "hop_sizes", [])),
        "win_lengths": list(getattr(mr_stft, "win_lengths", [])),
    }

    pesq_fn = try_import_pesq()
    sums = {"l2": 0.0, "amplitude": 0.0, "phase": 0.0, "stft": 0.0, "pesq": 0.0}
    n_pesq = 0

    for i in tqdm(range(n)):
        ts = testset[i]
        dry = ts["dry"].cuda(non_blocking=True).float().unsqueeze(0)
        wet = ts["wet"].cuda(non_blocking=True).float().unsqueeze(0)
        pos = ts["pos"].cuda(non_blocking=True).float().unsqueeze(0)
        with torch.no_grad():
            est = nfs(pos, dry)[0]

        scr = compute_metrics(est, wet)
        for k in ("l2", "amplitude", "phase"):
            sums[k] += float(scr[k])

        est_b = est.unsqueeze(0) if est.dim() == 2 else est
        wet_b = wet if wet.dim() == 3 else wet.unsqueeze(0)
        if est_b.shape[-1] != wet_b.shape[-1]:  # 极少数块会差 1 个采样点
            m = min(est_b.shape[-1], wet_b.shape[-1])
            est_b, wet_b = est_b[..., :m], wet_b[..., :m]
        sums["stft"] += float(mr_stft(est_b, wet_b))

        if pesq_fn is not None:
            import numpy as np
            import torchaudio as ta

            mix_e = est_b.mean(dim=1).squeeze(0)   # L/R 下混
            mix_w = wet_b.mean(dim=1).squeeze(0)
            resampler = ta.transforms.Resample(48000, 16000).to(mix_e.device)
            e16 = resampler(mix_e).clamp(-1.0, 1.0).cpu().numpy()
            w16 = resampler(mix_w).clamp(-1.0, 1.0).cpu().numpy()
            try:
                sums["pesq"] += float(pesq_fn(16000, w16, e16, "wb"))
                n_pesq += 1
            except Exception as exc:  # pesq 对极短/静音段会抛错，单独计数并跳过
                print(f"[warn] chunk {i} PESQ 失败: {exc}")

    mean = {
        "l2": sums["l2"] / n,
        "amplitude": sums["amplitude"] / n,
        "phase": sums["phase"] / n,
        "stft": sums["stft"] / n,
        "pesq": (sums["pesq"] / n_pesq) if n_pesq else None,
    }

    print(f"\n{'metric':>12s} {'ours':>12s} {'paper':>12s}")
    print("-" * 40)
    for k, label in (("l2", "L2 x1e3"), ("amplitude", "Amplitude"), ("phase", "Phase"),
                     ("stft", "L_STFT"), ("pesq", "PESQ")):
        ours = mean[k] * 1e3 if k == "l2" else mean[k]
        paper = PAPER[k] * 1e3 if k == "l2" else PAPER[k]
        txt = f"{ours:12.4f}" if ours is not None else f"{'-':>12s}"
        print(f"{label:>12s} {txt} {paper:12.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "protocol": "solver.Solver.test(mode='test') 同口径 6 s 分块（无 DDP）",
        "chunks": n,
        "test_lens_sec": args.test_lens_sec,
        "stft_config_auraloss_default": stft_cfg,
        "pesq_chunks_ok": n_pesq,
        "ours": mean,
        "paper_table1_nfs": PAPER,
        "notes": [
            "L_STFT 用 auraloss 默认 MRSTFT 参数，论文未公布其超参，故为同族指标对照。",
            "PESQ 为 L/R 下混后 16 kHz 宽带口径；论文未说明声道处理方式。",
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
