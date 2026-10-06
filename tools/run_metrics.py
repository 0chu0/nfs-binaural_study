"""
NFS benchmark metrics — protocol-parity version.

Calls the repo's OFFICIAL entry point `evaluate.compute_metrics()` instead of
hand-building the Loss objects, so the protocol is guaranteed to match the repo
(notably PhaseLoss is created with ignore_below=0.2 inside compute_metrics,
whereas hand-construction defaults to 0.1 — that silently shifts the numbers).

Compares:
    rendered : nfs-binaural/benchmark_eval/<subset>/binauralized.wav
    ground-truth : nfs-binaural/dataset/benchmark/testset/<subset>/binaural.wav

Aggregates two ways:
    subject1..subject8 -> the 8 subjects used in the paper (Table 1)
    all                -> includes validation_sequence (NOT part of the paper)

Usage:
    python tools/run_metrics.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(os.environ.get("NFS_REPO", str(Path(__file__).resolve().parent.parent / "nfs-binaural"))).resolve()
if not (REPO / "networks" / "nfs.py").is_file():
    raise SystemExit(f"[error] 找不到上游代码目录（应含 networks/nfs.py）: {REPO}\n"
                     f"        请用环境变量 NFS_REPO 指定，或把上游代码放在仓库根的 nfs-binaural/ 下。")
PRED = REPO / "benchmark_eval"
REF = REPO / "dataset" / "benchmark" / "testset"
OUT_JSON = Path(__file__).resolve().parent.parent / "results" / "benchmark_metrics.json"

# Paper: "Neural Fourier Shift for Binaural Speech Rendering" (ICASSP 2023), Table 1, row "Ours (NFS)"
PAPER = {"l2": 0.172e-3, "amplitude": 0.035, "phase": 0.999}


def main() -> int:
    os.chdir(str(REPO))
    sys.path.insert(0, str(REPO))
    from evaluate import compute_metrics  # noqa: E402  (needs chdir/sys.path first)

    if not PRED.is_dir():
        print(f"[ERROR] rendered dir not found: {PRED}")
        return 1

    rows: list[tuple[str, float, float, float]] = []
    for sub in sorted(d.name for d in PRED.iterdir() if d.is_dir()):
        pred_p = PRED / sub / "binauralized.wav"
        ref_p = REF / sub / "binaural.wav"
        if not (pred_p.exists() and ref_p.exists()):
            print(f"{sub:20s}  [skip] missing pair")
            continue
        yp, _ = sf.read(str(pred_p))
        yr, _ = sf.read(str(ref_p))
        n = min(len(yp), len(yr))          # rendered is zero-padded to a whole second
        yp, yr = yp[:n], yr[:n]
        if yp.ndim == 1:
            yp = yp[:, None]
        if yr.ndim == 1:
            yr = yr[:, None]
        t_pred = torch.from_numpy(yp).float().transpose(0, 1).unsqueeze(0)   # (1, C, T)
        t_ref = torch.from_numpy(yr).float().transpose(0, 1).unsqueeze(0)
        m = compute_metrics(t_pred, t_ref)
        rows.append((sub, float(m["l2"]), float(m["amplitude"]), float(m["phase"])))

    if not rows:
        print("[ERROR] no comparable pairs found")
        return 1

    print(f"{'subset':20s} {'L2x1e3':>10s} {'Amp':>10s} {'Phase':>10s}")
    print("-" * 54)
    for sub, l2, amp, phs in rows:
        print(f"{sub:20s} {l2 * 1e3:10.4f} {amp:10.4f} {phs:10.4f}")

    def mean(sel: list[tuple[str, float, float, float]]) -> tuple[float, float, float]:
        k = len(sel)
        return (
            sum(r[1] for r in sel) / k,
            sum(r[2] for r in sel) / k,
            sum(r[3] for r in sel) / k,
        )

    subjects = [r for r in rows if r[0].startswith("subject")]
    s_l2, s_amp, s_phs = mean(subjects)
    a_l2, a_amp, a_phs = mean(rows)

    print("-" * 54)
    print(f"{'MEAN(subject1-8)':20s} {s_l2 * 1e3:10.4f} {s_amp:10.4f} {s_phs:10.4f}")
    print(f"{'MEAN(all 9)':20s} {a_l2 * 1e3:10.4f} {a_amp:10.4f} {a_phs:10.4f}")
    print(f"{'PAPER (Table 1)':20s} {PAPER['l2'] * 1e3:10.4f} {PAPER['amplitude']:10.4f} {PAPER['phase']:10.4f}")
    print()
    print("ratios vs paper (ours/paper, 1.00 == exact match):")
    print(f"  subject1-8 : L2 {s_l2 / PAPER['l2']:.2f}x   Amp {s_amp / PAPER['amplitude']:.2f}x   "
          f"Phase {s_phs / PAPER['phase']:.2f}x")
    print(f"  all 9      : L2 {a_l2 / PAPER['l2']:.2f}x   Amp {a_amp / PAPER['amplitude']:.2f}x   "
          f"Phase {a_phs / PAPER['phase']:.2f}x")

    payload = {
        "protocol": "evaluate.compute_metrics (PhaseLoss ignore_below=0.2)",
        "per_subset": [
            {"subset": s, "l2": l2, "amplitude": amp, "phase": phs} for s, l2, amp, phs in rows
        ],
        "mean_subject1_8": {"l2": s_l2, "amplitude": s_amp, "phase": s_phs},
        "mean_all": {"l2": a_l2, "amplitude": a_amp, "phase": a_phs},
        "paper_table1_nfs": PAPER,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
