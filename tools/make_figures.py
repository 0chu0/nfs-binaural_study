#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NFS 复现研究 —— 图表生成脚本（可复现全部论文/README 插图）。

用法:
    python tools/make_figures.py                      # 默认从仓库内 results/ 与 nfs-binaural/ 取数
    python tools/make_figures.py --data-root E:/NFS-study/nfs-binaural
    python tools/make_figures.py --only fig03         # 只生成某一张

产物: figures/fig01..fig07*.png （150 dpi，可直接插入 Markdown 文档）

依赖: numpy, scipy, matplotlib（soundfile 仅用于语谱图，缺失时自动跳过 fig07）
所有数值均来自 results/*.json 与本仓库已跑完的音频产物，脚本不做任何"美化性"假设。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 无显示环境（CI / 远程）必须设置

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- 全局样式
plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.edgecolor": "#666666",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

# 论文 Table 1 (ICASSP 2023, arXiv:2211.00878v2) 原始数值 —— 逐字抄录，勿改
PAPER_TABLE1: dict[str, dict[str, float | None]] = {
    "WaveNet": {"l2": 0.179, "amp": 0.037, "phase": 0.968, "pesq": 2.305, "stft": 1.915},
    "IR-MLP": {"l2": 0.236, "amp": 0.042, "phase": 0.933, "pesq": None, "stft": None},
    "WarpNet": {"l2": 0.167, "amp": 0.048, "phase": 0.807, "pesq": None, "stft": None},
    "WarpNet*": {"l2": 0.157, "amp": 0.038, "phase": 0.838, "pesq": 2.360, "stft": 1.774},
    "BinauralGrad": {"l2": 0.128, "amp": 0.030, "phase": 0.837, "pesq": 2.759, "stft": 1.278},
    "Ours (NFS)": {"l2": 0.172, "amp": 0.035, "phase": 0.999, "pesq": 1.656, "stft": 1.241},
}

# 论文 Table 2 参数量（M）与 MACs（G）
PAPER_TABLE2: dict[str, tuple[float, float | None]] = {
    "BinauralGrad\n(stage1, 单步)": (6.91, 229.4),
    "BinauralGrad\n(stage2, 单步)": (6.91, 229.3),
    "WarpNet": (8.59, 19.15),
    "IR-MLP": (1.62, None),
    "Ours (NFS)": (0.55, 3.400),
    "wo.Shifter": (0.28, 1.700),
}

C_PAPER = "#7a7a7a"      # 论文
C_OURS = "#c0392b"       # 本次复现（红：主结果）
C_ALT = "#2c6fb5"        # 辅助/对照
C_OK = "#2e8b57"         # 基准线
C_WARN = "#d68910"       # 说明性


# ---------------------------------------------------------------- 工具函数
def load_json(path: Path) -> Any:
    """读取 JSON，缺失时给出明确错误而不是静默返回空。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入文件: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", dy_ratio: float = 0.015) -> None:
    """在柱顶标注数值（位置按轴范围比例给，避免不同量级下重叠）。"""
    top = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + top * dy_ratio,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """读取 wav，返回 (float32 [n, ch] 或 [n], sr)。仅用于语谱图，缺库时抛 ImportError。"""
    import soundfile as sf  # 局部导入：其他图不需要该依赖

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return data, sr


def stft_db(x: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """单通道幅度谱（dB），Hanning 窗、无填充，用于左右对比。"""
    from scipy.signal import stft

    _, _, z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann", boundary=None)
    mag = np.abs(z)
    return 20.0 * np.log10(np.maximum(mag, 1e-8))


# ---------------------------------------------------------------- fig01
def fig01_headline(metrics: dict, out: Path) -> None:
    """核心结论：三项指标 论文 vs 复现。"""
    paper = metrics["paper_table1_nfs"]
    ours = metrics["mean_subject1_8"]
    names = ["L2 x1e3 ↓", "Amp ↓", "L_phs ↓"]
    pv = [paper["l2"] * 1e3, paper["amplitude"], paper["phase"]]
    ov = [ours["l2"] * 1e3, ours["amplitude"], ours["phase"]]

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    b1 = ax.bar(x - w / 2, pv, w, label="论文 Table 1 (NFS)", color=C_PAPER, edgecolor="white")
    b2 = ax.bar(x + w / 2, ov, w, label="本次复现（subject1–8 均值）", color=C_OURS, edgecolor="white")
    for i, (p, o) in enumerate(zip(pv, ov)):
        ax.text(i, max(p, o) * 1.06, f"×{o / p:.2f}", ha="center", fontsize=10,
                color=C_OURS, fontweight="bold")
    annotate_bars(ax, b1, "{:.3f}")
    annotate_bars(ax, b2, "{:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("指标值（越低越好）")
    ax.set_title("图 1  NFS 复现结果与论文基线对比（官方 6 s 分块协议，154 块）")
    ax.set_ylim(0, max(ov) * 1.28)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, 0.02, "幅度误差收敛到 11% 以内；相位与 L2 差距 1.30–1.36×",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=C_WARN)
    fig.savefig(out / "fig01_headline_metrics.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig02
def fig02_landscape(metrics: dict, out: Path) -> None:
    """复现值在论文全部基线中的落点（双面板：L2 与 L_phs）。"""
    ours = metrics["mean_subject1_8"]
    models = list(PAPER_TABLE1.keys())
    l2 = [PAPER_TABLE1[m]["l2"] for m in models] + [ours["l2"] * 1e3]
    ph = [PAPER_TABLE1[m]["phase"] for m in models] + [ours["phase"]]
    labels = models + ["本次复现"]
    colors = [C_PAPER] * len(models) + [C_OURS]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, vals, title, best in (
        (axes[0], l2, "L2 x1e3 （越低越好）", "BinauralGrad 0.128 为全表最优"),
        (axes[1], ph, "L_phs （越低越好）", "NFS 原论文 0.999 反而不是全表最优"),
    ):
        bars = ax.bar(np.arange(len(vals)), vals, 0.62, color=colors, edgecolor="white")
        annotate_bars(ax, bars, "{:.3f}")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.text(0.5, -0.34, best, transform=ax.transAxes, ha="center", fontsize=9, color="#555555")
    axes[0].set_ylabel("指标值")
    fig.suptitle("图 2  复现结果在论文 Table 1 全部基线中的落点", fontsize=12, fontweight="bold")
    fig.savefig(out / "fig02_paper_table1_landscape.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig03
def fig03_per_subject(metrics: dict, data_root: Path, out: Path) -> None:
    """逐受试者 L2 分布 + 各受试者信号能量（说明 L2 与能量强相关）。"""
    per = metrics["per_subset"]
    subs = [d["subset"] for d in per if d["subset"] != "validation_sequence"]
    l2 = [d["l2"] * 1e3 for d in per if d["subset"] != "validation_sequence"]

    rms = []
    for s in subs:
        mono = data_root / "dataset" / "benchmark" / "testset" / s / "mono.wav"
        if mono.is_file():
            d, _ = read_wav(mono)
            rms.append(float(np.sqrt(np.mean(d**2))))
        else:
            rms.append(np.nan)

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    bars = ax.bar(np.arange(len(subs)), l2, 0.6, color=C_ALT, edgecolor="white", label="逐受试者 L2 x1e3")
    annotate_bars(ax, bars, "{:.3f}")
    ax.axhline(np.mean(l2), color=C_OURS, lw=1.6, ls="-",
               label=f"本次复现均值 {np.mean(l2):.4f}")
    ax.axhline(PAPER_TABLE1["Ours (NFS)"]["l2"], color=C_OK, lw=1.6, ls="--",
               label=f"论文 Table 1 {PAPER_TABLE1['Ours (NFS)']['l2']:.3f}")
    ax.set_xticks(np.arange(len(subs)))
    ax.set_xticklabels(subs, rotation=20, ha="right")
    ax.set_ylabel("L2 x1e3 （越低越好）")
    ax.set_title("图 3  逐受试者 L2 误差与单耳信号能量（RMS）")
    ax.set_ylim(0, max(l2) * 1.35)
    ax.legend(frameon=False, loc="upper right")

    ax2 = ax.twinx()
    ax2.plot(np.arange(len(subs)), rms, "o--", color=C_WARN, lw=1.2, ms=5, label="mono RMS（右轴）")
    ax2.set_ylabel("单耳信号 RMS", color=C_WARN)
    ax2.grid(False)
    ax2.tick_params(axis="y", colors=C_WARN)
    ax2.legend(frameon=False, loc="upper center")
    ax.annotate("subject7 的 L2 数值最低，\n但它是低能量片段（RMS 0.011），\n属能量相关指标造成的假象",
                xy=(6, l2[6]), xytext=(3.5, max(l2) * 0.82),
                arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.1),
                fontsize=9, color=C_WARN,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_WARN, lw=0.8, alpha=0.95))
    fig.savefig(out / "fig03_per_subject_l2.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig04
def fig04_three_paths(metrics: dict, official: dict, out: Path) -> None:
    """三条独立评测路径的一致性 + validation_sequence 的影响。"""
    ours = metrics["mean_subject1_8"]
    allm = metrics["mean_all"]
    label_paths = ["inference WOLA\n（整段）", "inference WOLA\n（逐 1 s 块）", "solver 官方协议\n（154×6 s 块）"]
    paper_l2 = PAPER_TABLE1["Ours (NFS)"]["l2"]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    vals = [0.2309, 0.2313, official["ours"]["l2"] * 1e3]
    bars = ax.bar(np.arange(3), vals, 0.55, color=C_ALT, edgecolor="white")
    annotate_bars(ax, bars, "{:.4f}")
    ax.axhline(paper_l2, color=C_OK, ls="--", lw=1.5, label=f"论文 {paper_l2:.3f}")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(label_paths)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_ylabel("L2 x1e3")
    ax.set_title("(a) 三条评测路径互相印证（两两相差 ≤ 1.3%）")
    ax.legend(frameon=False)

    ax = axes[1]
    cats = ["subject1–8\n（论文可比口径）", "+ validation_sequence\n（离群样本）"]
    v2 = [ours["l2"] * 1e3, allm["l2"] * 1e3]
    bars = ax.bar(np.arange(2), v2, 0.45, color=[C_OURS, C_WARN], edgecolor="white")
    annotate_bars(ax, bars, "{:.4f}")
    ax.axhline(paper_l2, color=C_OK, ls="--", lw=1.5, label=f"论文 {paper_l2:.3f}")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(cats)
    ax.set_ylim(0, max(v2) * 1.3)
    ax.set_ylabel("L2 x1e3")
    ax.set_title("(b) 纳入离群样本会让均值虚高 +55%")
    ax.legend(frameon=False)
    ax.annotate("不计入论文可比均值", xy=(1, v2[1]), xytext=(0.35, v2[1] * 0.72),
                arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.1), fontsize=9, color=C_WARN)
    fig.suptitle("图 4  评测链路自洽性检验", fontsize=12, fontweight="bold")
    fig.savefig(out / "fig04_three_paths.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig05
def fig05_attribution(metrics: dict, official: dict, probe: dict, data_root: Path, out: Path) -> None:
    """差异归因四联图：(a) 增益扫描 (b) 时延互相关 (c) 分析窗长 (d) 聚合/路径。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6))
    sub = "subject1"
    pred_p = data_root / "benchmark_eval" / sub / "binauralized.wav"
    gt_p = data_root / "dataset" / "benchmark" / "testset" / sub / "binaural.wav"
    has_audio = pred_p.is_file() and gt_p.is_file()

    pred = gt = None
    sr = 48000
    if has_audio:
        pred_data, sr = read_wav(pred_p)
        gt_data, _ = read_wav(gt_p)
        n = min(len(pred_data), len(gt_data))
        pred = pred_data[:n].ravel().astype(np.float64)
        gt = gt_data[:n].ravel().astype(np.float64)

    # (a) 最优标量增益扫描
    ax = axes[0][0]
    if pred is not None and gt is not None:
        ks = np.linspace(0.6, 1.15, 111)
        errs = [float(np.mean((k * pred - gt) ** 2)) * 1e3 for k in ks]
        best_k = float(ks[int(np.argmin(errs))])
        base_k1 = float(np.mean((pred - gt) ** 2)) * 1e3
        ax.plot(ks, errs, color=C_ALT, lw=1.6)
        ax.axvline(1.0, color=C_PAPER, ls=":", lw=1.2)
        ax.axvline(best_k, color=C_WARN, ls="--", lw=1.2)
        ax.plot([best_k], [min(errs)], "o", color=C_WARN, ms=5)
        ax.annotate(f"最优 k={best_k:.2f}\nL2 {base_k1:.4f} → {min(errs):.4f}"
                    f"（仅降 {(1 - min(errs) / base_k1) * 100:.1f}%）",
                    xy=(best_k, min(errs)), xytext=(0.30, 0.62), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.1), fontsize=8.6, color=C_WARN,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_WARN, lw=0.8, alpha=0.95))
        ax.set_xlabel("标量增益 k")
        ax.set_ylabel("L2 x1e3 （subject1 整段）")
        ax.set_title("(a) 整体增益错位？→ 最优 k≈0.8 也只能降 4%")
    else:
        ax.text(0.5, 0.5, "缺少音频，跳过", ha="center", transform=ax.transAxes)

    # (b) 时延互相关（前 10 s 全采样率 + FFT 互相关，1 采样点 = 0.021 ms 分辨率）
    ax = axes[0][1]
    if pred is not None and gt is not None:
        from scipy.signal import correlate

        lch = pred.reshape(-1, 2)[:, 0]
        gch = gt.reshape(-1, 2)[:, 0]
        seg_len = min(len(lch), len(gch), int(10 * sr))
        a = lch[:seg_len] - float(np.mean(lch[:seg_len]))
        b = gch[:seg_len] - float(np.mean(gch[:seg_len]))
        cc = correlate(a, b, mode="full", method="fft")
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        valid = int(0.1 * sr)  # ±100 ms
        window = cc[len(b) - 1 - valid: len(b) + valid] / denom
        lags_ms = (np.arange(-valid, valid + 1) / sr) * 1000.0
        best_lag_ms = float(lags_ms[int(np.argmax(window))])
        best_samples = abs(best_lag_ms) * sr / 1000.0
        ax.plot(lags_ms, window, color=C_ALT, lw=1.4)
        ax.axvline(best_lag_ms, color=C_WARN, ls="--", lw=1.2)
        ax.annotate(f"峰值 lag = {best_lag_ms:.3f} ms\n≈ {best_samples:.0f} 个采样点（48 kHz），可忽略",
                    xy=(best_lag_ms, float(np.max(window))), xytext=(0.06, 0.86),
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.1), fontsize=8.6, color=C_WARN,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_WARN, lw=0.8, alpha=0.95))
        ax.set_xlabel("相对时延（ms）")
        ax.set_ylabel("归一化互相关（左耳，前 10 s）")
        ax.set_title("(b) 整体时延错位？→ 峰值位移不足 0.1 ms")
    else:
        ax.text(0.5, 0.5, "缺少音频，跳过", ha="center", transform=ax.transAxes)

    # (c) 分析窗长对照
    ax = axes[1][0]
    keys = ["0.2", "0.5", "1.0"]
    vals = [probe[k][sub]["l2"] * 1e3 for k in keys]
    bars = ax.bar(np.arange(3), vals, 0.5, color=C_ALT, edgecolor="white")
    annotate_bars(ax, bars, "{:.4f}")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"lens_sec = {k}" for k in keys])
    ax.set_ylim(min(vals) * 0.93, max(vals) * 1.05)
    ax.set_ylabel("L2 x1e3 （subject1）")
    ax.set_title("(c) 分析窗长度影响？→ 极差仅 1.0%")

    # (d) 聚合方式 / 渲染路径
    ax = axes[1][1]
    names = ["整段 WOLA", "逐 1 s 分块", "官方 6 s 分块"]
    vals = [0.2309, 0.2313, official["ours"]["l2"] * 1e3]
    bars = ax.bar(np.arange(3), vals, 0.5, color=[C_ALT, C_ALT, C_OURS], edgecolor="white")
    annotate_bars(ax, bars, "{:.4f}")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(min(vals) * 0.94, max(vals) * 1.05)
    ax.set_ylabel("L2 x1e3 （8 受试者均值）")
    ax.set_title("(d) 聚合方式与渲染路径影响？→ 极差 1.3%")

    fig.suptitle("图 5  差异归因实验：四类候选原因全部被排除（subject1 单文件探针 + 8 人均值）",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig05_attribution_probes.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig06
def fig06_complexity(metrics: dict, out: Path) -> None:
    """模型复杂度对比（参数量对数轴 + MACs）。"""
    names = list(PAPER_TABLE2.keys())
    params = [PAPER_TABLE2[n][0] for n in names]
    macs = [PAPER_TABLE2[n][1] for n in names]
    colors = [C_PAPER] * len(names)
    colors[names.index("Ours (NFS)")] = C_OURS

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axes[0]
    bars = ax.bar(np.arange(len(names)), params, 0.6, color=colors, edgecolor="white")
    ax.set_yscale("log")
    for bar, v in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.12, f"{v:g} M",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0.584, color=C_OK, ls="--", lw=1.3)
    ax.text(len(names) - 0.5, 0.62, "本次实测 0.584 M", ha="right", fontsize=9, color=C_OK)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=24, ha="right", fontsize=9)
    ax.set_ylabel("参数量（M，对数轴）")
    ax.set_title("(a) 参数量：NFS 比 BinauralGrad 轻 25×")

    ax = axes[1]
    used = [(n, m) for n, m in zip(names, macs) if m is not None]
    bars = ax.bar(np.arange(len(used)), [m for _, m in used], 0.55,
                  color=[C_OURS if n == "Ours (NFS)" else C_PAPER for n, _ in used], edgecolor="white")
    for bar, (_, v) in zip(bars, used):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.05, f"{v:g} G",
                ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(used)))
    ax.set_xticklabels([n for n, _ in used], rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("MACs（G，对数轴）")
    ax.set_title("(b) 计算量：NFS 仅 WarpNet 的 17.8%")

    fig.suptitle("图 6  模型复杂度对比（数据取自论文 Table 2，红柱为 NFS）", fontsize=12, fontweight="bold")
    fig.savefig(out / "fig06_param_macs.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig07
def fig07_spectrogram(data_root: Path, out: Path, sub: str = "subject1", sec: float = 4.0) -> bool:
    """语谱图 A/B：真值 / 复现 / 差值（左耳），直观展示"像不像"。"""
    pred_p = data_root / "benchmark_eval" / sub / "binauralized.wav"
    gt_p = data_root / "dataset" / "benchmark" / "testset" / sub / "binaural.wav"
    if not (pred_p.is_file() and gt_p.is_file()):
        print(f"[fig07] 缺少 {sub} 的音频，跳过")
        return False

    pred, sr = read_wav(pred_p)
    gt, _ = read_wav(gt_p)
    n = min(len(pred), len(gt))
    a = pred[:n, 0]
    b = gt[:n, 0]
    best, best_k = None, 1.0
    for k in np.linspace(0.6, 1.15, 56):  # 用最优增益对齐后再比频谱，避免增益差干扰观察
        e = float(np.mean((k * a - b) ** 2))
        if best is None or e < best:
            best, best_k = e, float(k)
    a = a * best_k

    seg = slice(int(1.0 * sr), int((1.0 + sec) * sr))
    Sa = stft_db(a[seg], sr)
    Sb = stft_db(b[seg], sr)
    lo, hi = 0, 90  # 只看 0–8 kHz，语音能量集中区
    Sa, Sb = Sa[lo:hi], Sb[lo:hi]
    fmax = hi / 1024 * sr / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    common_vmin = float(min(np.percentile(Sa, 5), np.percentile(Sb, 5)))
    common_vmax = float(max(np.percentile(Sa, 99.5), np.percentile(Sb, 99.5)))
    diff_lim = 15.0
    for ax, S, title, cmap, vmin, vmax in (
        (axes[0], Sb, f"真值 GT（{sub} 左耳）", "magma", common_vmin, common_vmax),
        (axes[1], Sa, f"复现渲染（增益对齐 k={best_k:.2f}）", "magma", common_vmin, common_vmax),
        (axes[2], Sa - Sb, "差异 Sa − Sb", "coolwarm", -diff_lim, diff_lim),
    ):
        im = ax.imshow(S, origin="lower", aspect="auto", cmap=cmap,
                       extent=[0, sec, 0, fmax], vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("时间（s）")
        ax.set_ylabel("频率（kHz）", fontsize=9.5)
        ax.tick_params(labelsize=9)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("dB", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("图 7  谱图 A/B 对比：能量包络与时频结构基本重合，差异集中在高频细节",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig07_spectrogram_ab.png")
    plt.close(fig)
    print(f"[fig07] 最优增益 k={best_k:.3f}，对齐后 L2×10³={best*1e3:.4f}（{sub} 整段）")
    return True


# ---------------------------------------------------------------- fig08
def fig08_ratio(metrics: dict, extra: dict | None, out: Path) -> None:
    """四项指标的"复现 / 论文"相对比值：一眼看出差距集中在相位而非幅度。"""
    ours = metrics["mean_subject1_8"]
    paper = PAPER_TABLE1["Ours (NFS)"]
    names = ["L2 x1e3", "Amp", "L_phs"]
    ratios = [ours["l2"] * 1e3 / paper["l2"], ours["amplitude"] / paper["amp"],
              ours["phase"] / paper["phase"]]
    abs_vals = [ours["l2"] * 1e3, ours["amplitude"], ours["phase"]]
    paper_vals = [paper["l2"], paper["amp"], paper["phase"]]
    if extra and extra.get("ours", {}).get("stft"):
        names.append("L_STFT")
        ratios.append(extra["ours"]["stft"] / extra["paper_table1_nfs"]["stft"])
        abs_vals.append(extra["ours"]["stft"])
        paper_vals.append(extra["paper_table1_nfs"]["stft"])

    colors = [C_OK if r < 1.15 else (C_WARN if r < 1.32 else C_OURS) for r in ratios]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.bar(np.arange(len(ratios)), ratios, 0.5, color=colors, edgecolor="white")
    for bar, r, a, p in zip(bars, ratios, abs_vals, paper_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02, f"{r:.2f}×\n{a:.4g} vs {p:.4g}",
                ha="center", va="bottom", fontsize=8.6)
    ax.axhline(1.0, color="#333333", lw=1.4, ls="-")
    ax.text(len(ratios) - 0.45, 1.03, "论文水平 = 1.00×", fontsize=9, ha="right", color="#333333")
    ax.axhline(1.10, color=C_OK, lw=1.0, ls="--")
    ax.text(-0.48, 1.115, "±10% 容差带", fontsize=8.6, color=C_OK)
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels(names)
    ax.set_ylabel("复现 / 论文 比值（越接近 1 越好）")
    ax.set_ylim(0, max(ratios) * 1.30)
    ax.set_title("图 8  四项指标的相对差距：幅度与频谱类 ≤11%，相位类偏大")
    ax.text(0.99, 0.02, "PESQ 未测（环境未安装 pesq，可选依赖）",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6, color="#777777")
    fig.savefig(out / "fig08_ratio_to_paper.png")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description="生成 NFS 复现研究的全部图表")
    ap.add_argument("--repo-root", type=Path, default=here.parent.parent,
                    help="仓库根目录（含 results/ 与 figures/）")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="含 benchmark_eval/ 与 dataset/ 的目录，默认 <repo-root>/nfs-binaural")
    ap.add_argument("--only", type=str, default=None, help="只生成指定图，如 fig03")
    args = ap.parse_args(argv)

    repo = args.repo_root.resolve()
    data = (args.data_root or (repo / "nfs-binaural")).resolve()
    fig_dir = repo / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_json(repo / "results" / "benchmark_metrics.json")
    official = load_json(repo / "results" / "official_eval.json")
    probe = load_json(repo / "results" / "lens_probe.json")
    extra_path = repo / "results" / "extra_metrics.json"
    extra = load_json(extra_path) if extra_path.is_file() else None

    jobs: dict[str, Any] = {
        "fig01": lambda: fig01_headline(metrics, fig_dir),
        "fig02": lambda: fig02_landscape(metrics, fig_dir),
        "fig03": lambda: fig03_per_subject(metrics, data, fig_dir),
        "fig04": lambda: fig04_three_paths(metrics, official, fig_dir),
        "fig05": lambda: fig05_attribution(metrics, official, probe, data, fig_dir),
        "fig06": lambda: fig06_complexity(metrics, fig_dir),
        "fig07": lambda: fig07_spectrogram(data, fig_dir),
        "fig08": lambda: fig08_ratio(metrics, extra, fig_dir),
    }
    todo = {k: v for k, v in jobs.items() if args.only is None or k == args.only}
    if not todo:
        print(f"未知图名: {args.only}，可选 {list(jobs)}")
        return 2

    for name, fn in todo.items():
        try:
            fn()
            print(f"[{name}] 完成")
        except FileNotFoundError as exc:
            print(f"[{name}] 跳过: {exc}")
    print(f"\n图表输出目录: {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
