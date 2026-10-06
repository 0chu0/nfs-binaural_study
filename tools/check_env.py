#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""环境自检：确认解释器、依赖、GPU、权重与数据集是否就绪。

为什么需要它：
    本项目的三类典型失败——① 用了基础解释器（缺 torch/librosa）；
    ② 依赖装错环境（CPU 版 torch 或缺 CUDA）；③ 忘下载数据集/权重——
    都能在跑任何实测前被本脚本一次性暴露，避免浪费时间在中途报错。

用法:
    python tools/check_env.py
    python tools/check_env.py --code-root /path/to/upstream-nfs     # 上游代码不在默认位置时

退出码: 0 = 通过；2 = 未通过（会打印具体缺项与修复建议）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODE = Path(os.environ.get("NFS_REPO", str(REPO_ROOT / "nfs-binaural")))
REQUIRED_MODULES = ("torch", "torchaudio", "soundfile", "einops", "librosa",
                    "numpy", "scipy", "matplotlib", "tqdm", "torchmetrics", "auraloss")


def check_interpreter() -> bool:
    print("=" * 62)
    print("解释器")
    print("=" * 62)
    print(f"sys.executable : {sys.executable}")
    print(f"python version : {sys.version.split()[0]}")
    print(f"sys.prefix     : {sys.prefix}")
    print(f"base_prefix    : {sys.base_prefix}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"是否虚拟环境   : {'是' if in_venv else '否（这是基础解释器！）'}")
    verdict = in_venv
    print(f"解释器判定     : {'PASS 独立虚拟环境' if verdict else 'FAIL 当前是基础解释器，请切到项目 venv'}")
    return verdict


def check_deps() -> bool:
    print()
    print("=" * 62)
    print("依赖 / GPU")
    print("=" * 62)
    all_ok = True
    for mod in REQUIRED_MODULES:
        try:
            m = __import__(mod)
            print(f"  {mod:12s} OK   {getattr(m, '__version__', '?')}")
        except ImportError:
            all_ok = False
            print(f"  {mod:12s} 缺失（ImportError）")
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"\n  cuda available : {cuda}")
        print(f"  cuda version   : {torch.version.cuda}")
        if cuda:
            print(f"  device         : {torch.cuda.get_device_name(0)}")
            print(f"  arch list      : {torch.cuda.get_arch_list()[-3:]}")
        else:
            print("  [warn] CUDA 不可用：推理会退化到 CPU（速度约慢 10 倍以上）")
    except ImportError:
        print("\n  [跳过 GPU 检查] 未安装 torch")
    return all_ok


def check_assets(code_root: Path) -> bool:
    print()
    print("=" * 62)
    print("权重 / 数据")
    print("=" * 62)
    items = {
        "权重 nfs_1353.pt": code_root / "ckpt" / "nfs_1353.pt",
        "测试集 mono.wav": code_root / "dataset" / "benchmark" / "testset" / "subject1" / "mono.wav",
        "测试集 binaural.wav": code_root / "dataset" / "benchmark" / "testset" / "subject1" / "binaural.wav",
        "测试集 tx_positions.txt": code_root / "dataset" / "benchmark" / "testset" / "subject1" / "tx_positions.txt",
        "训练集目录": code_root / "dataset" / "benchmark" / "trainset",
    }
    ok = True
    for label, p in items.items():
        exists = p.exists()
        ok &= exists
        print(f"  {'OK  ' if exists else '缺失'} {label:24s} {p}")
    if not (code_root / "networks" / "nfs.py").is_file():
        ok = False
        print(f"  缺失 上游代码 networks/nfs.py  {code_root}")
    if not ok:
        print("\n  提示：数据集未随仓库分发，下载方式见 README「数据集」一节；")
        print("        权重在官方 release 或本仓库 nfs-binaural/ckpt/ 下。")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NFS 复现环境自检")
    ap.add_argument("--code-root", type=Path, default=DEFAULT_CODE,
                    help="上游代码目录（含 networks/、ckpt/、dataset/）")
    args = ap.parse_args(argv)
    code_root = args.code_root.resolve()

    ok_interp = check_interpreter()
    ok_deps = check_deps()
    ok_assets = check_assets(code_root)

    print()
    print("=" * 62)
    print(f"总判定: {'PASS 可以开始复现' if (ok_interp and ok_deps and ok_assets) else 'FAIL 见上方缺项'}")
    print("=" * 62)
    if not ok_interp:
        print("解释器修复：IDE 里 Add Local Interpreter -> Select existing ->")
        print("            <项目>/<venv>/Scripts/python.exe（Windows）或 <venv>/bin/python（Linux/macOS）")
    return 0 if (ok_interp and ok_deps and ok_assets) else 2


if __name__ == "__main__":
    raise SystemExit(main())
