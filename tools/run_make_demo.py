"""
make_demo.py 的 Windows 安全 launcher（不改动官方源码）。

修复点：
  1. Windows 下 glob 返回混合分隔符，导致 make_demo 内 dp.split('/')[-2] 取错 subset，
     这里把 glob 结果统一替换成 posix 分隔符，使其与官方 split('/') 解析一致。
  2. torch 2.14 默认 weights_only=True 会拒读含 optim/epoch 的 ckpt，这里显式传 False。
  3. make_demo 调用系统 ffmpeg 合成视频，本机无 ffmpeg；路由到 imageio-ffmpeg 自带二进制。

用法：
  NFS_DEMO_ONLY=subject1 运行单受试者验证；不设置则处理 testset 全部受试者。
"""
import os
import sys
import glob
import subprocess
import shutil
from pathlib import Path

REPO = Path(os.environ.get("NFS_REPO", str(Path(__file__).resolve().parent.parent / "nfs-binaural"))).resolve()
if not (REPO / "networks" / "nfs.py").is_file():
    raise SystemExit(f"[error] 找不到上游代码目录（应含 networks/nfs.py）: {REPO}\n"
                     f"        请用环境变量 NFS_REPO 指定，或把上游代码放在仓库根的 nfs-binaural/ 下。")
CKPT = REPO / "ckpt" / "nfs_1353.pt"
ROOT = REPO / "dataset" / "benchmark" / "testset"
SAVE = REPO / "demo_video"

ONLY = os.environ.get("NFS_DEMO_ONLY", "").strip()
ONLY = [s for s in ONLY.split(",") if s] if ONLY else None


# --- 修复 1：Windows glob 返回 posix 分隔符 ---
_orig_glob = glob.glob


def _posix_glob(pattern, *a, **k):
    return [p.replace(os.sep, "/") for p in _orig_glob(pattern, *a, **k)]


glob.glob = _posix_glob

if ONLY:
    _real_glob = glob.glob

    def _filtered(pattern, *a, **k):
        res = _real_glob(pattern, *a, **k)
        if "mono.wav" in pattern or "*.wav" in pattern:
            res = [p for p in res if Path(p).parent.name in ONLY]
        return res

    glob.glob = _filtered


# --- 修复 3：ffmpeg 路由到 imageio-ffmpeg 自带二进制 ---
import imageio_ffmpeg

_FF = imageio_ffmpeg.get_ffmpeg_exe()
_orig_call = subprocess.call


def _ff_call(cmd, *a, **k):
    if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "ffmpeg":
        cmd = [_FF] + list(cmd[1:])
    return _orig_call(cmd, *a, **k)


subprocess.call = _ff_call


# --- 修复 4：绕开环境的 bulk-delete 拦截 ---
# make_demo 末尾用 shutil.rmtree(temp) 清理帧图，环境会拦截并抛异常导致脚本中断。
# 这里改成纯 no-op：不删任何文件，避免触发删除拦截；临时帧图目录残留（每受试者约 74MB），
# 等全部视频产出后再单独清理。
def _safe_rmtree(path):
    return None


shutil.rmtree = _safe_rmtree

os.chdir(str(REPO))
sys.path.insert(0, str(REPO))

import make_demo  # noqa: E402  (需要 REPO 在 sys.path 后导入)
from networks.nfs import NFS  # noqa: E402
from utils import filter_dict  # noqa: E402
import torch  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nfs = NFS(window_ms=200, nch=128, cdim=128)
nfs.load_state_dict(
    filter_dict(torch.load(str(CKPT), map_location="cpu", weights_only=False)["nfs"])
)
nfs = nfs.to("cuda:0")

args = make_demo.get_parser().parse_args(
    [
        "--gpu",
        "0",
        "--ckpt",
        str(CKPT),
        "--root_dir",
        str(ROOT),
        "--save_dir",
        str(SAVE),
        "--is_eval_set",
        "--lens_sec",
        "1.0",
    ]
)

print(f"[launcher] ONLY={ONLY}  save_dir={SAVE}")
make_demo.inference(args, nfs)
print("[launcher] DONE")
