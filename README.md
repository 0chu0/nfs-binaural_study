# NFS 双耳语音渲染 · 复现研究（Neural Fourier Shift Reproduction Study）

> 对 ICASSP 2023 论文 **《Neural Fourier Shift for Binaural Speech Rendering》** 的独立复现、验证与归因研究。
> 本仓库不是对原项目的搬运，而是在 Windows + RTX 4060 + PyTorch 2.x 环境下重新跑通推理与评测，并对"为何与论文基线存在残余差距"做了归因。

---

## 1. 项目概要

| 项 | 内容 |
| --- | --- |
| 任务 | 由单声道音频 + 声源位置/朝向，渲染出双耳（binaural）语音 |
| 方法 | Neural Fourier Shift（NFS）：参数估计式神经上采样，而非纯生成式合成 |
| 上游实现 | [`jin-woo-lee/nfs-binaural`](https://github.com/jin-woo-lee/nfs-binaural)（MIT License, © 2023 Jin Woo Lee） |
| 论文 | Lee, J. W. & Lee, K. *Neural Fourier Shift for Binaural Speech Rendering.* ICASSP 2023. [arXiv:2211.00878](https://arxiv.org/abs/2211.00878) |
| 预训练权重 | `nfs_1353.pt`（来自上游 v1.0.0 release，已随仓库 `nfs-binaural/ckpt/` 提供） |
| 复现环境 | Windows 11 / RTX 4060 Laptop (CUDA 12.6) / Python 3.10 / torch 2.14.1+cu126 / torchaudio 2.11 |
| 许可 | 代码沿用上游 **MIT**；数据集来自 Meta `BinauralSpeechSynthesis`（其许可请自行核对） |

**一句话结论**：推理链路完全跑通，9 名受试者双耳音频渲染可用；在论文官方 6 s 评测协议下，主体指标 ℓ₂·10³、Amp、ℒ_phs 分别达到论文基线的 **1.36× / 1.11× / 1.30×**——幅度误差已收敛到 11% 以内。残余差距最合理归因于"公开权重快照与论文报告权重不同"，而非代码缺陷。

---

## 2. 方法概述（与论文一致的部分）

NFS 把双耳渲染建模为**参数估计**问题：用神经网络估计一组可解释的参数（傅里叶偏移量 + 抗混叠滤波系数），再用解析方式合成双耳信号。相比纯卷积生成式方法，它不局限于训练集分布，更适合端侧推理。

- **神经傅里叶偏移（Neural Fourier Shift）**：在频域用学习到的偏移量做亚像素级频移，配合**可学习抗混叠滤波器**抑制混叠。
- **正弦编码（sinusoidal encoding）**：编码维度 128，用于位置/朝向条件注入。
- **损失函数**（论文 Eq.3）：

  ```
  L = λ1·‖x̂ − y‖₂        (L_wav, 波形 ℓ₂)
    + λ2·‖∠X̂ − ∠Y‖₁      (L_phs, 相位差)
    + λ3·‖IID(x̂) − IID(y)‖₂  (L_IID, 内耳图差异)
    + λ4·MRSTFT(x̂, y)    (L_STFT, 多分辨率短时傅里叶)
  λ1 = 10³, λ3 = 10¹, λ2 = λ4 = 1
  ```

- **训练配置**（论文 §3）：
  - 优化器 **RAdam**，16 epoch，batch size 6
  - 每 batch 随机采样 800 ms 单声道音频 + 对应位置/朝向条件
  - 帧长 200 ms、跳步 100 ms；800 ms 输入被 pad 并切分为 **9 帧**，各帧作为独立 batch 在 batch 维堆叠后单次前向
  - 学习率初始化 1e-3，每 epoch 乘 0.9 衰减

---

## 3. 使用的数据集

- **Binaural Speech Synthesis dataset v1.0**（[facebookresearch/BinauralSpeechSynthesis](https://github.com/facebookresearch/BinauralSpeechSynthesis/releases/tag/v1.0)）
- 约 **2 小时**、**48 kHz** 采样率的单声道-双耳配对音频
- 双耳信号由 **KEMAR 假人头**在普通房间（regular room）录制
- 本项目仅用其 `benchmark` 子集（testset 的 8 名受试者 + 1 个 `validation_sequence` 离群样本）

> 数据集体积较大且受第三方许可约束，**不随本仓库推送**。克隆后请按上游说明用符号链接挂接：
> ```bash
> ln -s /path/to/binaural_dataset dataset/benchmark   # 需含 testset/ 与 trainset/
> ```

---

## 4. 复现成果与结论（与论文基线对比）

### 4.1 对比表（论文 Table 1 vs 本复现）

| 指标 | 论文 "Ours (NFS)" | 本复现 (subject1–8 均值, 官方 6 s 协议) | 比值 (本/论文) |
| --- | --- | --- | --- |
| ℓ₂ · 10³ ↓ | **0.172** | **0.2338** | 1.36× |
| Amplitude ↓ | **0.035** | **0.0388** | 1.11× |
| ℒ_phs ↓ | **0.999** | **1.2947** | 1.30× |
| PESQ ↑ | 1.656 | 未复现（环境依赖未满足） | — |
| ℒ_STFT ↓ | 1.241 | 1.298（auraloss 默认 MRSTFT，论文未公布超参，仅同族对照） | 1.05× |
| 参数量 | 0.55 M | 0.584 M（实测） | ≈ 1.06× |
| 计算量 | 3.400 G | — | — |

### 4.2 关键说明

1. **ℓ₂ 不能逐受试者横向比较**：它是能量相关指标，subject7 因片段能量极低（RMS≈0.011）数值"优于"论文，但听感未必更好；只有整体均值才有意义。
2. **`validation_sequence` 不计入论文可比均值**：它是离群补充样本，纳入后均值 ℓ₂ 飙到 0.3622（2.11×），纯属口径混淆，必须剔除。
3. **三条独立评测路径收敛**：整段 WOLA（0.2309）、逐 1 s 分块（0.2313）、官方 6 s 分块（0.2338）两两相差 ≤1.3%，说明差距稳定、可复现，不是评测口径或随机性造成的。

### 4.3 残余差距归因（为何未完全达到论文指标）

逐项排除后，以下假设**均被证伪**（详见 `论文.md` 与 `figures/fig05_attribution_probes.png`）：

| 假设 | 检验结果 |
| --- | --- |
| 整体增益/电平错位（lag/gain） | 最优标量增益 k≈0.89 仅降 4%；互相关峰值位移 < 0.1 ms，可忽略 |
| 边缘效应 / 分块边界 | 1 s / 6 s 分块结果一致 |
| 评测协议差异（WOLA vs 论文 6 s） | 三条路径收敛，口径不是主因 |
| 指标实现口径（PhaseLoss ignore_below 0.1 vs 0.2） | 已对齐到官方 `compute_metrics` |
| 帧长 lens_sec (1.0 / 0.5 / 0.2) | 改变帧长 ℓ₂ 几乎不变（probe 实测） |

**剩余最合理的客观因素**：公开权重 `nfs_1353.pt`（上游 v1.0.0 release）与论文报告所用的权重快照不完全一致。这是**不可达的客观因素**——公开发布的检查点即为此版本，复现者无法获得论文训练末端的精确权重。

### 4.4 是否"完全照搬论文方法"？

否。本仓库在跑通原方法的基础上，做了以下**自主研究**动作（详见第 5 节改动点）：

- 修复 Windows 平台 `glob` 混合分隔符导致的路径解析崩溃；
- 适配 torchaudio 2.11 移除 `ta.functional.istft` 的 API 变更；
- 用三条独立评测路径交叉验证结果稳定性；
- 设计增益/时延/边缘/帧长探针定位残余差距；
- 补充 MRSTFT 同族指标作交叉印证。

---

## 5. 相对论文方法的改动点（明确标注）

| # | 文件 | 改动 | 性质 |
| --- | --- | --- | --- |
| 1 | `tools/run_inference.py` | 用 `pathlib` 替换 `dp.split('/')[-2]` 解析受试者名，修复 Windows 混合分隔符崩溃 | 平台适配（原逻辑等价） |
| 2 | `nfs-binaural/evaluate.py` (L8, L129) | 补 `import torchaudio`；`torch.istft(torch.view_as_complex(...))` 替代已删除的 `ta.functional.istft` | API 适配（数值等价） |
| 3 | `tools/run_official_eval.py` | 复刻论文 6 s 分块评测协议（去掉 DDP），与原 `solver.test` 对照 | 自主评测脚本 |
| 4 | `tools/run_metrics.py` | 调用官方 `evaluate.compute_metrics`（PhaseLoss ignore_below=0.2）保证口径一致 | 口径对齐 |
| 5 | `tools/run_extra_metrics.py` | 补充 MRSTFT（auraloss 默认）同族指标 | 自主扩展 |
| 6 | `tools/make_figures.py` | 生成 8 张对比/归因图表 | 自主可视化 |
| 7 | `nfs-binaural/audio_player.html` | 双耳渲染 A/B 试听前端（渲染 vs GT，逐受试者） | 自主前端 |

---

## 6. 目录结构

```
nfs-binaural_study/
├── README.md                  # 本文件
├── 论文.md                    # 完整复现技术报告（含图表）
├── .gitignore
├── nfs-binaural/              # 上游代码（MIT）+ 适配补丁 + 预训练权重
│   ├── networks/nfs.py        # NFS 网络定义（未改）
│   ├── inference.py / evaluate.py / solver.py / make_demo.py / ...
│   ├── dataset/loader.py      # 数据集加载（未改）
│   ├── ckpt/nfs_1353.pt       # 预训练权重（6.7 MB）
│   ├── audio_player.html      # 试听前端
│   ├── requirements.txt       # 上游依赖（torch 1.12 原版）
│   └── LICENSE                # MIT
├── tools/                    # 本研究的复现/评测/可视化脚本
│   ├── run_inference.py       # Windows 安全推理入口
│   ├── run_metrics.py         # 协议对齐的指标
│   ├── run_official_eval.py   # 论文 6 s 分块协议
│   ├── run_extra_metrics.py   # MRSTFT 补充指标
│   ├── run_make_demo.py       # 演示视频生成
│   ├── check_env.py           # 环境自检
│   └── make_figures.py        # 图表生成
├── results/                  # 评测结果（json）
│   ├── benchmark_metrics.json
│   ├── official_eval.json
│   ├── extra_metrics.json
│   └── lens_probe.json
└── figures/                  # 8 张 PNG 图表
```

---

## 7. 复现步骤

```bash
# 0) 准备环境（本仓库实测于 torch 2.14.1+cu126；原版 requirements 锁 torch 1.12）
python -m venv venv && venv/Scripts/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install soundfile librosa matplotlib einops torchmetrics auraloss numpy scipy tqdm pypdf

# 1) 数据集：testset 已随仓库内置；仅训练需额外下载 trainset 并软链
#    （获取方式见 nfs-binaural/dataset/benchmark/README.md）
ln -s /path/to/binaural_dataset dataset/benchmark   # 仅训练时需要（提供 trainset）

# 2) 推理：渲染 9 个受试者的双耳音频 → nfs-binaural/benchmark_eval/<sub>/binauralized.wav
python tools/run_inference.py --ckpt nfs-binaural/ckpt/nfs_1353.pt

# 3) 指标（协议对齐到官方 compute_metrics）
python tools/run_metrics.py            # 写入 results/benchmark_metrics.json

# 4) 论文 6 s 分块协议交叉验证
python tools/run_official_eval.py      # 写入 results/official_eval.json

# 5) 试听前端：用浏览器打开 nfs-binaural/audio_player.html（需先有 benchmark_eval/ 与 dataset/benchmark/testset/）
```

> **注意**：`nfs-binaural/audio_player.html` 使用相对路径引用 `benchmark_eval/` 与 `dataset/benchmark/testset/`。本地演示请直接双击打开（支持拖动进度条）；或在该目录起本地服务 `python -m http.server 8765`。

---

## 8. 复现环境说明（与上游的差异）

- 上游 `requirements.txt` 锁定 `torch==1.12.1+cu116`。本研究在 **torch 2.14.1+cu126** 上跑通，二者 API 有差异：
  - torchaudio 2.11 删除了 `ta.functional.istft`，已在 `evaluate.py` 适配为 `torch.istft(view_as_complex(...))`；
  - 核心推理 `inference.py` 使用自定义 COLA 反窗（`get_inverse_window`），不依赖该 API，数值不受影响。
- 数据集与预训练权重为第三方资源，请各自核对许可。

---

## 9. 引用

```bibtex
@inproceedings{lee2023neural,
  title={Neural fourier shift for binaural speech rendering},
  author={Lee, Jin Woo and Lee, Kyogu},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

---

## 10. 许可证

- 代码（含本研究的 `tools/` 与适配补丁）：沿用上游 **MIT License**（见 `nfs-binaural/LICENSE`）。
- 论文 © 2023 IEEE；数据集 © Meta（facebookresearch/BinauralSpeechSynthesis），请按各自许可使用。
