# 产物清单（复现结果展示）

本仓库附带以下**可直接查看 / 试听 / 验证**的产物，克隆后无需重新训练或下载即可核对复现效果。
所有产物均 < 100 MB/文件，已纳入版本库；仅训练集（trainset）因含 > 100 MB 单文件被排除，获取方式见文末。

## 1. 测试集输入（评估 / 演示基准）
- 路径：`nfs-binaural/dataset/benchmark/testset/`
- 内容：9 名受试者（subject1–8 + validation_sequence）的 `mono.wav`（单耳输入）与 `binaural.wav`（双耳真值），48 kHz / 16-bit。
- 用途：NFS 模型的评估输入与演示源。

## 2. NFS 渲染双耳音频（模型输出）
- 路径：`nfs-binaural/benchmark_eval/`
- 内容：每名受试者 `binauralized.wav`（NFS 渲染结果，48 kHz / 2ch），与测试集真值逐样本时间对齐。
- 用途：A/B 试听、指标计算。

## 3. 演示视频
- 路径：`nfs-binaural/demo_video/`
- 内容：每名受试者 `binauralized.mp4`（h264 + AAC，约 110–120 s），可视化渲染效果。
- 清单：`nfs-binaural/demo_video/manifest.json`

## 4. 交互式试听前端
- 文件：`nfs-binaural/audio_player.html`
- 用途：浏览器打开即见的 A/B 试听页（渲染 vs 真值，逐受试者），面试演示用。

## 5. 量化评测结果
- `results/benchmark_metrics.json` — 逐受试者 + 均值指标（ℓ₂·10³ / Amp / ℒ_phs）
- `results/official_eval.json` — 复刻论文 6 s 分块协议的结果
- `results/extra_metrics.json` — 补充 MRSTFT 同族指标
- `results/lens_probe.json` — 归因探针（增益 / 时延 / 边缘等）

## 6. 对比图表
- `figures/fig01.png` … `fig08.png` — 头条对比、指标全景、逐受试者、三条路径、归因探针、复杂度、谱图 A/B、相对比值。

## 7. 预训练权重
- `nfs-binaural/ckpt/nfs_1353.pt`（6.7 MB，MIT，源自上游 v1.0.0 release）。

## 训练集（未纳入版本库）
- `trainset` 约 1.9 GB，且含 8 个 > 100 MB 的 wav，超出 GitHub 单文件直推硬限，故排除。
- 获取：从 [BinauralSpeechSynthesis v1.0 release](https://github.com/facebookresearch/BinauralSpeechSynthesis/releases/tag/v1.0) 下载并解压，软链 `dataset/benchmark` 指向解压目录即可（详见 `nfs-binaural/dataset/benchmark/README.md`）。
