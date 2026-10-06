# 数据集说明

完整数据集（含 `trainset` + `testset`）来自 [facebookresearch/BinauralSpeechSynthesis](https://github.com/facebookresearch/BinauralSpeechSynthesis) 的 **v1.0 release**（KEMAR 双耳语音基准，约 2 小时、48 kHz）。

## 本仓库已包含
- `testset/`（9 名受试者，~294 MB）：评估与演示用输入 / 真值，已纳入版本库，克隆即可用。

## 训练集（trainset）获取
`trainset/` 约 1.9 GB，含多个 > 100 MB 的 wav，超出 GitHub 单文件直推硬限，未纳入版本库。

```bash
# 1. 从上游 release 下载 binaural_dataset 并解压
#    地址：https://github.com/facebookresearch/BinauralSpeechSynthesis/releases/tag/v1.0
#    解压后得到含 trainset/ 与 testset/ 的目录（记为 /path/to/binaural_dataset）

# 2. 将解压目录软链为本仓库的 dataset/benchmark
ln -s /path/to/binaural_dataset dataset/benchmark
# Windows（管理员 PowerShell）：
# New-Item -ItemType SymbolicLink -Path dataset/benchmark -Target \path\to\binaural_dataset
```

软链后 `dataset/benchmark/trainset` 与 `dataset/benchmark/testset` 同时存在，即可运行训练脚本。
