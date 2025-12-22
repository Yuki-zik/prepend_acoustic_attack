# 梯度与显存占用差异分析（当前版本 vs 初始版本）

本文对比当前分支代码与项目初始版本（commit `9567bdf` 中的实现），聚焦训练阶段的梯度逻辑和显存占用路径。

## 梯度逻辑对比

- **前向与损失目标**
  - 初始版本在 `train_step` 中直接调用攻击模型的前向，取最后一步 logits 后以 EOT 概率为目标，未区分不同攻击 token，也未添加额外指标或多模型设置。
  - 现版本仍以末步 logits 为输入，但通过 `_get_tgt_tkn_id` 支持 EOT/Transcribe 等目标，并支持多模型攻击标志，保持同样的交叉熵式最优化方向。【F:src/attacker/audio_raw/learn_attack.py†L15-L55】

- **梯度计算与数值稳定**
  - 初始实现使用单精度反向传播，未做缩放。
  - 现版本在 GPU 上可选混合精度（`amp`），用 `GradScaler` 缩放梯度并在反向后更新缩放因子，以抑制溢出并降低显存占用。【F:src/attacker/audio_raw/learn_attack.py†L37-L66】

- **对抗向量投影/裁剪**
  - 初始版在更新后对对抗段做 `clamp_(0, clip_val)`，仅限制正向幅值。
  - 现版本改为对称裁剪 `clamp_(-clip_val, clip_val)`，保持梯度投影到有界 L∞ 球，避免偏向正值；同时在 `attack_args.clip_val=-1` 时放宽至大范围常数。【F:src/attacker/audio_raw/learn_attack.py†L69-L76】

- **SNR 统计与反传无关逻辑**
  - 初始实现无 SNR 计算，训练循环只跟踪损失。
  - 现版本可选计算批量 SNR（`compute_snr`），在训练中额外前向生成“加扰前/后”音频并在 CPU 侧统计，不影响梯度但会产生额外内存/带宽开销。【F:src/attacker/audio_raw/learn_attack.py†L77-L91】

## 数据加载与显存占用差异

- **数据对齐策略**
  - 初始版直接堆叠原始音频张量，假设长度一致；未使用多进程加载。
  - 现版本在 `_prep_dl` 中按批次最长样本右侧补零，再堆叠为单个大张量，并使用 `num_workers=8`。对于存在极长样本的场景，补零与多进程复制会显著放大内存/显存占用。【F:src/attacker/audio_raw/learn_attack.py†L93-L116】

- **混合精度的显存影响**
  - 初始版全程 FP32；现版本在启用 `--amp` 时，模型与前向激活可降为 FP16（取决于硬件），理论上可减少梯度与激活显存约 30%–50%，但需要额外的缩放器状态，占用较小。【F:src/attacker/audio_raw/learn_attack.py†L37-L66】

- **SNR 计算的显存与带宽开销**
  - 新增的 `compute_batch_snr` 会在训练时额外保留“干净音频”和“攻击音频”两个序列，并将其 `detach().cpu()` 复制到主机进行统计，导致单批显存峰值约翻倍，且多一次 GPU→CPU 拷贝。【F:src/attacker/audio_raw/learn_attack.py†L77-L91】

## 结论

总体来看，当前版本在梯度逻辑上引入了混合精度、对称裁剪和可选的 SNR 统计，强化了数值稳定性与指标监控；显存方面，混合精度有利于降低占用，但批内补零、SNR 双路音频与多进程 DataLoader 会增加内存/显存峰值，需要根据数据长度与硬件配置权衡是否启用或裁剪长音频。
