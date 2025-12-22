# 针对 `python train_attack.py --model_name whisper-medium-multi --data_name librispeech --attack_method audio-raw --max_epochs 40 --clip_val 0.02 --attack_size 10240 --save_freq 10` 的显存对比

下文对比了仓库初始实现（commit `736f5eb`，首次提供完整 raw audio attack 训练）与当前版本在上述训练命令下的显存占用差异，并给出量化估算。

## 初始实现（736f5eb）
* **模型加载**：`load_model` 只支持单一 Whisper 模型；命令中的 `whisper-medium-multi` 会映射为官方 `medium` 权重并唯一驻留在 GPU，不存在额外实例。
* **对抗段开销**：`attack_size=10240` 对应 40 KB（float32），训练期间仅常驻这一份参数。
* **前向/训练内存**：每个样本只会在 GPU 上保留“攻击后音频”这一份（攻击段 + 原始音频）；以 30 s/16 kHz 音频为例，约占 `((16000*30 + 10240) * 4) ≈ 1.96 MB`。

## 当前实现的新增开销
* **模型实例数量不变**：`train_attack.py` 根据 `len(model_name)` 是否大于 1 决定是否加载集成；该命令只含一个模型名，因而仍旧只实例化单个 Whisper，不会引入多模型显存。【F:train_attack.py†L71-L72】【F:src/models/load_model.py†L5-L16】
* **SNR 相关的双份音频**：
  * `_prepend_attack_to_batch` 在 GPU 上同时构造“攻击后音频”和同长度的“干净音频”零向量，为后续 SNR 计算准备。【F:src/attacker/audio_raw/audio_attack_model_wrapper.py†L46-L58】
  * `compute_batch_snr` 在训练循环中每个 batch 都会被调用，随后将这两份张量复制到 CPU 计算 SNR。【F:src/attacker/audio_raw/audio_attack_model_wrapper.py†L56-L58】【F:src/attacker/audio_raw/learn_attack.py†L68-L70】
  * GPU 端音频临时翻倍：继续以 30 s 样本估算，每条音频约 1.96 MB，双份后约 3.92 MB；`bs=16` 时仅音频 buffer 就比初始版本多占 ~31 MB，未计入 mel、梯度等其他开销。
  * CPU 端也会短暂持有一份同样大小的副本用于 SNR，但这部分不会占用 GPU。
* **对抗段参数**：仍为单份 40 KB（与初始一致），可忽略不计。【F:src/attacker/audio_raw/audio_attack_model_wrapper.py†L11-L29】

## 结论
在该训练命令下，相比初始实现，显存主要新增自 **SNR 计算阶段临时保留的干净音频副本**，使得每批音频张量在 GPU 上短暂翻倍；模型数量与对抗参数规模未产生额外 GPU 占用。若需要进一步压缩显存，可考虑只在较低频率的迭代记录 SNR，或在 `_prepend_attack_to_batch` 中将干净副本直接留在 CPU（但需调整后续计算逻辑以避免频繁的 GPU↔CPU 迁移）。
