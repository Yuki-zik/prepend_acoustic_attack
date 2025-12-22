# 新功能带来的显存开销分析

本文梳理近期功能变更在 GPU 显存占用上的影响，并结合代码位置给出量化或估算说明，便于后续优化或排查 OOM。

## 1. 通用前置扰动参数常驻 GPU
* **实现位置**：`AudioAttackModelWrapper` 在初始化时会创建可学习的 `audio_attack_segment`，并直接放置在攻击器所在设备上，默认长度 `attack_size=5120`。【F:src/attacker/audio_raw/audio_attack_model_wrapper.py†L11-L30】
* **显存开销**：以 32-bit float 计，默认 5120 采样点仅占 ~20 KB（`5120 * 4` 字节）。即使增大 `--attack_size`，开销也与长度线性相关，属于常驻参数成本。

## 2. 为计算 SNR 生成的双份音频
* **实现位置**：在前向/推理前，包装器会拼接攻击段并同时构造一个零填充的“干净”副本，用于后续 SNR 计算。【F:src/attacker/audio_raw/audio_attack_model_wrapper.py†L46-L58】【F:src/attacker/audio_raw/audio_attack_model_wrapper.py†L111-L128】
* **显存开销**：这一改动使得每个 batch 暂时持有“攻击后音频 + 干净音频”两份数据，显存约翻倍。以 30 秒、16 kHz 的单通道音频为例，原始张量约 1.9 MB；拼接攻击段后再复制一份会增加到 ~3.9 MB/样本（不含 mel 与模型中间激活）。

## 3. 集成模型（Whisper Ensemble）并行加载
* **实现位置**：`WhisperModelEnsemble` 会同时实例化并保存在 GPU 上的多个 Whisper checkpoint，`load_model` 在传入多模型名称时会走该分支。【F:src/models/whisper.py†L45-L56】【F:src/models/load_model.py†L4-L12】
* **显存开销**：每个 Whisper 模型的参数和缓存都会常驻 GPU，显存需求近似按模型个数线性增长。新增的多模型攻击/评估能力意味着要为每个额外模型预留与单模型同量级的显存（例如 base/ small/ medium 分别约需 1–5 GB，可按具体权重估算）。

## 4. 结论与建议
* 核心新增开销来自 **SNR 计算时的双份音频** 与 **集成模型的多实例**；前置扰动参数成本相对可忽略。
* 若遇到显存紧张，可考虑：
  * 仅在需要 SNR 时再生成干净副本，或在 CPU 上暂存；
  * 在 CLI 里将 `--dtype` 设为 `float16`/`bfloat16`，让 Whisper 以半精度加载并在攻击封装层中匹配精度（减少权重与激活占用）；
  * 使用更小的 Whisper 版本或减少集成模型数量；
  * 控制 `attack_size` 以防止不必要的增长。
