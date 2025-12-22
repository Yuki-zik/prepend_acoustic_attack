# 概览

这是论文《Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models》的官方代码库。该工作作为主会论文发表于 EMNLP 2024。

## 摘要
近来 Whisper 等大型语音基础模型被广泛用于自动语音识别 (ASR)。这些系统的词表包含 `<endoftext>` 等“特殊符号”来引导生成。我们展示这些符号可被对抗攻击利用来操控模型行为。本文提出一种简单有效的方法，学习 Whisper `<endoftext>` 符号的通用声学实现：将其预置于任意语音前可促使模型忽略语音、仅转写该符号，从而“静音”模型。实验表明，同一个 0.64 秒的通用对抗音频段能在 97% 以上的语音样本上成功静音目标 Whisper ASR 模型，并且常能迁移到新数据集与任务。该结果揭示 Whisper 对“静音”对抗攻击的脆弱性：既可能被用于绕过语音审核，也可能用于保护私人语音数据。

# 试用

已将全部预训练的通用声学对抗段放在 `./audio_attacks/`。打开 `demo.ipynb` 运行并体验，观察这些攻击如何在未见语音上静音 Whisper。

# 快速开始（运行代码）

以下为复现论文结果的训练、评估与分析示例命令。

论文中的模型命名与代码中的 `model_name` 对照如下：

| 论文中的模型名 | 代码中的 `model_name` |
| --------------- | ------------------- |
| tiny.en | whisper-tiny |
| tiny | whisper-tiny-multi |
| base.en | whisper-base |
| base | whisper-base-multi |
| small.en | whisper-small |
| small | whisper-small-multi |
| medium.en | whisper-medium |
| medium | whisper-medium-multi |

## 环境安装

最新版在 python>=3.10 上测试。

Fork 仓库并 clone：

`git clone https://github.com/<username>/prepend_acoustic_attack`

使用 `environment_py310.yml` 创建 conda 环境并安装依赖：

```
conda env create -f environment_py310.yml
conda env update -f environment_py310.yml
conda activate env_py310
```

旧版在 python>=3.9 上测试，对应环境文件为 `environment.yml`（不支持 canary）。

## 攻击配置的通用参数

所有脚本的参数见 `src/tools/args.py`。主要攻击相关参数：

- `model_name`：要学习通用攻击的 Whisper 模型。
- `attack_method`：声学攻击形式，本论文使用 `audio-raw`。
- `clip_val`：攻击音频段最大幅度（可感知性），论文用 `0.02`。
- `attack_size`：对抗音频帧数，标准为 `10240`，即 16kHz 约 0.64 秒。
- `data_name`：训练/评估所用数据集。训练在验证集上；测试可传列表。
- `task`：`transcribe` 或 `translate`（仅多语模型支持）。
- `language`：源语种，默认 `en`。

## 学习通用预置声学攻击

`train_attack.py` 用于在任意 Whisper 模型上学习通用攻击。可用附加参数：

- `max_epochs`：最大学习轮次。论文配置：tiny(40)、base(40)、small(120)、medium(160)。
- `bs`：训练批大小。
- `save_freq`：保存已学得攻击段的频率。

示例：
`python train_attack.py --model_name whisper-medium-multi --data_name librispeech --attack_method audio-raw --max_epochs 40 --clip_val 0.02 --attack_size 10240 --save_freq 10 --bs 6 --disable_snr --amp`

## 评估通用预置声学攻击

`eval_attack.py` 评估攻击效果，会同时评估“无攻击”（音频不改）与“有攻击”（预置通用对抗段）。报告两项指标：

1. **NSL**（Negative Sequence Length）：模型输出的平均词长取负值；越接近 0 越好。
2. **frac 0**：预测长度为 0 的样本占比，即完全成功静音的比例。

评估可用的附加参数：

- `attack_epoch`：选择评估训练到某个 epoch 的攻击；需与训练时的 `save_freq` 匹配。
- `not-none`：传入后不评估“无攻击”设置。

示例：
`python eval_attack.py \
  --model_name whisper-medium-multi \
  --data_name librispeech \
  --attack_method audio-raw \
  --clip_val 0.02 \
  --attack_size 10240 \
  --attack_epoch 30 \
  --transfer \
  --attack_model_dir \
  ./experiments/librispeech/whisper-medium-multi/transcribe/en/attack_train/audio-raw/attack_size10240/clip_val0.02/prepend_attack_models \
  --not_none
`

/root/autodl-tmp/prepend_acoustic_attack/experiments/librispeech/whisper-medium-multi/transcribe/en/attack_train/audio-raw/attack_size10240/clip_val0.02/prepend_attack_models/epoch30

/root/autodl-tmp/prepend_acoustic_attack/experiments/librispeech/whisper-medium-multi/transcribe/en/attack_train/audio-raw/attack_size10240/clip_val0.02/prepend_attack_models/epoch30/model.th

### 迁移攻击评估

可评估攻击在不同数据集/任务上的迁移：

- `transfer`：传入表示迁移实验。
- `attack_model_dir`：学得的通用攻击段所在模型包装目录（训练时自动创建）。

示例：将 _librispeech_ 上学得的攻击迁移到 _tedlium_：
`python eval_attack.py --model_name whisper-medium --data_name tedlium --attack_method audio-raw --attack_epoch 160 --attack_size 10240 --transfer --attack_model_dir experiments/librispeech/whisper-medium/transcribe/en/attack_train/audio-raw/attack_size10240/clip_val0.02/prepend_attack_models/ --not_none`

示例：将 _librispeech_ 上为 _transcribe_ 任务学得的攻击迁移到 _fleurs_ 法语数据集的 _translate_ 任务：
`python eval_attack.py --model_name whisper-tiny-multi --data_name fleurs --attack_size 10240 --language fr --task translate --attack_method audio-raw --attack_epoch 40 --transfer --attack_model_dir experiments/librispeech/whisper-tiny-multi/transcribe/en/attack_train/audio-raw/attack_size10240/clip_val0.02/prepend_attack_models/ --not_none`

# 引用

如果使用了本代码库或其一部分，请引用相关工作。

EMNLP 2024 “Muting Whisper”：
```bibtex
@inproceedings{raina-etal-2024-muting,
    title = "Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models",
    author = "Raina, Vyas  and
      Ma, Rao  and
      McGhee, Charles  and
      Knill, Kate  and
      Gales, Mark",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.430/",
    doi = "10.18653/v1/2024.emnlp-main.430",
    pages = "7549--7565"
}
```

SLT 2024 相关工作 “Controlling Whisper”：
```bibtex
@misc{raina2024controllingwhisperuniversalacoustic,
      title={Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models}, 
      author={Vyas Raina and Mark Gales},
      year={2024},
      eprint={2407.04482},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2407.04482}, 
}
```
