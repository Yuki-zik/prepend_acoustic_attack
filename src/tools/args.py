import argparse


def core_args():
    # 训练/评估通用参数（模型、数据集、语言等）
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument(
        "--model_name",
        type=str,
        default="whisper-small",
        nargs="+",
        help="ASR model. Can pass multiple models if multiple models to be loaded",   # 目标模型名称，可传入列表以加载集成
    )
    commandLineParser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task. N.b. translate is only X-en",                             # 任务：转写或翻译
    )
    commandLineParser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Source audio language or if performing machine translation do something like fr_en",  # 源语言或源_目标
    )
    commandLineParser.add_argument(
        "--gpu_id", type=int, default=0, help="select specific gpu"                  # 选择 GPU ID
    )
    commandLineParser.add_argument(
        "--data_name",
        type=str,
        default="librispeech",
        nargs="+",
        help="dataset for exps;",                                                    # 数据集名称，可多选
    )
    commandLineParser.add_argument(
        "--use_pred_for_ref", action="store_true", help="Implemented for Fleurs dataset. Use model predictions for the reference transcriptions."  # FLEURS 可选用模型预测作为参考文本
    )
    commandLineParser.add_argument("--seed", type=int, default=1, help="select seed")  # 随机种子
    commandLineParser.add_argument(
        "--force_cpu", action="store_true", help="force cpu use"                     # 强制使用 CPU
    )
    return commandLineParser.parse_known_args()


def attack_args():
    # 对抗攻击相关参数（训练与评估）
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    # train attack args
    commandLineParser.add_argument(
        "--attack_method",
        type=str,
        default="audio-raw",
        choices=["audio-raw", "mel"],
        help="Adversarial attack approach for training",        # 选择攻击空间：原始音频或 mel
    )
    commandLineParser.add_argument(
        "--attack_token",
        type=str,
        default="eot",
        choices=["eot", "transcribe"],
        help="Which non-acoustic token are we learning an acoustic realization for.",  # 针对哪个非声学 token 生成音频
    )
    commandLineParser.add_argument(
        "--attack_command",
        type=str,
        default="mute",
        choices=["mute", "hallucinate", "translate"],
        help="Objective of attack - hidden universal command/control.",               # 攻击目标：静音/幻觉/翻译
    )
    commandLineParser.add_argument(
        "--max_epochs", type=int, default=20, help="Training epochs for attack"       # 训练轮数
    )
    commandLineParser.add_argument(
        "--save_freq", type=int, default=1, help="Epoch frequency for saving attack" # 模型保存频率
    )
    commandLineParser.add_argument(
        "--attack_size", type=int, default=5120, help="Length of attack segment"     # 对抗前置音频长度
    )
    commandLineParser.add_argument(
        "--bs", type=int, default=16, help="Batch size for training attack"          # 批大小
    )
    commandLineParser.add_argument(
        "--lr", type=float, default=1e-3, help="Adversarial Attack learning rate"    # 学习率
    )
    commandLineParser.add_argument(
        "--clip_val",
        type=float,
        default=-1,
        help="Value (maximum) to clip the log mel vectors. -1 means no clipping",     # 对抗向量裁剪阈值，-1 不裁剪
    )
    commandLineParser.add_argument(
        "--attack_init",
        type=str,
        default="random",
        help="How to initialize attack. Give the path of a previously trained attack (model wrapper) if you want to initialize with it",  # 攻击初始化方式或路径
    )

    # eval attack args
    commandLineParser.add_argument(
        "--attack_epoch",
        type=int,
        default=-1,
        help="Specify which training epoch of attack to evaluate; -1 means no attack",  # 评估使用的攻击 epoch，-1 为无攻击
    )
    commandLineParser.add_argument(
        "--force_run",
        action="store_true",
        help="Do not load from cache",                                                 # 忽略缓存强制重算
    )
    commandLineParser.add_argument(
        "--not_none", action="store_true", help="Do not evaluate the none attack"      # 跳过无攻击基线
    )
    commandLineParser.add_argument(
        "--eval_train", action="store_true", help="Evaluate attack on the train split" # 在训练集上评估
    )
    # commandLineParser.add_argument('--only_wer', action='store_true', help='Evaluate only the WER.')
    commandLineParser.add_argument(
        "--eval_metrics",
        type=str,
        default="nsl frac0",
        nargs="+",
        help="Which metrics to evaluate from: asl, frac0, wer, frac_lang",             # 选择评估指标
    )
    commandLineParser.add_argument(
        "--frac_lang_langs",
        type=str,
        default="en fr",
        nargs="+",
        help="Which languages to evaluate for frac_lang metric",                       # 语言占比指标中的语言集合
    )

    # eval attack args for attack transferability
    commandLineParser.add_argument(
        "--transfer",
        action="store_true",
        help="Indicate it is a transferability attack (across model or dataset) for mel whitebox attack",  # 是否为迁移评估
    )
    commandLineParser.add_argument(
        "--attack_model_dir",
        type=str,
        default="",
        help="path to trained attack to evaluate",                                     # 外部攻击权重目录
    )
    return commandLineParser.parse_known_args()


def analysis_args():
    # 分析脚本使用的附加开关与路径
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument(
        "--spectrogram", action="store_true", help="do analysis to generate spectrogram"   # 生成对抗段谱图
    )
    commandLineParser.add_argument(
        "--compare_with_audio", action="store_true", help="Include a real audio file"      # 是否与真实音频对比
    )
    commandLineParser.add_argument(
        "--sample_id",
        type=int,
        default=42,
        help="Specify which data sample to compare to",                                     # 对比样本索引
    )

    commandLineParser.add_argument(
        "--wer_no_0", action="store_true", help="WER of non-zero length samples"            # 仅统计非空输出的 WER
    )
    commandLineParser.add_argument(
        "--no_attack_path",
        type=str,
        default="",
        help="path to predictions with no attack",                                          # 无攻击预测文件
    )
    commandLineParser.add_argument(
        "--attack_path", type=str, default="", help="path to predictions with attack"       # 有攻击预测文件
    )

    commandLineParser.add_argument(
        "--saliency",
        action="store_true",
        help="Do saliency analysis. If you want to get saliency for a transfer attack - use attack transferability arguments and attack_path argument",  # 显著性分析
    )
    commandLineParser.add_argument(
        "--saliency_plot",
        action="store_true",
        help="Plot frame-level saliency across the audio recording.",                       # 绘制帧级显著性
    )
    commandLineParser.add_argument(
        "--model_transfer_check", action="store_true", help="Determine if its possible for a muting attack is to transfer between different target models (passed in core_args.model_names)"  # 检查模型间迁移
    )

    commandLineParser.add_argument(
        "--model_emb_close_exs", action="store_true", help="Print the 10 closest words for target tokens as per the embedding matrix.)"  # 打印嵌入空间最近词
    )


    return commandLineParser.parse_known_args()
