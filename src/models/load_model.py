from .whisper import WhisperModel, WhisperModelEnsemble  # Whisper 封装
from .canary import CanaryModel                          # Canary 封装
import torch


def load_model(core_args, device=None, load_ensemble=False):
    # 根据需求选择单模型或集成模型，未启用集成时避免额外加载
    has_multiple_targets = len(core_args.model_name) > 1

    if load_ensemble and has_multiple_targets:  # 多模型 -> 集成
        return WhisperModelEnsemble(core_args.model_name, device=device, task=core_args.task, language=core_args.language)

    if 'canary' in core_args.model_name[0]:  # Canary 模型
        return CanaryModel(device=device, task=core_args.task, language=core_args.language, pnc=True)

    # Whisper 单模型（或仅取列表首个）
    return WhisperModel(core_args.model_name[0], device=device, task=core_args.task, language=core_args.language)
