from .whisper import WhisperModel, WhisperModelEnsemble  # Whisper 封装
from .canary import CanaryModel                          # Canary 封装

def load_model(core_args, device=None):
    # 根据模型名选择 Whisper 单模型/集成或 Canary，并保持任务与语言配置一致
    if len(core_args.model_name) > 1:  # 多模型 -> 集成
        return WhisperModelEnsemble(core_args.model_name, device=device, task=core_args.task, language=core_args.language)
    else:
        if 'canary' in core_args.model_name[0]:  # Canary 模型
            return CanaryModel(device=device, task=core_args.task, language=core_args.language, pnc=True)
        else:                                    # Whisper 单模型
            return WhisperModel(core_args.model_name[0], device=device, task=core_args.task, language=core_args.language)
