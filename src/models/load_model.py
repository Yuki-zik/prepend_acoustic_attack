from .whisper import WhisperModel, WhisperModelEnsemble  # Whisper 封装
from .canary import CanaryModel                          # Canary 封装
import torch


def _to_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_model(core_args, device=None, load_ensemble=False):
    # 根据需求选择单模型或集成模型，未启用集成时避免额外加载
    has_multiple_targets = len(core_args.model_name) > 1
    dtype = _to_torch_dtype(getattr(core_args, "dtype", "float32"))

    if load_ensemble and has_multiple_targets:  # 多模型 -> 集成
        return WhisperModelEnsemble(core_args.model_name, device=device, task=core_args.task, language=core_args.language, dtype=dtype)

    if 'canary' in core_args.model_name[0]:  # Canary 模型
        return CanaryModel(device=device, task=core_args.task, language=core_args.language, pnc=True)

    # Whisper 单模型（或仅取列表首个）
    return WhisperModel(core_args.model_name[0], device=device, task=core_args.task, language=core_args.language, dtype=dtype)
