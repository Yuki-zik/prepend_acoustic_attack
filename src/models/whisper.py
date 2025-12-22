import torch                                          # 张量与设备控制
import whisper                                        # Whisper 官方库
import editdistance                                   # 备用编辑距离库（未使用）
from whisper.tokenizer import get_tokenizer           # 加载 Whisper tokenizer


CACHE_DIR = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/experiments/rm2114/.cache'  # Whisper 模型缓存目录

MODEL_NAME_MAPPER = {  # 自定义名称到官方 checkpoint 名的映射
    'whisper-tiny'  : 'tiny.en',
    'whisper-tiny-multi'  : 'tiny',
    'whisper-base'  : 'base.en',
    'whisper-base-multi'  : 'base',
    'whisper-small' : 'small.en',
    'whisper-small-multi' : 'small',
    'whisper-medium'  : 'medium.en',
    'whisper-medium-multi'  : 'medium',
    'whisper-large'  : 'large',
}

class WhisperModel:
    '''
        Wrapper for Whisper ASR Transcription
    '''
    def __init__(self, model_name='whisper-small', device=torch.device('cpu'), task='transcribe', language='en', dtype=None):
        # 将自定义名称映射到实际权重名称，并缓存到本地
        self.model_name = model_name
        load_kwargs = dict(device=device, download_root=CACHE_DIR)

        # 某些 whisper 版本不接受 dtype 参数；若传入则尝试兼容处理
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        try:
            self.model = whisper.load_model(MODEL_NAME_MAPPER[model_name], **load_kwargs)
        except TypeError:
            load_kwargs.pop("dtype", None)
            self.model = whisper.load_model(MODEL_NAME_MAPPER[model_name], **load_kwargs)

        self.task = task                                            # 任务类型：transcribe/translate
        self.language = language.split('_')[0]                      # 取源语言（translate 时为 src）
        self.tokenizer = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages, language=self.language, task=self.task)  # 构造 tokenizer

    
    def predict(self, audio='', initial_prompt=None, without_timestamps=False):
        '''
            Whisper decoder output here
        '''
        result = self.model.transcribe(audio, language=self.language, task=self.task, initial_prompt=initial_prompt, without_timestamps=without_timestamps)  # 调用官方转写
        segments = []                                                                                               # 存放分段文本
        for segment in result['segments']:                                                                          # 遍历每段
            segments.append(segment['text'].strip())                                                                # 去除首尾空白并收集
        return ' '.join(segments)                                                                                   # 拼接为一句话


class WhisperModelEnsemble:
    '''
        Wrapper for Whisper ASR
        Ensemble
        Ensure all models are either multi-lingual or English only
    '''
    def __init__(self, model_names=['whisper-small'], device=torch.device('cpu'), task='transcribe', language='en', dtype=None):
        # 加载多模型并共享 tokenizer，用于集成攻击或评估
        load_kwargs = dict(device=device, download_root=CACHE_DIR)
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        try:
            self.models = [whisper.load_model(MODEL_NAME_MAPPER[model_name], **load_kwargs) for model_name in model_names]  # 逐个载入模型
        except TypeError:
            load_kwargs.pop("dtype", None)
            self.models = [whisper.load_model(MODEL_NAME_MAPPER[model_name], **load_kwargs) for model_name in model_names]

        self.task = task                                      # 保存任务类型
        self.language = language                              # 保存语言
        self.tokenizer = get_tokenizer(self.models[0].is_multilingual, num_languages=self.models[0].num_languages, language=self.language, task=self.task)  # 使用首个模型的 tokenizer
