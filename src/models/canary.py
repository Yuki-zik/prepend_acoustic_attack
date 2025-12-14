import torch                                                     # 设备与张量
from nemo.collections.asr.models import EncDecMultiTaskModel     # Nemo Canary 模型
import json                                                      # 写入清单文件
import os                                                        # 环境变量等

os.environ['NEMO_CACHE_DIR'] = '/home/vr313/rds/rds-altaslp-8YSp2LXTlkY/data/cache'  # 指定 Nemo 模型缓存

class CanaryModel:
    def __init__(self, device=torch.device('cpu'), task='transcribe', language='en', pnc=True):
        # 封装 Nemo Canary，多任务配置由 task 与 language 决定
        self.model_name = 'canary'
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b').to(device)  # 加载预训练模型到设备
        self.tokenizer = self.model.tokenizer                                            # 取 tokenizer
        self.taskname = 'asr' if task == 'transcribe' else 's2t_translation'             # Nemo 内部 task 名
        self.task = task                                                                 # 保存任务类型
        if task == 'transcribe':
            self.src_lang = language.split('_')[0]                                       # 转写时源=目标
            self.tgt_lang = self.src_lang
        else:
            self.src_lang = language.split('_')[0]                                       # 翻译时 src->tgt
            self.tgt_lang = language.split('_')[1]
        self.pnc = 'yes' if pnc else 'no'                                                # 是否保留标点
        
        # Update decode params
        decode_cfg = self.model.cfg.decoding                                             # 取解码配置
        decode_cfg.beam.beam_size = 1                                                    # 简化为贪心/beam=1
        self.model.change_decoding_strategy(decode_cfg)                                  # 应用解码策略

        self.prep_sot_ids()                                                              # 预备解码起始 token
    
    def prep_sot_ids(self):
        '''
            Special Tokens used by decoder
            <|startoftranscript|><|source_lang|><|taskname|><|target_lang|><|pnc|>
        '''
        # 预先缓存解码起始的特殊 token 序列，方便构造解码输入
        spl_tkns = self.tokenizer.special_tokens                                         # 取特殊 token 映射
        ids = []
        ids.append(spl_tkns['<|startoftranscript|>'])                                   # 起始 token
        ids.append(spl_tkns[f'<|{self.src_lang}|>'])                                    # 源语言 token
        ids.append(spl_tkns[f'<|{self.task}|>']) # transcribe or translate              # 任务 token
        ids.append(spl_tkns[f'<|{self.tgt_lang}|>'])                                    # 目标语言 token
        if self.pnc == 'yes':
            ids.append(spl_tkns['<|pnc|>'])                                             # 使用标点
        else:
            ids.append(spl_tkns['<|nopnc|>'])                                           # 不使用标点
        
        self.sot_ids = ids                                                              # 保存序列


    def create_manifest(self, audio_path):
        '''
            Create input manifest file for the Canary model
        '''
        manifest_entry = {
            "audio_filepath": audio_path,                                               # 待转写/翻译音频路径
            "duration": None,  # Duration can be set to None                            # 时长可留空
            "taskname": self.taskname,                                                  # Nemo 任务名
            "source_lang": self.src_lang,                                               # 源语言
            "target_lang": self.tgt_lang,                                               # 目标语言
            "pnc": self.pnc,                                                            # 标点开关
            "answer": "na"                                                              # 占位字段
        }
        
        manifest_path = 'experiments/input_manifest.json'                               # 清单保存位置
        with open(manifest_path, 'w') as f:
            json.dump(manifest_entry, f)                                                # 写入单行 JSON
            f.write('\n')                                                               # 加换行保持格式
        
        return manifest_path                                                            # 返回清单路径

    def predict(self, audio='', **kwargs):
        '''
            Run through Canary model
        '''
        # Create input manifest file
        manifest_path = self.create_manifest(audio)                                     # 生成清单
        
        # Pass through Canary model
        predicted_text = self.model.transcribe(                                         # 调用 Nemo 推理
            manifest_path,
            batch_size=1,
            verbose=False
        )
        return predicted_text[0] if predicted_text else None                            # 取第一条结果


