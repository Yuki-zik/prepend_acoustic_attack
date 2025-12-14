from .mel.soft_prompt_attack import SoftPromptAttack                 # mel 空间软提示攻击
from .mel.base import MelBaseAttacker                                # mel 空间基类
from .audio_raw.base import AudioBaseAttacker                        # 原始音频攻击基类
from .audio_raw.learn_attack import AudioAttack                      # 静音攻击
from .audio_raw.learn_attack_hallucinate import AudioAttackHallucinate # 幻觉攻击
from .audio_raw.learn_attack_translate import AudioAttackTranslate   # 翻译强制攻击

def select_eval_attacker(attack_args, core_args, model, device=None):
    # 评估阶段仅支持单模型，根据 attack_method 选择对应攻击器
    if len(core_args.model_name) > 1:
        raise ValueError("Code is designed to only evaluate a single model")  # 评估不支持集成

    if attack_args.attack_method == 'mel':
        return MelBaseAttacker(attack_args, model, device)            # mel 空间基线
    elif attack_args.attack_method == 'audio-raw':
        return AudioBaseAttacker(attack_args, model, device, attack_init=attack_args.attack_init)  # 原始音频基线


def select_train_attacker(attack_args, core_args, model, word_list=None, device=None):
    # 训练阶段根据命令/任务切换具体实现（音频空间或 mel 空间）
    if attack_args.attack_method == 'mel':
        return SoftPromptAttack(attack_args, model, device)           # mel 软提示训练器
    elif attack_args.attack_method == 'audio-raw':
        multiple_model_attack = False                                # 默认单模型攻击
        if len(core_args.model_name) > 1:
            multiple_model_attack = True                             # 集成攻击开启
        if attack_args.attack_command == 'mute':
            return AudioAttack(attack_args, model, device, lr=attack_args.lr, multiple_model_attack=multiple_model_attack, attack_init=attack_args.attack_init)  # 静音
        elif attack_args.attack_command == 'hallucinate':
            return AudioAttackHallucinate(attack_args, model, device, lr=attack_args.lr, multiple_model_attack=multiple_model_attack, attack_init=attack_args.attack_init)  # 幻觉
        elif attack_args.attack_command == 'translate':
            return AudioAttackTranslate(attack_args, model, device, lr=attack_args.lr, multiple_model_attack=multiple_model_attack, attack_init=attack_args.attack_init)    # 翻译强制

   
