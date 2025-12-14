'''
 General small processing activities - extract and save audio segment
'''
import argparse
import torch
import os
import sys
import numpy as np

from src.attacker.audio_raw.audio_attack_model_wrapper import AudioAttackModelWrapper


def get_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--attack_model_path', type=str, default='', help='Full path to attack model')  # 攻击模型权重路径
    commandLineParser.add_argument('--save_path', type=str, default='', help='Full path for where to save numpy array')  # 导出的 numpy 保存路径

    return commandLineParser.parse_known_args()

if __name__ == "__main__":

    args, _ = get_args()

    # 记录命令行调用，方便复现
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/process.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # 从攻击模型中提取可学习的音频对抗段并保存为 numpy 数组，便于复用或可视化

    attack_model = AudioAttackModelWrapper(None, attack_size=10240)                 # 构建包装器占位（不需要 tokenizer）
    attack_model.load_state_dict(torch.load(f'{args.attack_model_path}'))            # 加载权重
    audio = attack_model.audio_attack_segment.cpu().detach().numpy()                # 取出可学习音频段
    np.save(args.save_path, audio)                                                  # 保存为 numpy
