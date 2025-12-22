'''
 General small processing activities - extract and save audio segment
'''
import argparse
import torch
import os
import sys
import numpy as np

from src.attacker.audio_raw.audio_attack_model_wrapper import AudioAttackModelWrapper
###
# 使用脚本:
# python process.py \
#   --attack_model_path /root/autodl-tmp/prepend_acoustic_attack/experiments/librispeech/whisper-medium-multi/transcribe/en/attack_train/audio-raw/attack_size10240/clip_val0.02/prepend_attack_models/epoch30/model.th \
#   --save_path /root/autodl-tmp/prepend_acoustic_attack/audio_attack_segments/epoch30_extract.npy
###



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

    # 从攻击模型 state dict 中直接提取可学习的音频对抗段并保存为 numpy 数组

    state_dict = torch.load(f"{args.attack_model_path}", map_location="cpu")       # 读取 checkpoint
    if "audio_attack_segment" not in state_dict:                                   # 检查必需参数
        raise KeyError("audio_attack_segment not found in provided attack model")

    audio = state_dict["audio_attack_segment"].cpu().detach().numpy()              # 取出可学习音频段
    np.save(args.save_path, audio)                                                  # 保存为 numpy
