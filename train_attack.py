'''
    Train adversarial segment
'''

import random
import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json

from src.tools.args import core_args, attack_args
from src.data.load_data import load_data
from src.models.load_model import load_model
from src.tools.tools import get_default_device, set_seeds
from src.attacker.selector import select_train_attacker
from src.tools.saving import base_path_creator, attack_base_path_creator_train

if __name__ == "__main__":

    # 获取命令行参数（核心参数 + 攻击参数）
    core_args, c = core_args()
    attack_args, a = attack_args()

    # 设定随机种子，保证实验可复现
    set_seeds(core_args.seed)
    base_path = base_path_creator(core_args)                     # 创建实验主目录
    attack_base_path = attack_base_path_creator_train(attack_args, base_path)  # 创建攻击子目录

    # 记录命令行调用，便于复现
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # 选择计算设备（优先 GPU）
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # 加载训练数据
    data, _ = load_data(core_args)

    # 加载 ASR 模型
    model = load_model(core_args, device=device)

    # 构造攻击器并开始训练对抗前置音频段
    attacker = select_train_attacker(attack_args, core_args, model, device=device)
    attacker.train_process(data, attack_base_path)
    
