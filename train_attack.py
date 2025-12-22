'''
    Train adversarial segment
    脚本用途：训练对抗性音频片段（即那个能让 Whisper 变哑巴的噪音）
'''

import random
import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json

# 导入自定义模块
# src.tools.args: 处理命令行参数
from src.tools.args import core_args, attack_args
# src.data.load_data: 数据加载逻辑（你刚才修Bug的地方就在这里面）
from src.data.load_data import load_data
# src.models.load_model: 加载 Whisper 模型
from src.models.load_model import load_model
# src.tools.tools: 设备选择（CPU/GPU）和随机种子设置工具
from src.tools.tools import get_default_device, set_seeds
# src.attacker.selector: 根据参数选择具体的攻击器类（比如 AudioRawAttacker）
from src.attacker.selector import select_train_attacker
# src.tools.saving: 负责生成实验结果保存的路径结构
from src.tools.saving import base_path_creator, attack_base_path_creator_train

if __name__ == "__main__":

    # get command line arguments
    # 解析核心参数（如 dataset_name, model_name, seed 等）
    core_args, c = core_args()
    # 解析攻击参数（如 attack_method, epsilon, epoch 等）
    attack_args, a = attack_args()

    # set seeds
    # 设置随机种子，确保实验结果可复现
    set_seeds(core_args.seed)
    
    # 创建实验的基础保存路径（通常基于模型名和数据集名）
    base_path = base_path_creator(core_args)
    # 在基础路径下，创建本次特定攻击配置的保存路径（包含攻击方法、大小等信息）
    attack_base_path = attack_base_path_creator_train(attack_args, base_path)

    # Save the command run
    # 检查是否存在 'CMDs' 目录，没有则创建，用于记录运行过的命令
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    
    # 将本次运行的完整命令行指令追加写入到文件中，方便后续查看和复现
    with open('CMDs/train_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    # 如果参数指定强制使用 CPU，则设置设备为 cpu
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        # 否则自动获取可用的 GPU 设备
        device = get_default_device(core_args.gpu_id)
    print(device) # 打印当前使用的设备（如 cuda:0）

    # load training data
    # 加载训练数据。
    # load_data 返回两个值 (train_set, val_set)，这里用 _ 忽略了第二个返回值，因为训练脚本主要只需要训练集。
    # 注意：这正是会触发你之前 FileNotFoundError 的地方，因为它内部调用了 speech.py
    data, _ = load_data(core_args)

    # load model
    # 加载目标 Whisper 模型，并将其移动到指定的计算设备上
    multiple_model_attack = len(core_args.model_name) > 1
    model = load_model(core_args, device=device, load_ensemble=multiple_model_attack)

    # 初始化攻击器
    # 根据 attack_args 配置（如 'audio-raw'），选择对应的攻击类实例
    attacker = select_train_attacker(attack_args, core_args, model, device=device)
    
    # 开始训练流程
    # 传入数据和保存路径，开始迭代优化对抗样本
    attacker.train_process(data, attack_base_path)