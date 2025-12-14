"""
    Evaluate attack
"""

import sys
import os
import torch
import numpy as np

from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, attack_args
from src.tools.saving import (
    base_path_creator,
    attack_base_path_creator_eval,
    attack_base_path_creator_train,
)
from src.data.load_data import load_data
from src.models.load_model import load_model
from src.attacker.selector import select_eval_attacker

if __name__ == "__main__":

    # 获取命令行参数
    core_args, c = core_args()
    attack_args, a = attack_args()

    print(core_args)   # 打印基础配置
    print(attack_args) # 打印攻击配置

    set_seeds(core_args.seed)  # 固定随机种子
    if not attack_args.transfer:
        base_path = base_path_creator(core_args)                         # 生成实验主目录
        attack_base_path = attack_base_path_creator_eval(attack_args, base_path)  # 生成评估输出目录
    else:
        base_path = None
        attack_base_path = None

    # 记录命令行调用
    if not os.path.isdir("CMDs"):
        os.mkdir("CMDs")
    with open("CMDs/eval_attack.cmd", "a") as f:
        f.write(" ".join(sys.argv) + "\n")

    # 选择计算设备
    if core_args.force_cpu:
        device = torch.device("cpu")
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # 加载数据集（可选择评估训练集或测试集）
    train_data, test_data = load_data(core_args)
    if attack_args.eval_train:
        test_data = train_data

    # 加载待评估的 ASR 模型
    model = load_model(core_args, device=device)

    # 根据配置构造评估阶段的攻击器
    attacker = select_eval_attacker(attack_args, core_args, model, device=device)

    # 确定攻击模型目录（可评估迁移攻击）
    if not attack_args.transfer:
        attack_model_dir = f"{attack_base_path_creator_train(attack_args, base_path)}/prepend_attack_models"
    else:
        attack_model_dir = attack_args.attack_model_dir

    # 1) 不添加攻击段的基线测试
    if not attack_args.not_none:
        print("No attack")
        out = attacker.eval_uni_attack(
            test_data,
            attack_model_dir=attack_model_dir,
            attack_epoch=-1,                    # -1 表示不加载攻击权重
            cache_dir=attack_base_path,
            force_run=attack_args.force_run,
            metrics=attack_args.eval_metrics,
            frac_lang_languages=attack_args.frac_lang_langs,
        )
        print(out)
        print()

    # 2) 评估指定 epoch 的攻击效果
    print("Attack")
    out = attacker.eval_uni_attack(
        test_data,
        attack_model_dir=attack_model_dir,
        attack_epoch=attack_args.attack_epoch, # 指定使用的攻击 checkpoint
        cache_dir=attack_base_path,
        force_run=attack_args.force_run,
        metrics=attack_args.eval_metrics,
        frac_lang_languages=attack_args.frac_lang_langs,
    )
    print(out)
    print()
