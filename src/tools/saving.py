import os

def base_path_creator(core_args, create=True):
    # 按数据集/模型/任务/语言/随机种子逐层创建实验目录
    path = '.'
    path = next_dir(path, 'experiments', create=create)                  # 根目录 /experiments
    path = next_dir(path, '_'.join(core_args.data_name), create=create)  # 数据集名
    path = next_dir(path, '_'.join(core_args.model_name), create=create) # 模型名
    path = next_dir(path, core_args.task, create=create)                 # 任务
    path = next_dir(path, core_args.language, create=create)             # 语言
    if core_args.seed != 1:
        path = next_dir(path, f'seed{core_args.seed}', create=create)    # 非默认种子单独目录
    return path

def create_attack_base_path(attack_args, path='.', mode='train', create=True):
    # 根据 train/eval 模式构建攻击实验目录
    base_dir = 'attack_train' if mode == 'train' else 'attack_eval'
    path = next_dir(path, base_dir, create=create)

    # Common directory structure for both train and eval
    path = next_dir(path, attack_args.attack_method, create=create)
    if attack_args.attack_command != 'mute':
        path = next_dir(path, f'command_{attack_args.attack_command}', create=create)  # 指定攻击命令
    if attack_args.attack_token != 'eot':
        path = next_dir(path, f'attack_token{attack_args.attack_token}', create=create) # 目标 token
    if attack_args.attack_init != 'random':
        attack_init_path_str = attack_args.attack_init
        attack_init_path_str = '-'.join(attack_init_path_str.split('/'))
        path = next_dir(path, f'attack_init_{attack_init_path_str}', create=create)     # 初始化来源
    path = next_dir(path, f'attack_size{attack_args.attack_size}', create=create)       # 对抗段长度
    path = next_dir(path, f'clip_val{attack_args.clip_val}', create=create)             # 裁剪阈值

    # Additional directory for eval mode
    if mode == 'eval':
        path = next_dir(path, f'attack-epoch{attack_args.attack_epoch}', create=create) # 评估用的 epoch
    
    return path

def attack_base_path_creator_train(attack_args, path='.', create=True):
    return create_attack_base_path(attack_args, path, 'train', create)

def attack_base_path_creator_eval(attack_args, path='.', create=True):
    return create_attack_base_path(attack_args, path, 'eval', create)



def next_dir(path, dir_name, create=True):
    if not os.path.isdir(f'{path}/{dir_name}'):
        try:
            if create:
                os.mkdir(f'{path}/{dir_name}')
            else:
                raise ValueError ("provided args do not give a valid model path")
        except:
            # path has already been created in parallel
            pass
    path += f'/{dir_name}'
    return path
