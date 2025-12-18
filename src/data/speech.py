LIBRISPEECH_DIR = '/root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech'      # LibriSpeech 根目录
TEDLIUM_DIR = '/root/autodl-tmp/prepend_acoustic_attack/data/tedlium/tedlium/test/' # TedLium3 测试集根目录
MGB_DIR = '/root/autodl-tmp/prepend_acoustic_attack/data/mvse/MGB-3/mgb3/test/'     # MGB-3 测试集根目录
ARTIE_DIR = '/root/autodl-tmp/prepend_acoustic_attack/data/artie-bias-corpus/data/' # ARTIE 偏见数据根目录

def _librispeech(sub_dir):
    '''
        for clean audio, set `sub_dir' to dev_clean/test_clean as dev/test sets
        for noisy audio, set `sub_dir' to dev_other/test_other as dev/test sets
    '''
    # 原来是: return _process(..., ['rm2114', 'vr313'])
    # 改成:
    return _process(f'{LIBRISPEECH_DIR}/{sub_dir}/audio_ref_pair_list')
def _tedlium():
    '''
        Returns the test split for TedLium3 dataset
    '''
    return _process(f'{TEDLIUM_DIR}/audio_ref_pair_list')  # 只返回测试集列表

def _mgb():
    '''
        Returns the test split for MGB-3 dataset
    '''
    return _process(f'{MGB_DIR}/audio_ref_pair_list', ['mq227', 'vr313'])  # 进行用户路径替换

def _artie():
    '''
        Returns the test split for ARTIE BIAS dataset
    '''
    return _process(f'{ARTIE_DIR}/audio_ref_pair')  # 直接解析 ARTIE 列表


def _process(fname, replace_user=None):
    audio_transcript_pair_list = []                   # 存放解析后的样本字典
    # === 🔴 请插入下面这 3 行调试代码 ===
    # import os
    # print(f"\n[DEBUG] Python试图打开的路径: {fname}")
    # print(f"[DEBUG] 路径的真实身份 (repr): {repr(fname)}")  # 这行能显示出隐藏的 \n 或空格
    # print(f"[DEBUG] Python能看到它吗?: {os.path.exists(fname)}\n")
    # === 🟢 插入结束 ===
    with open(fname, 'r') as fin:                     # 逐行读取 audio-ref 列表
        for line in fin:
            _, audio, ref = line.split(None, 2)       # 每行格式：<id> <audio_path> <transcript>
            ref = ref.rstrip('\n')                    # 去掉末尾换行

            if replace_user is not None:
                # 根据 replace_user 替换路径中的用户名片段
                audio = audio.replace(replace_user[0], replace_user[1])

            sample = {
                    'audio': audio,                   # 音频文件路径
                    'ref': ref                        # 参考转写
                }
            audio_transcript_pair_list.append(sample) # 追加到列表
    return audio_transcript_pair_list                 # 返回样本集合
