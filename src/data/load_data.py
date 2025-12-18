from .speech import _librispeech, _tedlium, _mgb, _artie  # 语音数据集本地列表解析
from .fleurs import _fleurs                               # HuggingFace FLEURS 数据集处理
from src.tools.tools import get_default_device             # 获取设备用于 FLEURS 预测


def load_data(core_args):
    '''
        Return data as train_data, test_data
        Each data is a list (over data samples), where each sample is a dictionary
            sample = {
                        'audio':    <path to utterance audio file>,
                        'ref':      <Reference transcription>,
                    }
    '''
    # 支持单/多数据集输入，按名称派发到具体加载函数
    def load_single_dataset(data_name):
        if data_name == 'fleurs':  # FLEURS 需要设备以便可选生成预测参考
            device = get_default_device(core_args.gpu_id)
            return _fleurs(lang=core_args.language, use_pred_for_ref=core_args.use_pred_for_ref, model_name=core_args.model_name[0], device=device)
        elif data_name == 'tedlium':   # TedLium 仅测试集
            return None, _tedlium()
        elif data_name == 'mgb':       # MGB-3 仅测试集
            return None, _mgb()
        elif data_name == 'artie':     # ARTIE 偏见测试集
            return None, _artie()
        elif data_name == 'librispeech':
                # 改成 clean
            return _librispeech('dev-clean'), _librispeech('test-clean')
        else:
            raise ValueError(f"Unknown dataset name: {data_name}")  # 未知名称直接报错

    if isinstance(core_args.data_name, list) and len(core_args.data_name) > 1:
        train_data_combined, test_data_combined = [], []   # 聚合多数据集样本
        for data_name in core_args.data_name:
            train_data, test_data = load_single_dataset(data_name)  # 逐个加载
            if train_data is not None:
                train_data_combined.extend(train_data)      # 追加训练样本
            if test_data is not None:
                test_data_combined.extend(test_data)        # 追加测试样本
        return train_data_combined, test_data_combined      # 返回合并后的集合
    else:
        # data_name 为单个字符串或长度为 1 的列表时直接加载
        return load_single_dataset(core_args.data_name[0] if isinstance(core_args.data_name, list) else core_args.data_name)



# def load_data(core_args):
#     '''
#         Return data as train_data, test_data
#         Each data is a list (over data samples), where each sample is a dictionary
#             sample = {
#                         'audio':    <path to utterance audio file>,
#                         'ref':      <Reference transcription>,
#                     }
#     '''
#     if core_args.data_name == 'fleurs':
#         device = get_default_device(core_args.gpu_id)
#         return _fleurs(lang=core_args.language, use_pred_for_ref=core_args.use_pred_for_ref, model_name=core_args.model_name[0], device=device)

#     if core_args.data_name == 'tedlium':
#         return None, _tedlium()

#     if core_args.data_name == 'mgb':
#         return None, _mgb()

#     if core_args.data_name == 'artie':
#         return None, _artie()

#     if core_args.data_name == 'librispeech':
#         return _librispeech('dev_other'), _librispeech('test_other')

    


