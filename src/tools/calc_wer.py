import sys
import os
import editdistance
import json

from whisper.normalizers import EnglishTextNormalizer


# 读取假设与参考文件路径（假设文件为 JSON）
fout = sys.argv[1]   # 预测 JSON 路径
fref = sys.argv[2]   # 参考 trn 文本路径

hyp_file = sys.argv[1] + '_hyp'  # sclite 期望的预测文件路径
ref_file = sys.argv[1] + '_ref'  # sclite 期望的参考文件路径

hyp_writer = open(hyp_file, 'w') # 输出文件句柄
ref_writer = open(ref_file, 'w')


std = EnglishTextNormalizer()    # whisper 的英文标准化器

ref_data = []                    # 暂存参考 (uid, sent)

for line in open(fref):
    uid, sent = line.strip().split(None, 1)
    sent = std(sent)
    ref_data.append([uid, sent])
    if not sent:
        sent = ' '
    ref_writer.write(f"{sent}({uid})\n")  # 写入 sclite trn 参考格式


with open(sys.argv[1]) as fin:
    hyp_data = json.load(fin)
    assert len(hyp_data) == len(ref_data)
    for ii, sent in enumerate(hyp_data):
        uid = ref_data[ii][0]
        sent = std(sent)
        if not sent:
            sent = ' '
        hyp_writer.write(f"{sent}({uid})\n")  # 写入 sclite trn 预测格式
