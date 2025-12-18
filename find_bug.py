import os

# 这是报错里那个死活找不到的路径
# 我直接从你的日志里复制过来的，原封不动
target_path = '/root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech/dev-clean/audio_ref_pair_list'

print(f"正在诊断路径: {target_path}\n")

current = '/'
parts = target_path.strip('/').split('/')

for i, part in enumerate(parts):
    # 拼接下一层
    next_path = os.path.join(current, part)
    
    # 检查是否存在
    if os.path.exists(next_path):
        print(f"✅ 第 {i+1} 层通过: {next_path}")
        current = next_path
    else:
        print(f"❌ 【断在这里了！】: {next_path}")
        print(f"   Python 认为 '{os.path.basename(current)}' 目录下没有叫 '{part}' 的东西。")
        
        # 列出这一层到底有啥
        try:
            siblings = os.listdir(current)
            print(f"   👀 实际上 '{os.path.basename(current)}' 里面只有这些: {siblings}")
            
            # 帮你找找是不是大小写或空格问题
            for s in siblings:
                if s.strip() == part.strip():
                     print(f"   💡 破案了！你写的是 '{part}' (长度{len(part)})，但实际是 '{s}' (长度{len(s)})。")
                     if len(part) != len(s):
                         print("      (⚠️ 注意：文件名末尾可能有空格！)")
        except Exception as e:
            print(f"   (无法读取目录: {e})")
        break