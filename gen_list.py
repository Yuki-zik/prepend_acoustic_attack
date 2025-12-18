import os

# === 你的实际路径 ===
# 指向包含 dev-clean 的父目录
DATA_ROOT = '/root/autodl-tmp/prepend_acoustic_attack/data/librispeech/LibriSpeech/'

# !!! 关键修改：这里要用连字符 '-' (hyphen)，对应你硬盘上的真实文件夹名 !!!
SUBSETS = ['dev-clean', 'test-clean'] 

def generate_list(subset_name):
    # 拼凑路径: .../LibriSpeech/dev-clean
    subset_dir = os.path.join(DATA_ROOT, subset_name)
    # 生成文件: .../LibriSpeech/dev-clean/audio_ref_pair_list
    output_file = os.path.join(subset_dir, 'audio_ref_pair_list')
    
    if not os.path.exists(subset_dir):
        print(f"❌ 错误: 找不到目录 {subset_dir}")
        return

    print(f"正在扫描: {subset_dir} ...")
    
    lines = []
    for root, dirs, files in os.walk(subset_dir):
        for file in files:
            if file.endswith('.trans.txt'):
                trans_path = os.path.join(root, file)
                with open(trans_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) != 2: continue
                        
                        file_id = parts[0]
                        transcript = parts[1]
                        
                        audio_file = f"{file_id}.flac"
                        audio_path = os.path.join(root, audio_file)
                        
                        if os.path.exists(audio_path):
                            lines.append(f"{file_id} {audio_path} {transcript}")
    
    if len(lines) == 0:
        print(f"❌ 警告: {subset_name} 没扫到数据！")
    else:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write('\n'.join(lines))
        print(f"✅ 成功生成: {output_file} (共 {len(lines)} 条)")

if __name__ == '__main__':
    for subset in SUBSETS:
        generate_list(subset)