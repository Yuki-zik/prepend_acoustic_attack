from whisper.normalizers import EnglishTextNormalizer  # Whisper 文本标准化
# 通用工具：评估指标、随机种子、设备选择等
import jiwer                                           # WER 计算库
import torch                                           # 张量/设备
import random                                          # Python 随机
from tqdm import tqdm                                  # 进度条
import seaborn as sns                                  # 绘图

import nltk                                            # BLEU 依赖
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # BLEU 计算
import matplotlib.pyplot as plt                        # 绘图
from comet import download_model, load_from_checkpoint # COMET 评估


from langdetect import detect, DetectorFactory, detect_langs             # 语言检测
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 0                                                 # 固定 langdetect 随机种子

def set_seeds(seed):
    # 统一 PyTorch 与 Python 随机种子
    torch.manual_seed(seed)                      # 设定 torch 随机种子
    random.seed(seed)                            # 设定 Python 随机种子

def get_default_device(gpu_id=0):
    # 首选指定 GPU，否则回退到 CPU
    if torch.cuda.is_available():                # 检查 CUDA
        print("Got CUDA!")
        return torch.device(f'cuda:{gpu_id}')    # 返回指定 GPU
    else:
        print("No CUDA found")
        return torch.device('cpu')               # 默认 CPU

def eval_wer(hyps, refs, get_details=False):
    # assuming the texts are already aligned and there is no ID in the texts
    # WER
    # 计算词错误率，必要时返回插入/删除/替换细分
    std = EnglishTextNormalizer()                          # 文本规范化器
    hyps = [std(hyp) for hyp in hyps]                      # 规范化预测
    refs = [std(ref) for ref in refs]                      # 规范化参考
    out = jiwer.process_words(refs, hyps)                  # 计算 WER 等统计
    
    total_ref = sum(len(ref.split()) for ref in refs)      # total number of words in the reference
    
    if not get_details:
        return out.wer                                     # 只返回 WER
    else:
        # return ins, del and sub rates
        ins_rate = out.insertions / total_ref              # 插入率
        del_rate = out.deletions / total_ref               # 删除率
        sub_rate = out.substitutions / total_ref           # 替换率
        return {'WER': out.wer, 'INS': ins_rate, 'DEL': del_rate, 'SUB': sub_rate, 'HIT': out.hits/total_ref}  # 详细指标字典

def get_english_probability(text):
    """
    Returns the probability of the given text being in English.
    
    :param text: The text to analyze
    :return: Probability of the text being in English
    """
    try:
        # Detect the languages and their probabilities
        languages = detect_langs(text)                     # langdetect 返回概率分布
        
        # Find the English probability
        for lang in languages:
            if lang.lang == 'en':                          # 找到英语概率
                return lang.prob
        
        # If English is not found, return 0
        return 0.0
    except Exception as e:
        # Handle cases where language detection fails
        print(f"Error detecting language: {e}")            # 打印错误并返回 0
        return 0.0

def eval_english_probability(hyps):
    """
    Returns the average probability of English for a list of sentences.
    
    :param hyps: List of sentences
    :return: Average probability of the sentences being in English
    """
    if not hyps:
        return 0.0                                         # 空列表返回 0
    
    total_prob = 0.0                                       # 累加概率
    for sentence in hyps:                                  # 遍历每个句子
        total_prob += get_english_probability(sentence)    # 求英语概率
    
    return total_prob / len(hyps)                          # 返回均值

def eval_english_probability_dist(hyps, attack=False):
    """
    Computes the probability of each sentence being in English for a list of sentences,
    plots the distribution with KDE, and saves it to a file called 'experiments/plots/english_prob_dist.png'.

    Args:
        hyps (list of str): List of hypothesis strings.

    Returns:
        float: Overall average probability of the sentences being in English.
    """
    if not hyps:
        return 0.0

    # 逐句估计是英语的概率，再统计分布
    probabilities = [get_english_probability(sentence) for sentence in hyps]  # 计算每句的英语概率

    # Plot the distribution of probabilities with KDE
    plt.figure(figsize=(10, 6))                                                # 初始化图像
    sns.histplot(probabilities, bins=20, color='blue', edgecolor='black', kde=True, stat='density', alpha=0.3)  # 直方图+核密度
    sns.kdeplot(probabilities, color='blue', linewidth=2)                      # 叠加 KDE
    plt.xlabel('Probability of being in English')                              # X 轴标签
    plt.ylabel('Density')                                                      # Y 轴标签
    
    # Save the plot to a file
    fpath = 'experiments/plots/english_prob_dist_no_attack.png'                # 默认无攻击输出路径
    if attack:
        fpath = 'experiments/plots/english_prob_dist_attack.png'               # 攻击下的输出路径
    plt.savefig(fpath, bbox_inches='tight')                                    # 保存图像
    plt.close()                                                                # 关闭图

    # Return the average probability
    return sum(probabilities) / len(probabilities)                             # 返回均值

def eval_bleu(hyps, refs):
    """
    Computes the BLEU score for a list of hypotheses and references.

    Args:
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.

    Returns:
        float: Overall BLEU score.
    """
    # 对句子分词后按句平均 BLEU
    hyps = [hyp.split() for hyp in hyps]                  # 预测分词
    refs = [[ref.split()] for ref in refs]  # Note: refs need to be a list of lists of lists for sentence_bleu  # 参考分词

    # Compute the overall BLEU score
    total_bleu = 0                                       # 累加 BLEU
    for hyp, ref in zip(hyps, refs):
        total_bleu += sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)  # 单句 BLEU
    return total_bleu / len(hyps)                        # 均值

def eval_bleu_dist(hyps, refs, attack=False):
    """
    Computes the sentence-level BLEU scores for a list of hypotheses and references,
    plots the distribution, and saves it to a file called 'experiments/plots/bleu_dist.png'.

    Args:
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.

    Returns:
        list of float: List of sentence-level BLEU scores.
    """
    # Tokenize the sentences
    hyps = [hyp.split() for hyp in hyps]                # 预测分词
    refs = [[ref.split()] for ref in refs]  # Note: refs need to be a list of lists of lists for sentence_bleu  # 参考分词

    # Compute sentence-level BLEU scores
    bleu_scores = []                                    # 保存每句 BLEU
    for hyp, ref in zip(hyps, refs):
        score = sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)  # 单句 BLEU
        bleu_scores.append(score)                       # 追加

    # Plot the distribution of BLEU scores
    plt.figure(figsize=(10, 6))                         # 画布
    sns.histplot(bleu_scores, bins=20, color='blue', edgecolor='black', kde=True, stat='density', alpha=0.3)  # 直方图+KDE
    sns.kdeplot(bleu_scores, color='blue', linewidth=2) # KDE
    plt.xlabel('COMET score')                           # 这里标签应为 BLEU（原代码沿用 COMET 字样）
    plt.ylabel('Density')
    
    # Save the plot to a file
    fpath = 'experiments/plots/bleu_dist_no_attack.png' # 默认无攻击
    if attack:
        fpath = 'experiments/plots/bleu_dist_attack.png' # 攻击结果
    plt.savefig(fpath, bbox_inches='tight')             # 保存
    plt.close()                                         # 关闭

    return sum(bleu_scores) / len(bleu_scores)


def eval_comet(srcs, hyps, refs):
    """
    Computes the overall COMET score for a list of sources, hypotheses, and references.

    Args:
        srcs (list of str): List of source strings.
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.

    Returns:
        float: Overall COMET score.
    """
    # Load the pre-trained COMET model for evaluation
    model_path = download_model("wmt20-comet-da")           # 下载/缓存 COMET 模型
    model = load_from_checkpoint(model_path)                # 加载模型

    # Prepare input data
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(srcs, hyps, refs)]  # 按样本打包字典列表

    # Compute COMET

    scores = model.predict(data, batch_size=8)['scores']    # 批量推理得到分数
    scores = [s if s>0 else 0 for s in scores]              # 负数截断为 0
    return sum(scores)/len(scores) # average COMET score    # 返回平均分


def eval_comet_dist(srcs, hyps, refs, attack=False):
    """
    Computes the sentence-level COMET scores for a list of sources, hypotheses, and references,
    plots the distribution, and saves it to a file called 'experiments/plots/comet_dist.png'.

    Args:
        srcs (list of str): List of source strings.
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.

    Returns:
        float: Overall average COMET score.
    """
    # Load the pre-trained COMET model for evaluation
    model_path = download_model("wmt20-comet-da")           # 下载/缓存模型
    model = load_from_checkpoint(model_path)                # 载入

    # Prepare input data
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(srcs, hyps, refs)]  # 打包输入

    # Compute COMET scores
    comet_scores = model.predict(data, batch_size=8)['scores']  # 获取分数
    comet_scores = [s if s>0 else 0 for s in comet_scores]      # 负数截断

    # Plot the distribution of COMET scores
    plt.figure(figsize=(10, 6))                                 # 图像
    sns.histplot(comet_scores, bins=20, color='blue', edgecolor='black', kde=True, stat='density', alpha=0.3)  # 直方图
    sns.kdeplot(comet_scores, color='blue', linewidth=2)         # KDE
    plt.xlabel('COMET score')                                    # X 轴
    plt.ylabel('Density')                                        # Y 轴
    
    # Save the plot to a file
    fpath = 'experiments/plots/comet_dist_no_attack.png'        # 无攻击路径
    if attack:
        fpath = 'experiments/plots/comet_dist_attack.png'       # 攻击路径
    plt.savefig(fpath, bbox_inches='tight')                     # 保存
    plt.close()                                                 # 关闭

    # Return the average COMET score
    return sum(comet_scores) / len(comet_scores)



def eval_neg_seq_len(hyps):
    '''
        Average sequence length (negative)
    '''
    nlens = 0                                        # 累加 token 数
    for hyp in hyps:
        nlens += (len(hyp.split()))                 # 计算每句 token 数
    return (-1)*nlens/len(hyps)                     # 取负的平均长度

def eval_frac_0_samples(hyps):
    '''
        Fraction of samples of 0 tokens
    '''
    no_len_count = 0                                # 统计空输出数量
    for hyp in hyps:
        if len(hyp) == 0:                           # 空字符串则计数
            no_len_count +=1
    return no_len_count/len(hyps)                   # 返回比例

def eval_average_fraction_of_languages(sentences, languages):
    """
    Calculates the average fraction of specified languages across a list of sentences.
    
    Args:
        sentences (list of str): A list of sentences to analyze.
        languages (list of str): A list of language codes to detect (e.g., 'en' for English, 'fr' for French).
    
    Returns:
        dict: A dictionary where keys are language codes and values are the average fractions of each language.
    """
    total_language_counts = {lang: 0 for lang in languages}
    total_words = 0                                 # 总词数
    
    for sentence in sentences:                      # 遍历句子
        words = sentence.split()                    # 简单分词
        total_words += len(words)                   # 累计词数
        language_counts = {lang: 0 for lang in languages}  # 每句语言计数
        
        for word in words:                          # 遍历词汇
            try:
                detected_language = detect(word)    # 检测词语言
                if detected_language in language_counts:
                    language_counts[detected_language] += 1  # 属于目标语言则计数
            except LangDetectException:
                # Skip words that cannot be detected
                continue                            # 检测失败跳过
        
        for lang in languages:                      # 将本句计数累加到全局
            total_language_counts[lang] += language_counts[lang]
    
    if total_words == 0:
        return {lang: 0 for lang in languages}      # 空文本返回 0
    
    average_language_fractions = {lang: total_language_counts[lang] / total_words for lang in languages}  # 计算占比
    return average_language_fractions


def eval_bleu_english_prob_recall(hyps, refs, attack=False, rev_attack=False):
    """
    Computes the BLEU scores and English probabilities for given samples, ranks the samples 
    by their probability of being English, and computes the average BLEU score up to each rank.
    Plots the 'BLEU score of samples classified as English (successfully attacked)' against 
    the '% of samples classified as English (successfully attacked)'.

    If rev_attack = True, calculates BLEU score for samples unsuccessfully attacked
    Args:
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.
        attack (bool): Flag indicating whether the attack mode is enabled. If False, the function 
                       will not run and return None.

    Returns:
        None
    """
    if not attack:
        return None

    # Compute BLEU scores
    hyps_tok = [hyp.split() for hyp in hyps]                           # 预测分词
    refs_tok = [[ref.split()] for ref in refs]  # refs need to be a list of lists of lists for sentence_bleu  # 参考分词
    bleu_scores = [
        sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)  # 单句 BLEU
        for hyp, ref in zip(hyps_tok, refs_tok)
    ]

    # Compute English probabilities
    english_probs = [get_english_probability(hyp) for hyp in hyps]     # 计算英语概率

    # Combine BLEU scores and English probabilities
    combined = list(zip(bleu_scores, english_probs))                   # 组成 (BLEU, prob) 对
    
    # Rank samples by probability of English (highest to lowest) -- if rev_attack then in reverse
    combined.sort(key=lambda x: x[1], reverse=not rev_attack)          # 按概率排序，可反向

    # Compute average BLEU score up to each rank
    avg_bleu_scores = []                                               # 前 k 平均 BLEU
    percent_en = []                                                    # 前 k 百分比
    cumulative_sum = 0.0                                               # 累积和
    for i, (bleu_score, en_prob) in enumerate(combined, start=1):
        cumulative_sum += bleu_score                                   # 累积
        avg_bleu_scores.append(cumulative_sum / i)                     # 计算均值
        percent_en.append((i / len(combined)) * 100)                   # 已覆盖百分比

    # Plotting the results
    plt.figure(figsize=(10, 6))                                        # 画布
    plt.plot(percent_en, avg_bleu_scores, marker='o', linestyle='-', color='b')  # 绘制曲线
    if rev_attack:
        plt.xlabel('Samples Classed as NOT En (%)')                    # 横轴：非英语比例
        plt.ylabel('Average BLEU Score of Samples Classed as NOT En')  # 纵轴：均值 BLEU
        save_path = 'experiments/plots/bleu_vs_not_en_probability.png' # 输出路径
    else:
        plt.xlabel('Samples Classed as En (%)')                        # 横轴：英语比例
        plt.ylabel('Average BLEU Score of Samples Classed as En')      # 纵轴：均值 BLEU
        save_path = 'experiments/plots/bleu_vs_en_probability.png'     # 输出路径
    plt.grid(True)                                                     # 网格

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')                        # 保存
    plt.close()                                                        # 关闭

    return None


def eval_comet_english_prob_recall(srcs, hyps, refs, attack=False, rev_attack=False):
    """
    Computes the COMET scores and English probabilities for given samples, ranks the samples 
    by their probability of being English, and computes the average COMET score up to each rank.

    Args:
        srcs (list of str): List of source strings.
        hyps (list of str): List of hypothesis strings.
        refs (list of str): List of reference strings.
        attack (bool): Flag indicating whether the attack mode is enabled. If False, the function 
                       will not run and return None.
        if rev_attack: COMET score of unsuccessfully attacked samples

    Plots:
        dict: A dictionary with two keys:
              - 'comet_scores': List of average COMET scores for the top-k samples by English probability.
              - 'percent_en': List of percentages of samples classified as English up to each k.
    """
    if not attack:
        return None

    # Load the pre-trained COMET model for evaluation
    model_path = download_model("wmt20-comet-da")                       # 下载/缓存 COMET 模型
    model = load_from_checkpoint(model_path)                            # 加载模型

    # Prepare input data
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(srcs, hyps, refs)]  # 打包输入

    # Compute COMET scores
    comet_scores = model.predict(data, batch_size=8)['scores']          # 批量预测
    comet_scores = [s if s > 0 else 0 for s in comet_scores]            # 负数截断

    # Compute English probabilities
    english_probs = [get_english_probability(hyp) for hyp in hyps]      # 计算英语概率

    # Combine COMET scores and English probabilities
    combined = list(zip(comet_scores, english_probs))                   # 组合列表
    
    # Rank samples by probability of English (highest to lowest)
    combined.sort(key=lambda x: x[1], reverse=not rev_attack)           # 按概率排序

    # Compute average COMET score up to each rank
    avg_comet_scores = []                                               # 前 k 平均 COMET
    percent_en = []                                                     # 对应百分比
    cumulative_sum = 0.0                                                # 累积和
    for i, (comet_score, en_prob) in enumerate(combined, start=1):
        cumulative_sum += comet_score                                   # 累加
        avg_comet_scores.append(cumulative_sum / i)                     # 均值
        percent_en.append((i / len(combined)) * 100)                    # 进度百分比

    # Plotting the results
    plt.figure(figsize=(10, 6))                                         # 画布
    plt.plot(percent_en, avg_comet_scores, marker='o', linestyle='-', color='b')  # 曲线
    if rev_attack:
        plt.xlabel('Samples Classed as NOT En (%)')                     # 非英语样本比例
        plt.ylabel('Average COMET Score of Samples Classed as NOT En')  # 均值 COMET
        save_path = 'experiments/plots/comet_vs_not_en_probability.png' # 输出路径
    else:
        plt.xlabel('Samples Classed as En (%)')                         # 英语样本比例
        plt.ylabel('Average COMET Score of Samples Classed as En')      # 均值 COMET
        save_path = 'experiments/plots/comet_vs_en_probability.png'     # 输出路径
    plt.grid(True)                                                      # 网格

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')                         # 保存
    plt.close()                                                         # 关闭

    return None


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()                           # 初始化统计量

    def reset(self):
        self.val = 0                           # 当前值
        self.avg = 0                           # 平均值
        self.sum = 0                           # 累积和
        self.count = 0                         # 样本数

    def update(self, val, n=1):
        self.val = val                         # 更新当前值
        self.sum += val * n                    # 增加总和
        self.count += n                        # 增加计数
        self.avg = self.sum / self.count       # 重新计算平均值
