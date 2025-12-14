import torch                                # 张量计算
from whisper.audio import load_audio        # Whisper 音频加载
import string                               # 字符判定

def saliency(audio, audio_attack_model, whisper_model, device):
    '''
        Get saliency of audio and audio_attack_segment
    '''
    # 对对抗段与原始语音分别计算梯度范数，衡量两者对输出的敏感度
    adv_grad, audio_grad = _saliency_calculation(audio, audio_attack_model, whisper_model, device)

    adv_grad_norm = torch.linalg.vector_norm(adv_grad)        # 对抗段梯度范数
    audio_grad_norm = torch.linalg.vector_norm(audio_grad)    # 语音梯度范数

    if len(audio) == 0:                                       # 空音频防止 nan
        audio_grad_norm = 0
    else:
        audio_grad_norm = audio_grad_norm.detach().cpu().item()

    return adv_grad_norm.detach().cpu().item(), audio_grad_norm


def frame_level_saliency(audio, audio_attack_model, whisper_model, device):
    '''
        get the saliency per frame of attack segment and speech signal
    '''
    # 将两段信号的梯度拼接，得到逐帧显著性曲线
    adv_grad, audio_grad = _saliency_calculation(audio, audio_attack_model, whisper_model, device)
    saliencies = torch.abs(torch.cat((adv_grad, audio_grad), dim=0))  # 拼接并取绝对值
    return saliencies.detach().cpu()                                  # 返回 CPU 张量

def _saliency_calculation(audio, audio_attack_model, whisper_model, device):
    '''
        Forward-backward pass
    '''
    # 前向传播后对预测概率反向求导，获得输入及对抗段的梯度
    if isinstance(audio, str):
        audio = load_audio(audio)                                      # 如为路径则先读入
    audio = torch.from_numpy(audio).to(device)                         # 转为张量并放设备
    audio.requires_grad = True                                         # 开启梯度
    audio.retain_grad()                                                # 保留梯度

    audio_attack_model.eval()                                          # 评估模式

    # forward pass
    logits = audio_attack_model.forward(audio.unsqueeze(dim=0), whisper_model)[:,-1,:].squeeze(dim=1).squeeze(dim=0)  # 取最后一步 logits
    sf = torch.nn.Softmax(dim=0)                                       # softmax
    probs = sf(logits)                                                 # 概率分布
    pred_class = torch.argmax(probs)                                   # 预测类别
    prob = probs[pred_class]                                           # 对应概率

    # compute gradients
    prob.backward()                                                    # 反向传播
    adv_grad = audio_attack_model.audio_attack_segment.grad            # 取对抗段梯度
    audio_grad = audio.grad                                            # 取音频梯度

    return adv_grad, audio_grad                                        # 返回梯度


def get_decoder_proj_mat(whisper_model):
    '''
    Extract the final projection matrix used in the Whisper decoder to obtain the logits

    N.B. this projection matrix is the same as the token id to embedding matrix used in the input to the decoder
        This is standard for the Transformer decoder (refer to the attention is all you need paper)

    Should return W: Tensor [V x k]
        V = vocabulary size
        k = embedding size
    '''
    W = whisper_model.decoder.token_embedding.weight                   # decoder 投影矩阵
    return W

def get_rel_pos(W, real_token_ids, device=torch.device('cpu')):
    '''
    Return the similarity of each row vector (normalized) with every other row vector,
    where each row (r) vector is the rel_pos_vector for target token r.

    W: Tensor [V x k]
    real_token_ids: List[int] - List of real acoustic token ids

    Return rel_pos_matrix: Tensor [V x V]
    '''
    V = W.shape[0]                                                     # 词表大小
    W_norm = W / W.norm(dim=1, keepdim=True)                           # 行向量归一化

    # 计算所有词向量的两两余弦相似度
    rel_pos_matrix = torch.matmul(W_norm, W_norm.T).cpu()

    # 只保留真实语音 token 相关的列，其余置零
    mask = torch.zeros(V, V)                                           # 创建掩码
    mask[:, real_token_ids] = 1                                        # 仅真实语音列为 1
    rel_pos_matrix = rel_pos_matrix * mask                             # 应用掩码

    return rel_pos_matrix


def get_real_acoustic_token_ids(tokenizer, vocab_size):
    '''
    Identify real acoustic token ids based on the criteria that the token begins with a
    letter in the English alphabet or a numeral 1-9.

    tokenizer: Tokenizer object
    vocab_size: int - The size of the vocabulary

    Return real_token_ids: List[int]
    '''
    real_token_ids = []                                                # 存放真实语音 token id
    for token_id in range(vocab_size):                                 # 遍历词表
        token = tokenizer.decode([token_id])                           # 解码单个 token
        if token and is_real_acoustic_token(token):                    # 判断是否真实语音 token
            real_token_ids.append(token_id)
    return real_token_ids                                              # 返回列表

def is_real_acoustic_token(token):
    ''' Check if the token begins with a letter in the English alphabet or a numeral 1-9. '''
    return token[0] in string.ascii_letters or token[0] in '123456789' # 首字符为字母或 1-9
