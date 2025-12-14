import torch                                                             # 张量/设备
import torch.nn as nn                                                    # 网络层
from whisper.audio import load_audio                                     # 音频读取
import torchaudio                                                        # 保存临时音频
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES, load_audio  # mel 处理


class AudioAttackCanaryModelWrapper(nn.Module):
    '''
        Canary Model wrapper with learnable audio segment attack prepended to speech signals
    '''
    def __init__(self, tokenizer, attack_size=5120, device=None, attack_init='random'):
        super(AudioAttackCanaryModelWrapper, self).__init__()
        self.attack_size = attack_size                                    # 对抗段长度
        self.tokenizer = tokenizer                                        # Canary tokenizer
        self.device = device                                              # 设备

        self.len_sot_ids = 5 # always 5 for canary model                  # Canary 固定 5 个起始 token

        if attack_init == 'random':
            self.audio_attack_segment = nn.Parameter(torch.rand(attack_size))  # 随机初始化
        else:
            # load init attack from attack_init path
            loaded_params = torch.load(attack_init)                       # 读取已有参数
            if 'audio_attack_segment' in loaded_params:
                initial_value = loaded_params['audio_attack_segment']
                self.audio_attack_segment = nn.Parameter(initial_value.to(device))  # 使用预训练对抗段
            else:
                raise ValueError("Invalid attack_init path provided.")     # 未找到参数报错

    def lens_to_mask(self, lens, max_length):
        # 将长度向量转换为掩码，供 Transformer 解码器使用
        if isinstance(lens, int):
            lens = torch.tensor([lens]).to(self.device)                   # 单值转张量
        batch_size = lens.shape[0]
        mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]  # 生成 <= 长度的布尔掩码
        return mask

    def _audio_to_mel(self, audio: torch.Tensor):
        '''
            audio: [Batch x Audio length]
            based on https://github.com/openai/whisper/blob/main/whisper/audio.py
        '''
        n_mels = 128                                                      # Canary 采用 128 mel 维度
        padded_mel = log_mel_spectrogram(audio, n_mels, padding=N_SAMPLES)  # 计算 log-mel
        mel = pad_or_trim(padded_mel, N_FRAMES)                           # 对齐长度
        return mel
    
    def forward(self, audio_vector, canary_model, decoder_input=None):
        '''
            audio_vector: Torch.tensor: [Batch x Audio Length]
            canary_model: encoder (conformer) - Transformer decoder model

            Returns the logits
        '''
        # prepend attack segment
        X = self.audio_attack_segment.unsqueeze(0).expand(audio_vector.size(0), -1)  # 扩展攻击段到 batch
        attacked_audio_vector = torch.cat((X, audio_vector), dim=1)                  # 拼接攻击段

        # Forward pass through preprocessor（自定义 mel 以保持可微）
        processed_signal = self._audio_to_mel(attacked_audio_vector)                 # 转 mel
        processed_signal_length = torch.tensor([processed_signal.size(2)] * processed_signal.size(0)).to(self.device)  # 构造长度

        # Forward pass through encoder
        encoded, encoded_len = canary_model.model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )                                                                           # 通过编码器

        # Project encoder outputs if necessary
        enc_states = encoded.permute(0, 2, 1)                                       # 调整维度
        enc_states = canary_model.model.encoder_decoder_proj(enc_states)            # 线性投影
        enc_mask = self.lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)  # 编码器掩码
        if canary_model.model.use_transf_encoder:
            enc_states = canary_model.model.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)  # 可选额外 encoder

        # Ensure the decoder input starts with the appropriate start-of-transcript tokens
        sot_ids = torch.tensor(canary_model.sot_ids).to(self.device)                # SOT token
        decoder_input_ids = sot_ids.unsqueeze(0).expand(enc_states.size(0), -1)     # 扩展到 batch
        
        if decoder_input is not None:
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_input), dim=1) # 拼接教师强制输入

        # Create decoder mask
        dec_mask = self.lens_to_mask(decoder_input_ids.size(1), decoder_input_ids.size(1)).to(decoder_input_ids.dtype)  # 解码器掩码
        dec_states = canary_model.model.transf_decoder(
            input_ids=decoder_input_ids, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask
        )                                                                           # 通过解码器
        transf_log_probs = canary_model.model.log_softmax(hidden_states=dec_states) # 得到 log probs

        return transf_log_probs
    
    
    def transcribe(self,
        canary_model,
        audio,
        do_attack=True,
    ):

        '''
            Mimics the original Canary transcribe functions but prepends the adversarial attack
            in the audio space

                do_attack parameter is a boolean to do the attack or not
        '''
        if do_attack:
            # prepend attack
            if isinstance(audio, str):
                audio = load_audio(audio)                                            # 路径则加载
            audio = torch.from_numpy(audio).to(self.device)                          # 转张量
            audio = torch.cat((self.audio_attack_segment, audio), dim=0)             # 拼接攻击段

            # Ensure audio tensor is in the correct shape [channels, samples]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # Add a channel dimension for mono audio # 单通道扩一维
            sample_rate = 16000
            torchaudio.save('experiments/temp_audio.wav', audio.cpu(), sample_rate)   # 保存临时 wav，供 Nemo 读取

            return canary_model.predict('experiments/temp_audio.wav')                # 使用 Canary 预测
        return canary_model.predict(audio)                                           # 不攻击直接预测


