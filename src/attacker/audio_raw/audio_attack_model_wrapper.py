import torch                                                          # 张量/设备
import torch.nn as nn                                                 # 神经网络层
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES, load_audio  # 音频处理

from .snr import calculate_snr

class AudioAttackModelWrapper(nn.Module):
    '''
        Whisper Model wrapper with learnable audio segment attack prepended to speech signals
    '''
    def __init__(self, tokenizer, attack_size=5120, device=None, attack_init='random'):
        super(AudioAttackModelWrapper, self).__init__()
        self.attack_size = attack_size                                # 对抗段长度
        self.tokenizer = tokenizer                                    # Whisper tokenizer
        self.device = device                                          # 设备
        self.multiple_model_attack = False                            # 是否针对集成模型

        self.sot_ids = self.tokenizer.sot_sequence_including_notimestamps  # SOT token 序列
        self.len_sot_ids = len(torch.tensor(self.sot_ids))            # SOT token 数量

        if attack_init == 'random':
            self.audio_attack_segment = nn.Parameter(torch.rand(attack_size))  # 随机初始化对抗段
        else:
            # load init attack from attack_init path
            loaded_params = torch.load(attack_init)                   # 从给定路径加载
            if 'audio_attack_segment' in loaded_params:
                initial_value = loaded_params['audio_attack_segment']
                self.audio_attack_segment = nn.Parameter(initial_value.to(device))  # 使用已有权重初始化
            else:
                raise ValueError("Invalid attack_init path provided.") # 缺少必要参数时报错
    
    def forward(self, audio_vector, whisper_model, decoder_input=None):
        '''
            audio_vector: Torch.tensor: [Batch x Audio Length]
            whisper_model: encoder-decoder model

            Returns the logits
        '''
        attacked_audio_vector, _ = self._prepend_attack_to_batch(audio_vector)       # 拼接得到攻击后音频

        # forward pass through full model
        mel = self._audio_to_mel(attacked_audio_vector, whisper_model)               # 转 mel 频谱
        return self._mel_to_logit(mel, whisper_model, decoder_input=decoder_input)   # 通过解码器得到 logits


    def _prepend_attack_to_batch(self, audio_vector: torch.Tensor):
        attack = self.audio_attack_segment.unsqueeze(0).expand(audio_vector.size(0), -1)
        attacked_audio_vector = torch.cat((attack, audio_vector), dim=1)
        clean_audio_vector = torch.cat((torch.zeros_like(attack), audio_vector), dim=1)
        return attacked_audio_vector, clean_audio_vector

    def _prepend_attack_to_audio(self, audio_vector: torch.Tensor):
        attacked_audio, clean_audio = self._prepend_attack_to_batch(audio_vector.unsqueeze(0))
        return attacked_audio.squeeze(0), clean_audio.squeeze(0)

    def compute_batch_snr(self, audio_vector: torch.Tensor):
        attacked_audio_vector, clean_audio_vector = self._prepend_attack_to_batch(audio_vector)
        return calculate_snr(clean_audio_vector, attacked_audio_vector)
    

    def _audio_to_mel(self, audio: torch.Tensor, whisper_model):
        '''
            audio: [Batch x Audio length]
            based on https://github.com/openai/whisper/blob/main/whisper/audio.py
        '''
        if self.multiple_model_attack:
            n_mels = whisper_model.models[0].dims.n_mels                             # 集成时取第一个模型的 mel 维度
        else:
            n_mels = whisper_model.model.dims.n_mels                                 # 单模型 mel 维度
        padded_mel = log_mel_spectrogram(audio, n_mels, padding=N_SAMPLES)           # 计算 log-mel 并补长
        mel = pad_or_trim(padded_mel, N_FRAMES)                                      # 截断/填充到固定帧
        return mel
    
    def _mel_to_logit(self, mel: torch.Tensor, whisper_model, decoder_input=None):
        '''
            Forward pass through the whisper model of the mel vectors
            expect mel vectors passed as a batch and padded to 30s of audio length
            mel: torch.Tensor [B x dim x num_vectors]
        '''
        # create batch of start of transcript tokens
        sot_ids = torch.tensor(self.sot_ids)
        sot_ids = sot_ids.to(self.device)
        decoder_input_ids = sot_ids.unsqueeze(0).expand(mel.size(0), -1)             # 扩展到 batch
        if decoder_input is not None:
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_input), dim=1) # 拼接附加 decoder 输入

        if self.multiple_model_attack:
            # pass through each target model
            sf = nn.Softmax(dim=-1)
            pred_probs = []
            for model in whisper_model.models:                                       # 对每个模型前向
                pred_probs.append(sf(model.forward(mel, decoder_input_ids)))         # softmax 概率
            return torch.mean(torch.stack(pred_probs), dim=0)                        # 取概率平均
        else:
            return whisper_model.model.forward(mel, decoder_input_ids)               # 单模型前向
    
    def transcribe(self,
        whisper_model,
        audio,
        do_attack=True,
        without_timestamps=False,
        return_snr=False,
    ):

        '''
            Mimics the original Whisper transcribe functions but prepends the adversarial attack
            in the audio space

                do_attack parameter is a boolean to do the attack or not
        '''
        snr = None
        if do_attack:
            # prepend attack
            if isinstance(audio, str):
                audio = load_audio(audio)                                            # 路径则先加载
            if isinstance(audio, torch.Tensor):
                audio_tensor = audio.to(self.device)
            else:
                audio_tensor = torch.from_numpy(audio).to(self.device)               # 转张量

            audio, clean_audio = self._prepend_attack_to_audio(audio_tensor)         # 拼接攻击段
            snr = calculate_snr(clean_audio, audio)                                  # 计算 SNR

        hyp = whisper_model.predict(audio, without_timestamps=without_timestamps)    # 调用封装模型预测

        if return_snr:
            return hyp, snr.item() if snr is not None else None
        return hyp


