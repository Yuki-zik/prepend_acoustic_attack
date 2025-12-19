import torch                                                         # 张量/设备
import torch.nn as nn                                                # 损失相关
from torch.utils.data import TensorDataset, DataLoader               # 数据加载
import random                                                        # 未用到
import os                                                            # 文件操作
from tqdm import tqdm                                                # 进度条
from whisper.audio import load_audio                                 # 读取音频

from .base import AudioBaseAttacker                                  # 基类
from src.tools.tools import AverageMeter                             # 统计平均



class AudioAttack(AudioBaseAttacker):
    '''
       Prepend adversarial attack in audio space -- designed to mute Whisper by maximizing eot token as first generated token
    '''
    def __init__(self, attack_args, whisper_model, device, lr=1e-3, multiple_model_attack=False, attack_init='random'):
        AudioBaseAttacker.__init__(self, attack_args, whisper_model, device, attack_init=attack_init)  # 初始化基类
        self.audio_attack_model.multiple_model_attack = multiple_model_attack                         # 是否集成攻击
        self.optimizer = torch.optim.AdamW(self.audio_attack_model.parameters(), lr=lr, eps=1e-8)     # 优化器


    def _loss(self, logits):
        '''
        The (average) negative log probability of the end of transcript token

        logits: Torch.tensor [batch x vocab_size]
        '''
        # 通过最大化 EOT 概率来推动模型尽快结束解码
        tgt_id = self._get_tgt_tkn_id()                                          # 目标 token id
        sf = nn.Softmax(dim=1)                                                   # softmax
        log_probs = torch.log(sf(logits))                                       # 对数概率
        tgt_probs = log_probs[:,tgt_id].squeeze()                               # 取目标列
        return -1*torch.mean(tgt_probs)                                         # 负均值作为损失
    

    def train_step(self, train_loader, epoch, print_freq=25):
        '''
            Run one train epoch - Projected Gradient Descent
        '''
        losses = AverageMeter()                                                  # 记录损失
        snrs = AverageMeter()                                                    # 记录 SNR

        # switch to train mode
        self.audio_attack_model.train()                                          # 训练模式

        for i, (audio) in enumerate(train_loader):
            audio = audio[0].to(self.device)                                     # 取 batch 音频到设备

            # Forward pass
            logits = self.audio_attack_model(audio, self.whisper_model)[:,-1,:].squeeze(dim=1)  # 仅取最后一步 logits
            loss = self._loss(logits)                                            # 计算损失

            # Backward pass and update
            self.optimizer.zero_grad()                                           # 梯度清零
            loss.backward()                                                      # 反向
            self.optimizer.step()                                                # 更新

            if self.attack_args.clip_val != -1:                                  # 可选幅度裁剪
                max_val = self.attack_args.clip_val
            else:
                max_val = 100000
            with torch.no_grad():  
                self.audio_attack_model.audio_attack_segment.clamp_(min=-1*max_val, max=max_val)  # 对抗段裁剪
        
            # record loss
            losses.update(loss.item(), audio.size(0))                            # 更新统计
            batch_snr = self.audio_attack_model.compute_batch_snr(audio)         # 计算当前 batch SNR
            snrs.update(batch_snr.mean().item(), audio.size(0))                  # 更新 SNR 统计
            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.5f} ({losses.avg:.5f})\tSNR {snrs.val:.2f} ({snrs.avg:.2f})')

        return losses.avg, snrs.avg


    @staticmethod
    def _prep_dl(data, bs=16, shuffle=False):
        '''
        Create batch of audio vectors
        '''

        print('Loading and batching audio files')
        audio_vectors = []
        for d in tqdm(data):
            audio_np = load_audio(d['audio'])                                    # 加载音频 numpy
            audio_vector = torch.from_numpy(audio_np)                            # 转张量
            audio_vectors.append(audio_vector)
        
        def pad_sequence(tensors, padding_value=0):
            max_length = max(len(tensor) for tensor in tensors)                  # 找最大长度
            padded_tensors = []
            for tensor in tensors:
                padded_tensor = torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=padding_value)  # 右侧补齐
                padded_tensors.append(padded_tensor)
            return padded_tensors

        audio_vectors = pad_sequence(audio_vectors)                              # 对齐长度
        audio_vectors = torch.stack(audio_vectors, dim=0)                        # 叠成 batch
        ds = TensorDataset(audio_vectors)                                        # 构造数据集
        dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=8)       # 构造数据加载器
        return dl


    def train_process(self, train_data, cache_dir):

        fpath = f'{cache_dir}/prepend_attack_models'                             # 模型保存根目录
        if not os.path.isdir(fpath):
            os.mkdir(fpath)

        train_dl = self._prep_dl(train_data, bs=self.attack_args.bs, shuffle=True)  # 构造数据加载器

        for epoch in range(self.attack_args.max_epochs):
            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr'])) # 打印学习率
            avg_loss, avg_snr = self.train_step(train_dl, epoch)                    # 单轮训练
            print(f'Epoch {epoch}: Average Loss {avg_loss:.5f}, Average SNR {avg_snr:.2f} dB')

            if epoch==self.attack_args.max_epochs-1 or (epoch+1)%self.attack_args.save_freq==0:
                # save model at this epoch
                if not os.path.isdir(f'{fpath}/epoch{epoch+1}'):
                    os.mkdir(f'{fpath}/epoch{epoch+1}')
                state = self.audio_attack_model.state_dict()                      # 提取状态
                torch.save(state, f'{fpath}/epoch{epoch+1}/model.th')             # 保存 checkpoint










            


