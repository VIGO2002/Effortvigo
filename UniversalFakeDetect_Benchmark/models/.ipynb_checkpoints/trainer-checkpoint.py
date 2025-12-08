import functools
import torch
import torch.nn as nn
from .base_model import BaseModel, init_weights
import sys
from models import get_model


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = get_model(opt.arch, opt)
        self.lr = opt.lr
        
        # [修改] 遍历 EBM_Head 中的所有线性层进行初始化
        for m in self.model.fc.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.0, opt.init_gain)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            # import time 
            # time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        #self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.8
            self.lr = param_group['lr']
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()  # 前向传播，计算出 self.output (即能量值)

        # 获取能量和标签
        # self.output shape: [batch_size, 1] -> squeeze -> [batch_size]
        energy = self.output.squeeze(1)
        label = self.label  # 0 for real, 1 for fake

        # --- EBM Hinge Loss 计算开始 ---
        m_real = -10.0  # 真图能量上界
        m_fake = 10.0  # 假图能量下界

        # 1. 计算真图损失: max(0, energy - m_real)
        # 只有当真图能量 > -10 时才会有 Loss
        real_energies = energy[label == 0]
        if len(real_energies) > 0:
            loss_real = torch.clamp(real_energies - m_real, min=0).mean()
        else:
            # 如果这一个 Batch 里全是假图，真图 Loss 为 0
            loss_real = torch.tensor(0.0).to(self.device)

        # 2. 计算假图损失: max(0, m_fake - energy)
        # 只有当假图能量 < 10 时才会有 Loss
        fake_energies = energy[label == 1]
        if len(fake_energies) > 0:
            loss_fake = torch.clamp(m_fake - fake_energies, min=0).mean()
        else:
            # 如果这一个 Batch 里全是真图，假图 Loss 为 0
            loss_fake = torch.tensor(0.0).to(self.device)

        # 总损失
        self.loss = loss_real + loss_fake
        # --- EBM Hinge Loss 计算结束 ---

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



