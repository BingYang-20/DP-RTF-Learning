""" 
    Function:   Define loss functions
	Author:     Bing Yang
    Copyright Bing Yang
"""

import numpy as np
import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, mic_type, mode='mse'):
        super(Loss, self).__init__()
        self.mic_type = mic_type
        self.mode = mode
        if self.mode =='ce': # Cross entropy loss
            self.ce = nn.CrossEntropyLoss()
        elif self.mode =='mse': # MSE loss
            self.mse = nn.MSELoss()
        else:
            assert False, 'Loss mode error ~'
        
    def forward(self, output, rtf, rtf_gt_set, target):
        # rtf/rtf_gt: (nbatch, 3nf, 1)
        # rtf_gt_set: (nbatch, 3nf, ndoa)
        label = target[:, 0, 0].long()
        if self.mode =='ce': 
            loss = self.ce(output, label)
        elif self.mode == 'mse': 
            if self.mic_type == 'MIT':
                temp = rtf_gt_set[:, label].permute(1, 0)
                rtf_gt = temp[:, :, np.newaxis]
                loss = self.mse(rtf, rtf_gt)  
            elif self.mic_type == 'CIPIC':
                nbatch, _, ndoa = rtf_gt_set.shape
                candid = torch.arange(ndoa).long()
                candid = candid[np.newaxis, :].expand(nbatch, ndoa).cuda()
                c = label[:, np.newaxis].expand(nbatch, ndoa)
                eq_temp = candid.eq(c).float()
                rtf_gt = torch.bmm(rtf_gt_set, eq_temp[:, :, np.newaxis])
                loss = self.mse(rtf, rtf_gt) 
 
        return loss

if __name__ == '__main__':
    # nbatch=100, ndoa=37, nf=128 
    output = torch.rand(100, 37).cuda()
    rtf = torch.rand(100, 384, 1).cuda() 
    rtf_gt_set = torch.rand(384, 37).cuda()
    target = torch.rand(100, 1, 1).cuda()
    MyLoss = Loss(mic_type='MIT', nn_mode='doa')
    loss_value = MyLoss(output=output, rtf=rtf, en_sig=None, rtf_gt_set=rtf_gt_set, target=target)
    print(loss_value)
