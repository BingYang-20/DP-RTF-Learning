""" 
    Function:   Define models
	Author:     Bing Yang
    Copyright Bing Yang
"""

import torch
import torch.nn as nn
import os
import scipy.io
import numpy as np
from common.config import dirconfig

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class BasicBlock(nn.Module): 
    """ Function: Basic convolutional block
        reference: resnet, https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
    """
    # expansion = 1
    def __init__(self, inplanes, planes, stride=1, use_res=True, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.use_res = use_res

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_res == True:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out

class RTFLearn(nn.Module):
    def __init__(self, inplanes_mag, inplanes_phase, planes, res_flag, rnn_in_dim, rnn_hid_dim, rnn_out_dim, rnn_bdflag = False, archit='CRNN_our'):
        super(RTFLearn, self).__init__()
        self.archit = archit
        if (self.archit == 'CRNN_conv1x1'):
            self.conv1 = nn.Sequential(
                conv1x1(inplanes_mag, int(planes/2)),
                # nn.BatchNorm2d(planes),
            )
            self.conv2 = nn.Sequential(
                conv1x1(inplanes_phase, int(planes/2)),
                # nn.BatchNorm2d(planes),
            )

        if (self.archit == 'CRNN_baseline'):
            inplanes = inplanes_mag + inplanes_phase
            self.layer0 = nn.Sequential(
                conv3x3(inplanes, planes),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                conv3x3(planes, planes),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            self.layer2 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            self.layer3 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            self.layer4 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            self.layer5 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )

        elif (archit == 'CRNN_our'):
            self.layer_gate = nn.Sequential(
                conv3x3(inplanes_mag, int(planes/2)),
                nn.BatchNorm2d(int(planes/2)),
                nn.ReLU(inplace=True),
                conv3x3(int(planes/2), int(planes/2)),
                nn.BatchNorm2d(int(planes/2)),
                nn.Sigmoid(),
            )
            self.layer_dm = nn.Sequential(
                conv1x1(inplanes_mag, int(planes/2)),
            )
            self.layer_dm_2 = nn.Sequential(
                conv1x1(int(planes/2), int(planes/2)),
                nn.BatchNorm2d(int(planes/2)),
                nn.ReLU(inplace=True),
            )
            self.layer_dp = nn.Sequential(
                conv1x1(inplanes_phase, int(planes/2)),
            )
            self.layer_dp_2 = nn.Sequential(
                conv1x1(planes, int(planes/2)),
                nn.BatchNorm2d(int(planes/2)),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            self.layer2 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            self.layer3 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            self.layer4 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            self.layer5 = nn.Sequential(
                BasicBlock(inplanes=planes, planes=planes, use_res=res_flag),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )

        self.rnn_bdflag = rnn_bdflag
        if rnn_bdflag:
            rnn_ndirection = 2
        else:
            rnn_ndirection = 1
        self.rnn = torch.nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=1, batch_first=True, bias=True, dropout=0.4, bidirectional=rnn_bdflag)
        self.rnn_fc = nn.Sequential(
            torch.nn.Linear(in_features=rnn_ndirection * rnn_hid_dim, out_features=rnn_out_dim),  # bias=False
            nn.Tanh(),
        )

    def forward(self, mag, phase):
        if (self.archit == 'CRNN_cov1x1'):
            magd = self.conv1(mag[:, 0:2,:,:])
            magd = torch.tanh(magd)
            phased = self.conv2(phase[:, 0:2,:,:])
            sinphased = torch.sin(phased)
            cosphased = torch.cos(phased)
            fea = torch.cat((magd, sinphased, cosphased), 1)

        elif (self.archit == 'CRNN_baseline'):
            fea = torch.cat((mag[:, 0:2, :, :], phase[:, 0:2, :, :]), 1)
            fea = self.layer0(fea)
            fea = self.layer1(fea)
            fea = self.layer2(fea)
            fea = self.layer3(fea)
            fea = self.layer4(fea)
            fea = self.layer5(fea)

        elif (self.archit == 'CRNN_our'):
            gate = self.layer_gate(mag)

            fea_magd = self.layer_dm(mag)
            fea_magd = torch.tanh(fea_magd)
            fea_magd = self.layer_dm_2(fea_magd)

            fea_phased = self.layer_dp(phase)
            fea_phased = torch.cat((torch.sin(fea_phased), torch.cos(fea_phased)), 1)
            fea_phased = self.layer_dp_2(fea_phased)
            
            fea = torch.cat((fea_magd*gate, fea_phased*gate), 1)

            fea = self.layer1(fea)
            fea = self.layer2(fea)
            fea = self.layer3(fea)
            fea = self.layer4(fea)
            fea = self.layer5(fea)
        
        fea_cnn = fea.view(fea.size(0), -1, fea.size(3))  # (nb, nch*nf, nt)
        fea_cnn = fea_cnn.permute(0, 2, 1)
        fea_rnn, _ = self.rnn(fea_cnn)
        if self.rnn_bdflag:
            fea_rnn_fc = self.rnn_fc(fea_rnn[:, round(fea_rnn.size(3)/2), :])
        else:
            fea_rnn_fc = self.rnn_fc(fea_rnn[:, -1, :])
        output = fea_rnn_fc[:, :, np.newaxis]

        return output

class CRNN(nn.Module):
    def __init__(self, mic_type):
        super(CRNN, self).__init__()

        self.mic_type = mic_type
        self.out_feature = 'rtf' # 'rtf' for RTF regression, 'onehot' for location classification
        archit = 'CRNN_our' #  value: 'CRNN_conv1x1' 'CRNN_baseline' 'CRNN_our'

        # load DP-RTF template
        ger_dirs = dirconfig()
        rtf_dir = ger_dirs['rtf']
        if self.mic_type == 'MIT':
            path = os.path.join(rtf_dir, 'rtf_dp_Hkemar.mat') # test507030_dist140
            rtf_dp = scipy.io.loadmat(path)['rtf_dp']   # (3nf, ndoa)
            rtf_dp = torch.from_numpy(rtf_dp.astype('float32')).cuda()
            self.rtf_dp_set = rtf_dp
            self.rtf_dp = None
        elif self.mic_type == 'CIPIC':
            path = os.path.join(rtf_dir, 'rtf_dp_set_ori.mat')
            rtf_dp_set = scipy.io.loadmat(path)['rtf_dp']   # (3nf, ndoa, nhead)
            rtf_dp_set = torch.from_numpy(rtf_dp_set.astype('float32')).cuda()
            mic_idx_set = scipy.io.loadmat(path)['mic_idx']  # (1, nhead)
            mic_idx_set = torch.from_numpy(mic_idx_set.astype('float32')).cuda()
            self.rtf_dp_set = rtf_dp_set
            self.mic_idx_set = mic_idx_set
            self.rtf_dp = None

        # learn DP-RTF
        rtf_dim, ndoa, _ = self.rtf_dp_set.shape
        self.rtflearn_block = RTFLearn(inplanes_mag=2, inplanes_phase=2, planes=64, res_flag=False, rnn_in_dim=256, rnn_hid_dim=256, rnn_out_dim=rtf_dim, rnn_bdflag = False, archit=archit)

        if self.out_feature == 'onehot':
            self.fc = nn.Sequential(
                # nn.Dropout(0.5),
                torch.nn.Linear(in_features=rtf_dim, out_features=ndoa),
            )

    def forward(self, data, mic_idx_est = None):
        nbatch = data.shape[0]  # data: (nb, -, nf, nt)

        mag = data[:, 0:2, :, :]
        phase = data[:, 2:4, :, :]  
        mic_label = data[:, -1:, 0, 0]
        mag_log = torch.log10(mag + 0.00001)

        out_rtf = self.rtflearn_block(mag_log, phase)
                
        if self.out_feature == 'rtf':
            if self.mic_type == 'MIT':
                self.rtf_dp = self.rtf_dp_set[np.newaxis, :, :].repeat(nbatch, 1, 1) # (nbatch, 3nf, ndoa)
            elif self.mic_type == 'CIPIC':
                if mic_idx_est == None:
                    mic_idx = mic_label.expand(nbatch, self.mic_idx_set.shape[1])
                    mic_idx_set = self.mic_idx_set.expand(nbatch, self.mic_idx_set.shape[1])
                    eq_result = mic_idx.eq(mic_idx_set).float()
                    self.rtf_dp = torch.matmul(self.rtf_dp_set, eq_result.permute([1, 0])).permute([2, 0, 1]) # (nbatch, 3nf, ndoa)
                elif mic_idx_est == 'mean':
                    rtf_temp = torch.mean(self.rtf_dp_set, dim=2)
                    self.rtf_dp = rtf_temp.unsqueeze(0).expand(nbatch, rtf_temp.shape[0], rtf_temp.shape[1]) # (nbatch, 3nf, ndoa)
                else:
                    # select: mic_idx_est
                    rtf_temp_set = torch.zeros((nbatch,self.rtf_dp_set.shape[0],self.rtf_dp_set.shape[1], len(mic_idx_est))).cuda()
                    for idx in range(0,len(mic_idx_est),1):
                        mic_idx = mic_idx_est[idx]*torch.ones((nbatch, self.mic_idx_set.shape[1])).cuda()
                        mic_idx_set = self.mic_idx_set.expand(nbatch, self.mic_idx_set.shape[1])
                        eq_result = mic_idx.eq(mic_idx_set).float()
                        rtf_temp_set[:, :, :, idx] = torch.matmul(self.rtf_dp_set, eq_result.permute([1, 0])).permute([2, 0, 1])
                    self.rtf_dp = torch.mean(rtf_temp_set, dim=3) 

            match_flag = 'l2' # 'l2' for L2 norm, 'inner' for inner product
            fea = out_rtf * 1  # (nbatch, 3nf, nt)
            fea_gt = self.rtf_dp * 1 # (nbatch, 3nf, ndoa)
            if match_flag == 'inner':
                sim = torch.bmm(fea_gt.permute(0, 2, 1), fea) # (nbatch, ndoa, nt)
                out_onehot = torch.mean(sim, 2)
            elif match_flag == 'l2':
                sim = pow(abs(fea_gt - fea.expand(nbatch, fea.shape[1], fea_gt.shape[2])), 2).permute(0, 2, 1) * (-1)
                out_onehot = torch.sum(sim, 2)

        elif self.out_feature == 'onehot':
            out_onehot = self.fc(out_rtf[:, :, 0]) 

        return out_onehot, out_rtf, self.rtf_dp


if __name__ == '__main__':
    x = torch.rand(100, 9, 128, 31).cuda()  # (nbatch, nch, nf, nt)
    Net = CRNN(mic_type='CIPIC').to('cuda')
    y1, y2, y3 = Net(x)
    print(x.size(), y1.size(), y2.size(), y3.size())