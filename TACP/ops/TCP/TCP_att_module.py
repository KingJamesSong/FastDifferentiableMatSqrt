'''
Implementation of 'Temporal-attentive Covariance Pooling Networks for Action Recognition'
Authors: Zilin Gao, Qilong Wang, Bingbing Zhang, Qinghua Hu and Peihua Li.
'''

import pdb
import torch
import torch.nn as nn
# from fvcore.nn.weight_init import c2_msra_fill
from ops.TCP.TSA import TSA
from ops.TCP.TCA import TCA

class TCP_att_module(nn.Module):

    def __init__(
        self,
        dim,
        frame=8,
        ch_flag=True,
        sp_flag=True,
        conv_1d_flag=True,
    ):
        """
        Args:
            dim (int): TCP_dim.
            frame (int): frame number.
            ch_flag (bool): whether use temporal-based channel attention, default: True.
            sp_flag (bool): whether use temporal-based spatial attention, default: True.
            conv_1d_flag (bool): whether use temporal convolution, default: True.

        """
        super(TCP_att_module, self).__init__()
        self.dim = dim
        self.frame = frame
        self.ch_flag = ch_flag
        self.sp_flag = sp_flag
        self.conv_1d_flag = conv_1d_flag
        self.k = frame // 2 + 1 #temporal conv kernel
        self._construct_TCP_att(
        )

    def _construct_TCP_att(self
    ):
        self.relu = nn.ReLU(inplace=True)

        if self.ch_flag:
            self.TCA = TCA(self.dim, self.frame)

        if self.conv_1d_flag :
            print("-" * 20 + "1D conv Module called" + '-' * 20)
            if (torch.__version__).split('.')[1] >= '5': #version 1.5+
                k_padding = (self.k-1)//2
            else :
                k_padding  = self.k-1
            self.conv_1d = nn.Conv3d(
                self.dim, self.dim, kernel_size=(self.k,1,1),
                stride=1, padding=(k_padding,0,0),
                padding_mode='circular'
            )

        if self.sp_flag:
            print("-" * 20 + "TSA Module called" + '-' * 20)
            self.TSA = TSA(self.dim, self.frame)

        self.init_weights()



    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    #c2_msra_fill(m)
                    m.weight.data.normal_(mean=0.0, std=0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d) \
                        or isinstance(m, nn.BatchNorm3d):
                    batchnorm_weight = 0.
                    if m.weight is not None:
                        m.weight.data.fill_(batchnorm_weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
            if self.conv_1d_flag:
                self.conv_1d.weight.data.fill_(0.)
            if self.sp_flag:
                self.TSA.init_weights()

    def forward(self, x):

        if self.sp_flag:
            x_att = self.TSA(x)
        else :
            x_att = x

        if self.ch_flag :
            x_att = self.TCA(x, x_att)

        if self.conv_1d_flag:
            [N, C, H, W] = x_att.size()
            V = N // self.frame
            x_att = x_att.reshape(V, self.frame, C, H, W).permute(0,2,1,3,4)
            x_att = self.conv_1d(x_att)
            x_att = x_att.permute(0,2,1,3,4,).reshape(N, C, H, W)
            return self.relu(x_att + x)
        else :
            return x_att  # following SE, w/o shortcut


