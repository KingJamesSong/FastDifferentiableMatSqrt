'''
Implementation of 'Temporal-attentive Covariance Pooling Networks for Action Recognition'
Authors: Zilin Gao, Qilong Wang, Bingbing Zhang, Qinghua Hu and Peihua Li.
'''

import pdb
import torch
import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill


class TCA(nn.Module):
    def __init__(self, dim, frame):
        super(TCA, self).__init__()
        self.dim = dim
        self.frame = frame

        print("-" * 20 + "TCA called" + '-' * 20)
        self.g1 = nn.Conv2d(
            self.dim, self.dim * 4, kernel_size=1, stride=1, padding=0
        )
        self.g2 = nn.Conv2d(
            self.dim * 4, self.dim, kernel_size=1, stride=1, padding=0
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, x_att):
        x = self.pool(x)
        [N, C, H, W] = x.size()
        V = N // self.frame
        x_t = x.view(V, self.frame, C, H, W)
        x_t1 = torch.cat((x_t[:,-1,:,:,:].unsqueeze(1), x_t[:,:-1,:,:,:,]), 1)
        x_t2 = torch.cat((x_t[:, -2:, :, :, :], x_t[:, :-2, :, :, :, ]), 1)

        x_diff1 = x_t - x_t1
        x_diff1 = x_diff1.view(N, C, H, W)
        x_diff2 = (x_t - x_t2)*0.5
        x_diff2 = x_diff2.view(N, C, H, W)

        x_diff1 = self.TCA_diff(x_diff1)
        x_diff2 = self.TCA_diff(x_diff2)

        return  x_att * (x_diff1  + x_diff2)*0.5


    def TCA_diff(self, x):
        x = self.g1(x)
        x = self.relu(x)
        x = self.g2(x)
        x = self.sigmoid(x)
        return x