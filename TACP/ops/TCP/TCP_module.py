'''
Implementation of 'Temporal-attentive Covariance Pooling Networks for Action Recognition'
Authors: Zilin Gao, Qilong Wang, Bingbing Zhang, Qinghua Hu and Peihua Li.
'''

import torch
import torch.nn as nn
from ops.MPNCOV import MPNCOV
import time
import pdb

class TCP(nn.Module):
     def __init__(self, dim_in=2048, dim_out=128, num_segments=8, level='video',
                  sp_flag=True, ch_flag=True, conv_1d_flag=True):
         """
         Args:
             dim_in (int): input dimension of TCP dimension reduction layer, default: 2048.
             dim_out (int): TCP_dim, default: 128.
             num_segments (int): frame number.
             level (str): TCP level: 'video' or 'frame'.
             sp_flag (bool): whether use temporal-based spatial attention, default: True.
             ch_flag (bool): whether use temporal-based channel attention, default: True.
             conv_1d_flag (bool): whether use temporal convolution, default: True.

         """

         super(TCP, self).__init__()
         print("-"*20 + "Temporal-attentive Covariance Pooling called" + '-'*20)
         self.level = level
         self.sp_flag = sp_flag
         self.ch_flag = ch_flag
         self.conv_1d_flag = conv_1d_flag
         dim_inner = 256 # dimension of first dimension reduction layer, dim_in -> dim_inner -> TCP_dim
         self.num_segments = num_segments
         self.layer_reduce1 = nn.Conv2d(
             dim_in,
             dim_inner,
             kernel_size=1,
             stride=[1,1],
             padding=0,
             bias=False,
         )

         self.layer_reduce_bn1 = nn.BatchNorm2d(
             num_features=dim_inner,
         )

         self.layer_reduce2 = nn.Conv2d(
             dim_inner,
             dim_out,
             kernel_size=1,
             stride=[1,1],
             padding=0,
             bias=False,
         )

         self.layer_reduce_bn2 = nn.BatchNorm2d(
             num_features=dim_out,
         )

         self.relu_op = nn.ReLU(inplace=True)

         if self.sp_flag or self.ch_flag or self.conv_1d_flag:
             from ops.TCP.TCP_att_module import TCP_att_module as att_module
             self.TCP_att = att_module(
                 dim_out, frame=num_segments,
                 sp_flag=sp_flag, ch_flag=ch_flag, conv_1d_flag=conv_1d_flag)
         else :
             # w/o att module, only matrix normalization
             from ops.Identity import Identity as att_module
             self.TCP_att = att_module()

         self.parameter_init()

     def parameter_init(self):
         for n, m in self.named_modules():
             if m == self.TCP_att :
                 # print('-----skip att module initialization in TCP module---')
                 return 0
             elif isinstance(m, nn.Conv2d):
                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                 # print(n + '  layer  kaiming initialed ')
                 if hasattr(m, 'bias'):
                     if m.bias is not None:
                         nn.init.zeros_(m.bias)
             elif isinstance(m, nn.BatchNorm2d):
                     nn.init.ones_(m.weight)
                     nn.init.zeros_(m.bias)
                     print(n + '  layer  initialed ')



     def forward(self, input):

         x = self.layer_reduce1(input)
         x = self.layer_reduce_bn1(x)
         x = self.relu_op(x)

         x = self.layer_reduce2(x)
         x = self.layer_reduce_bn2(x)
         x = self.relu_op(x)

         # attention module
         x = self.TCP_att(x)

         if self.level == 'video':
             x = x.view((-1, self.num_segments) + x.size()[1:])
             [bs, fr, dim, h, w] = x.size()
             x = x.permute(0,2,1,3,4)
             x = x.reshape(bs, dim, fr*h, w)

         #P_{TCP}
         x = MPNCOV.CovpoolLayer(x)
         # fast matrix power normalization, P_{TCP}^{1/2}
         #x = MPNCOV.SqrtmLayer(x, 3)
         x = MPNCOV.MPALayer(x)
         x = MPNCOV.TriuvecLayer(x)
         x = x.unsqueeze(-1)

         return x
