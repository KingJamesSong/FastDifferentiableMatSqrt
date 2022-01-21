import torch
import torch.nn as nn
import torch.nn.functional as F
from .MPN import Triuvec
from torch.autograd import Function
import numpy as np


class svPN(nn.Module):
    def __init__(self,
                alpha=0.5,
                iterNum=1,
                svNum=1,
                init_weight='learn', 
                vec='full',
                input_dim=2048,
                regular=None,
        ):
        super(svPN, self).__init__()
        self.iterNum=iterNum
        self.svNum = svNum
        self.init_weight=init_weight
        self.alpha = alpha
        self.vec = vec
        self.input_dim = input_dim
        self.regular = regular
        if self.vec is not None:
            if self.vec == 'triu':
                self.output_dim = int(self.input_dim[0]*(self.input_dim[1]+1)/2)
            elif self.vec == 'full':
                self.output_dim = int(self.input_dim[0]*self.input_dim[1])
        else:
            self.output_dim = (self.input_dim, self.input_dim)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.init_weight == 'learn':
            self.weight = nn.Parameter(torch.randn(self.input_dim[0]))
        else:
            self.weight = None

    def _remove_mean(self, x):
        _mean = F.adaptive_avg_pool2d(x, (1,1))
        x = x - _mean
        return x
        
    def _singularValuePowerNorm(self, x, top=None):
        batchsize, d, N = x.size()
        x = x.view(x.size(0), d, N).transpose(1, 2)  # bs, N, d
        if top is None:
            top = min(N, d)
            norm_remain = False
        else:
            norm_remain = True
        U = torch.zeros(batchsize, N, top, device=x.device)
        S = torch.zeros(batchsize, top, top, device=x.device)
        V = torch.zeros(batchsize, d, top, device=x.device)
        for i in range(top):
            u, s, v = self._powerIteration(x, uv=True)
            U[:,:,i], S[:,i,i],V[:,:,i] = u.squeeze(-1), s.view(-1), v.squeeze(-1)
            x = x - u.bmm(s).bmm(v.transpose(1, 2)) # u*s*v.^{T}
        y = U.bmm(S.pow(self.alpha)).bmm(V.transpose(1,2)) # U*S.^{\alpha}*V.^{T}
        if norm_remain:
            y = y + x.div(s.pow(1 - self.alpha) + 1e-5) 
        y = y.transpose(1, 2)
        return y

    def _powerIteration(self, x, uv=False):
        batchsize, N, d = x.shape
        if self.init_weight == 'learn':
            _v = self.weight.view(1, d, 1).expand(batchsize, d, 1)
        elif self.init_weight =='one':
            _v = torch.ones(batchsize, d, 1, device = x.device)
        for i in range(self.iterNum):
            _u = F.normalize(x.bmm(_v), dim=1)
            _v = F.normalize(x.transpose(1,2).bmm(_u), dim=1)
        spect = _u.transpose(1,2).bmm(x).bmm(_v)
        if uv:
            return _u, spect, _v
        else:
            return spect
    
    def _triuvec(self, x):
         return Triuvec.apply(x)

    def forward(self, x):
        x = self._singularValuePowerNorm(x, top=self.svNum)                          
        if self.vec == 'full':
            x = x.reshape(x.shape[0], -1)
        elif self.vec == 'triu':
            x = self._triuvec(x)
        if self.regular is not None:
            x = self.regular(x)
        return x