from torch.nn.parameter import Parameter
import torch
import torch.nn as nn

from torch_utils import *

eps=1e-10

class BatchNorm(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-2, momentum=0.1, affine=True):
        super(BatchNorm, self).__init__()
        print('BatchNorm Group Num {}'.format(groups))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(groups, int(num_features/groups), 1))
        self.bias = Parameter(torch.Tensor(groups, int(num_features/groups), 1))
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        self.register_buffer("running_subspace", torch.eye(length, length).view(1,length,length).repeat(self.groups,1,1))

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            x = x.transpose(0,1).contiguous().view(C, -1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1,keepdim=True)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum)  * mean + self.momentum * self.running_mean
                self.running_var = (1 - self.momentum)  * var + self.momentum * self.running_mean
            G = self.groups
            n_mem = C // G
            x = x.view(G, n_mem, -1)
            mu = x.mean(2, keepdim=True)
            xx = torch.bmm((x-mu), (x-mu).transpose(1, 2)) / (N * H * W) + torch.eye(n_mem, out=torch.empty_like(x)).unsqueeze(
                0) * self.eps
            x = (x-mean)/torch.sqrt(var+1e-10) * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x
        else:
            N, C, H, W = x.size()
            x = x.transpose(0,1).contiguous().view(C, -1)
            x = (x - self.running_mean) / torch.sqrt(self.running_var+eps)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class ZCANormBatch(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-2, momentum=0.1, affine=True):
        super(ZCANormBatch, self).__init__()
        print('ZCANormBatch Group Num {}'.format(groups))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(groups, int(num_features/groups), 1))
        self.bias = Parameter(torch.Tensor(groups, int(num_features/groups), 1))
        #Matrix Square Root or Inverse Square Root layer
        self.svdlayer = MPA_Lya.apply
        self.register_buffer('running_mean', torch.zeros(groups, int(num_features/groups), 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        self.register_buffer("running_subspace", torch.eye(length, length).view(1,length,length).repeat(self.groups,1,1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            n_mem = C // G
            x = x.transpose(0, 1).contiguous().view(G, n_mem, -1)
            mu = x.mean(2, keepdim=True)
            x = x - mu
            xxt = torch.bmm(x, x.transpose(1,2)) / (N * H * W) + torch.eye(n_mem, out=torch.empty_like(x)).unsqueeze(0) * self.eps
            assert C % G == 0
            subspace = torch.inverse(self.svdlayer(xxt))
            xr = torch.bmm(subspace, x)
            with torch.no_grad():
                running_subspace = self.__getattr__('running_subspace')
                running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr

        else:
            N, C, H, W = x.size()
            G = self.groups
            n_mem = C // G
            x = x.transpose(0, 1).contiguous().view(G, n_mem, -1)
            x = (x - self.running_mean)
            subspace = self.__getattr__('running_subspace')
            x= torch.bmm(subspace, x)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormBatch, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

