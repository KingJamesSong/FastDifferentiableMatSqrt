import torch
import torch.nn as nn
from torch.autograd import Function
from mpmath import *
import numpy as np
mp.dps = 32
one = mpf(1)
mp.pretty = True

def f(x):
    return sqrt(one-x)
a = taylor(f, 0, 10)
pade_p, pade_q = pade(a, 5, 5)
a = torch.from_numpy(np.array(a).astype(float)).half()
pade_p = torch.from_numpy(np.array(pade_p).astype(float)).half()
pade_q = torch.from_numpy(np.array(pade_q).astype(float)).half()

def taylor_expansion(p, I):
    p_sqrt= I
    p_app = I - p
    p_hat = p_app
    for i in range(10):
      p_sqrt += a[i+1]*p_hat
      p_hat = p_hat.bmm(p_app)
    return p_sqrt

def pade_approximant(p,I):
    p_sqrt = pade_p[0]*I
    q_sqrt = pade_q[0]*I
    p_app = I - p
    p_hat = p_app
    for i in range(5):
        p_sqrt += pade_p[i+1]*p_hat
        q_sqrt += pade_q[i+1]*p_hat
        p_hat = p_hat.bmm(p_app)
    return torch.linalg.solve(q_sqrt.float(),p_sqrt.float()).half()

class PadeSqt(nn.Module):
     """
     Matrix Square Root calculated by Matrix Pade Approximant or Matrix Taylor Polynomial

     Args:
         iterNum: #iteration of Newton-schulz method
         is_sqrt: whether perform matrix square root or not
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
         dimension_reduction: if None, it will not use 1x1 conv to
                               reduce the #channel of feature.
                              if 256 or others, the #channel of feature
                               will be reduced to 256 or others.
     """
     def __init__(self, input_dim=[256,256]):

        super(PadeSqt, self).__init__()
        #self.output_dim = int(input_dim*(input_dim+1)/2)
        self.input_dim = input_dim
        self.output_dim = int(self.input_dim[0] * (self.input_dim[1] + 1) / 2)

     def _sqrtm(self, x):
         return Sqrtm.apply(x)
     def _triuvec(self, x):
         return Triuvec.apply(x)

     def forward(self, x):
        x = self._sqrtm(x)
        x = self._triuvec(x)
        x = x.view(x.shape[0], -1)
        return x


class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         dtype = x.dtype
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device, dtype=dtype) \
              + (1./M)*torch.eye(M,M,device = x.device, dtype=dtype)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         grad_output = grad_output.type(x.dtype)
         I_hat = I_hat.type(x.dtype)
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input


class Sqrtm(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        normx = torch.norm(x, dim=[1, 2]).view(x.size(0), 1, 1)
        I = torch.eye(x.size(1), x.size(2), requires_grad=False, device=x.device).reshape(1, x.size(1),x.size(2)).repeat(x.size(0),1, 1).type(dtype)
        y = pade_approximant(x / normx, I)
        y = y * torch.sqrt(normx)
        ctx.save_for_backward(input, y, normx, I)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        M, M_sqrt, normM, I = ctx.saved_tensors
        b = M_sqrt / torch.sqrt(normM)
        c = grad_output / torch.sqrt(normM)
        for i in range(8):
            # In case you might terminate the iteration by checking convergence
            # if th.norm(b-I)<1e-4:
            #    break
            b_2 = b.mm(b)
            c = 0.5 * (c.mm(3.0 * I - b_2) - b_2.mm(c) + b.mm(c).mm(b))
            b = 0.5 * b.mm(3.0 * I - b_2)
        grad_input = 0.5 * c
        return grad_input


class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().reshape(dim*dim)
         index = I.nonzero(as_tuple=False)
         y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device).type(dtype)
         y = x[:,index]
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = grad_output.dtype
         grad_input = torch.zeros(batchSize,dim*dim,device = x.device,requires_grad=False).type(dtype)
         grad_input[:,index] = grad_output
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var):
    return Sqrtm.apply(var)

def TriuvecLayer(var):
    return Triuvec.apply(var)
