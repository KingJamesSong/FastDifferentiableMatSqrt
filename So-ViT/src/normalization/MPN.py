'''
@file: MPNCOV.py
@author: Jiangtao Xie
@author: Peihua Li
Please cite the paper below if you use the code:

Peihua Li, Jiangtao Xie, Qilong Wang and Zilin Gao. Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 947-955, 2018.

Peihua Li, Jiangtao Xie, Qilong Wang and Wangmeng Zuo. Is Second-order Information Helpful for Large-scale Visual Recognition? IEEE Int. Conf. on Computer Vision (ICCV),  pp. 2070-2078, 2017.

Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
'''
import torch
import torch.nn as nn
from torch.autograd import Function

class MPN(nn.Module):
     """Matrix power normalized Covariance pooling (MPNCOV)
        implementation of fast MPN-COV (i.e.,iSQRT-COV)
        https://arxiv.org/abs/1712.01034

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
     def __init__(self, iterNum=5, input_dim=256):

        super(MPN, self).__init__()
        self.iterNum=iterNum
        self.input_dim = input_dim
        self.output_dim = int(self.input_dim[0] * (self.input_dim[1] + 1) / 2)

     def _sqrtm(self, x):
         return Sqrtm.apply(x, self.iterNum)
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
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device).type(dtype)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1).type(dtype)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            YZY = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
               Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
               Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            YZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
         y = YZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, YZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         grad_output = grad_output.type(x.dtype)
         A = A.type(x.dtype)
         ZY = ZY.type(x.dtype)
         normA = normA.type(x.dtype)
         Y = Y.type(x.dtype)
         X = Z.type(x.dtype)
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = grad_output.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                          Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) -
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) -
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) -
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         der_NSiter = der_NSiter.transpose(1, 2)
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag().type(dtype)
         return grad_input, None

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

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

def TriuvecLayer(var):
    return Triuvec.apply(var)


def cov_forward(x):
    batchsize = x.data.shape[0]
    dim = x.data.shape[1]
    h = x.data.shape[2]
    w = x.data.shape[3]
    M = h*w
    x = x.reshape(batchsize,dim,M)
    I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M/M)*torch.eye(M,M,device = x.device)
    I_hat = I_hat.view(1,M,M).repeat(batchsize,1,1).type(x.dtype)
    y = x.bmm(I_hat).bmm(x.transpose(1,2))
    return y


def sqrt_forward(x,numIters=5):
    batchSize = x.data.shape[0]
    dim = x.data.shape[1]
    dtype = x.dtype
    I = torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    normA = x.mul(I).sum(dim=1).sum(dim=1)
    Y = x.div(normA.view(batchSize, 1, 1).expand_as(x))
    Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
      T = 0.5*(3.0*I - Z.bmm(Y))
      Y = Y.bmm(T)
      Z = T.bmm(Z)
    y = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
    return y


def triu_forward(x):
    batchSize = x.data.shape[0]
    dim = x.data.shape[1]
    dtype = x.dtype
    x = x.reshape(batchSize, dim*dim)
    I = torch.ones(dim,dim).triu().t().reshape(dim*dim)
    index = I.nonzero(as_tuple=False)
    y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device).type(dtype)
    y = x[:,index]
    return y
