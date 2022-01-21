'''
Implementation of 'Temporal-attentive Covariance Pooling Networks for Action Recognition'
Authors: Zilin Gao, Qilong Wang, Bingbing Zhang, Qinghua Hu and Peihua Li.

This file is modified from non-local code.
'''

import pdb
import torch
import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill


class TSA(nn.Module):

    def __init__(
        self,
        dim,
        frame=8,
        instantiation="softmax",
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim (int): number of dimension for the input.
            frame (int): frame number.
            instantiation (string): supports two different instantiation method:
                "dot_product": normalizing correlation matrix with L2.
                "softmax": normalizing correlation matrix with Softmax.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm2d.
        """
        super(TSA, self).__init__()
        self.dim = dim
        self.dim = dim
        self.frame = frame
        self.instantiation = instantiation
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_TSA(
            norm_module
        )

    def _construct_TSA(
        self, norm_module
    ):
        # Three convolution : conv_phi0 (x), conv_phi_1 (x-1), and conv_phi_2 (x-2).
        self.conv_phi_1 = nn.Conv2d(
            self.dim, self.dim, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi_2 = nn.Conv2d(
            self.dim, self.dim, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi0 = nn.Conv2d(
            self.dim, self.dim, kernel_size=1, stride=1, padding=0
        )

        # TODO: change the name to `norm`
        self.norm = norm_module(
            self.dim,
            eps=self.norm_eps,
        )

        self.init_weights()


    def init_weights(self):
            """
            Performs ResNet style weight initialization.
            """
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    """
                    Follow the initialization method proposed in:
                    {He, Kaiming, et al.
                    "Delving deep into rectifiers: Surpassing human-level
                    performance on imagenet classification."
                    arXiv preprint arXiv:1502.01852 (2015)}
                    """
                    c2_msra_fill(m)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    batchnorm_weight = 0.
                    if m.weight is not None:
                        m.weight.data.fill_(batchnorm_weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=0.01) 
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        x_identity = x
        N, C, H, W = x.size()

        T = self.frame
        V = N // T
        x = x.view(V, T, C, H, W)
        x_1 = torch.cat((x[:,0,:,:,:].unsqueeze(1),x[:,:-1,:,:,:]), 1)
        x_2 = torch.cat((x_1[:, 0, :, :, :].unsqueeze(1), x_1[:, :-1, :, :, :]), 1)
        x_1 = x_1.view(N, C, H, W)
        x_2 = x_2.view(N, C, H, W)
        x = x.view(-1, C, H, W)

        phi_x_1 = self.conv_phi_1(x_1)
        phi_x_2 = self.conv_phi_2(x_2)
        phi_x0 = self.conv_phi0(x)

        phi_x_1 = phi_x_1.view(N, self.dim, -1)
        phi_x_2 = phi_x_2.view(N, self.dim, -1)
        phi_x0 = phi_x0.view(N, self.dim, -1)

        # (N, C, HxW) * (N, C, HxW) => (N, HxW, HxW).
        theta_x_1_2 = torch.einsum("nca,ncb->nab", (phi_x_1, phi_x_2))
        # For original Non-local paper, there are two main ways to normalize
        # the affinity tensor:
        #   1) Softmax normalization (norm on exp).
        #   2) dot_product normalization.
        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_x_1_2 before softmax.
            theta_x_1_2 = theta_x_1_2 * (self.dim ** -0.5)
            theta_x_1_2 = nn.functional.softmax(theta_x_1_2, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_x_1_2.shape[2]
            theta_x_1_2 = theta_x_1_2 / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )

        # (N, HxW, HxW) * (N, C, HxW) => (N, C, HxW).
        x_out = torch.einsum("ntg,ncg->nct", (theta_x_1_2, phi_x0))

        # (N, C, HxW) => (N, C, H, W).
        x_out = x_out.contiguous().view(N, self.dim, H, W)

        x_out = self.norm(x_out)
        return x_identity + x_out
