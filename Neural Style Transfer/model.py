import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.torch_utils import *

class Content_Encoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=4, norm='in', activation='relu'):
        super(Content_Encoder, self).__init__()
        layers = []
        layers += [ConvBlock(3, conv_dim, 7, 1, 3, norm=norm, activation=activation)] # H,W,3 => H,W,64

        # Down-sampling layers
        curr_dim = conv_dim
        for i in range(2):
            layers += [ConvBlock(curr_dim, curr_dim*2, 4, 2, 1, norm=norm, activation=activation)] # H,W,64 => H/2,W/2,128 => H/4,W/4,256
            curr_dim = curr_dim * 2

        # Bottleneck layers
        for i in range(repeat_num):
            layers += [ResidualBlock(dim=curr_dim, norm=norm, activation=activation)]

        self.main = nn.Sequential(*layers)
        self.curr_dim = curr_dim

    def forward(self, x):
        return self.main(x)


class Style_Encoder(nn.Module):
    def __init__(self, conv_dim=64, n_group=32, norm='ln', activation='relu'):
        super(Style_Encoder, self).__init__()
        curr_dim = conv_dim
        layers = []
        layers += [ConvBlock(3, conv_dim, 7, 1, 3, norm='none', n_group=n_group, activation=activation)] # H,W,3 => H,W,64

        # Down-sampling layers (dim*2)
        curr_dim = conv_dim
        for i in range(2):
            layers += [ConvBlock(curr_dim, curr_dim*2, 4, 2, 1, norm=norm, n_group=n_group, activation=activation)] # H,W,64 => H/2,W/2,128 => H/4,W/4,256
            curr_dim = curr_dim * 2

        # Down-sampling layers (keep dim)
        for i in range(2): # original: 2
            layers += [ConvBlock(curr_dim, curr_dim, 4, 2, 1, norm=norm, n_group=n_group, activation=activation)] # H/4,W/4,256, H/8,W/8,256, H/16,W/16,256

        layers += [nn.AdaptiveAvgPool2d(1)] # H/16,W/16,256 => 1,1,256

        self.main = nn.Sequential(*layers)
        self.curr_dim = curr_dim

    def forward(self, x):
        return self.main(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, num_block=1, norm='none', n_group=32, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        curr_dim = dim
        layers += [LinearBlock(input_dim, curr_dim, norm=norm, n_group=n_group, activation=activation)]

        for _ in range(num_block):
            layers += [LinearBlock(curr_dim, curr_dim, norm=norm, n_group=n_group, activation=activation)]
            
        layers += [LinearBlock(curr_dim, output_dim, norm='none', activation='none')] # no output activations
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x.view(x.size(0), -1))

class WCT(nn.Module):
    def __init__(self, n_group, device, input_dim, mlp_dim, bias_dim, mask, w_alpha=0.4):
        super(WCT, self).__init__()
        self.G = n_group
        self.device = device
        self.alpha = nn.Parameter(torch.ones(1)-w_alpha)
        self.mask = mask

        self.sqrt_root =  MPA_Lya.apply
        self.sqrt_root_inv = MPA_Lya_Inv.apply

    def forward(self, c_A, s_B):
        return self.wct(c_A, s_B)

    def wct(self, c_A, s_B):
        B,C,H,W = c_A.size()
        n_mem = C // self.G # 32 if G==8
        eps = 1e-5

        #Whitening Transform
        c_A_ = c_A.view(self.G*B, n_mem, -1)  # B,C,H,W => C,B,H,W => GB,C//G,HW
        c_A_mean = torch.mean(c_A_, dim=2, keepdim=True)
        c_A_ = c_A_ - c_A_mean  # GB,C//G,HW
        cov_c = torch.bmm(c_A_, c_A_.transpose(1,2)).div(H * W - 1) + eps*torch.eye(n_mem).unsqueeze(0).to(self.device) # GB,C//G,C//G
        cov_c_inv_sqrt=self.sqrt_root_inv(cov_c)
        whitened = cov_c_inv_sqrt.bmm(c_A_)


        #Coloring Transform
        s_B_ = s_B.view(self.G*B, n_mem, -1)  # B,C,H,W => C,B,H,W => GB,C//G,HW
        s_B_mean = torch.mean(s_B_, dim=2, keepdim=True)
        s_B_ = s_B_ - s_B_mean  # GB,C//G,HW
        cov_b = torch.bmm(s_B_, s_B_.transpose(1,2)).div(H * W - 1) + eps * torch.eye(n_mem).unsqueeze(0).to(self.device)  # GB,C//G,C//G
        cov_b_sqrt=self.sqrt_root(cov_b)
        colored_B = cov_b_sqrt.bmm(whitened).view(B, C, H, W)


        return self.alpha * (colored_B + s_B_mean.view(B,C,1,1)) + (1 - self.alpha) * c_A


class Decoder(nn.Module):
    def __init__(self, input_dim, mask, n_group, bias_dim, mlp_dim, repeat_num=4, 
                norm='ln', device=None):
        super(Decoder, self).__init__()
        
        curr_dim = input_dim
        
        # Bottleneck layers
        self.resblocks = nn.ModuleList([ResidualBlock(dim=curr_dim, norm='none', n_group=n_group) for i in range(repeat_num)])
        self.gdwct_modules = nn.ModuleList([WCT(n_group, device, input_dim, mlp_dim, bias_dim, mask) for i in range(repeat_num+1)])
        
        # Up-sampling layers
        layers = []
        for i in range(2):
            layers += [Upsample(scale_factor=2, mode='nearest')]
            layers += [ConvBlock(curr_dim, curr_dim//2, 5, 1, 2, norm=norm, n_group=n_group)]
            curr_dim = curr_dim // 2

        layers += [ConvBlock(curr_dim, 3, 7, 1, 3, norm='none', activation='tanh')]

        self.main = nn.Sequential(*layers)

    def forward(self, c_A, s_B):

        # Multi-hops
        for i, resblock in enumerate(self.resblocks):
            if i == 0:
                c_A = self.gdwct_modules[i](c_A, s_B)

            c_A = resblock(c_A)
            c_A = self.gdwct_modules[i+1](c_A, s_B)

        return self.main(c_A)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=8, mask=None, n_group=16, 
                mlp_dim=256, bias_dim=512, content_dim=256, device=None):
        super(Generator, self).__init__()
        self.c_encoder = Content_Encoder(conv_dim, repeat_num//2, norm='in',activation='relu')
        self.s_encoder = Style_Encoder(conv_dim, n_group, norm= 'gn', activation='relu')
        self.decoder = Decoder(content_dim, mask, n_group, bias_dim, mlp_dim, repeat_num//2, norm='ln', device=device)

    def forward(self, c_A, s_B_):
        return self.decoder(c_A, s_B_)


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim, norm='in', n_group=32, activation='relu', use_affine=True):
        super(ResidualBlock, self).__init__()
        layers = []
        layers += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, n_group=n_group, activation=activation, use_affine=use_affine)]
        layers += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, n_group=n_group, activation='none', use_affine=use_affine)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.main(x)


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, k, s, p, dilation=False, norm='in', n_group=32, 
                        activation='relu', pad_type='mirror', use_affine=True, use_bias=True):
        super(ConvBlock, self).__init__()

        # Init Normalization
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(output_dim, affine=use_affine, track_running_stats=True)
        elif norm == 'ln':
            # LayerNorm(output_dim, affine=use_affine)
            self.norm = nn.GroupNorm(1, output_dim)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(n_group, output_dim)
        elif norm == 'none':
            self.norm = None

        # Init Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU(num_parameters=1, init=0.25)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

        # Init pad-type
        if pad_type == 'mirror':
            self.pad = nn.ReflectionPad2d(p)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(p)

        # initialize convolution
        if dilation:
            self.conv = nn.Conv2d(input_dim, output_dim, k, s, dilation=p, bias=use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, k, s, bias=use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='ln', n_group=32, activation='relu', use_affine=True):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # Init Normalization
        if norm == 'ln':
            # self.norm = LayerNorm(output_dim, affine=use_affine)
            self.norm = nn.GroupNorm(1, output_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(n_group, output_dim)

        elif norm == 'none':
            self.norm = None

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU(num_parameters=1, init=0.25)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class Discriminator(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(Discriminator, self).__init__()
        self.n_layer = params['N_LAYER']
        self.gan_type = params['GAN_TYPE']
        self.dim = params['FIRST_DIM']
        self.norm = params['NORM']
        self.activ = params['ACTIVATION']
        self.num_scales = params['NUM_SCALES']
        self.pad_type = params['PAD_TYPE']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [ConvBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [ConvBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss