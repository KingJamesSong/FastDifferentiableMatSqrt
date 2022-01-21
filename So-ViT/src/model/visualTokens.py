import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from timm.models.densenet import DenseBlock
from timm.models.densenet import DenseTransition

class Bottleneck(nn.Module):
    """
    Bottleneck block from ResNet
    """
    def __init__(self, inplanes, planes, stride=1, downsample=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1),
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if downsample is not False:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class InceptionBlock(nn.Module):
    """

    """
    def __init__(self, stride=1, in_channels=64):
        super(InceptionBlock, self).__init__()
        self.in_channels =  in_channels
        self.branch0 = BasicConv2d(self.in_channels, 32, kernel_size=1, stride=stride)
        self.branch1 = nn.Sequential(
            BasicConv2d(self.in_channels, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=stride, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(self.in_channels, 16, kernel_size=1, stride=1),
            BasicConv2d(16, 16, kernel_size=5, stride=stride, padding=2)
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            BasicConv2d(self.in_channels, 16, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels=64, inter_channels=None, out_channels=128, sub_sample=False, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """
        super(_NonLocalBlockND, self).__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.bn_layer = bn_layer

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d
        if bn_layer:
            self.bn_norm = bn(self.out_channels)


        self.g = nn.Sequential(
                conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),
                bn(self.inter_channels)
            )

        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.out_channels,
                            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Sequential(
                conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),
                bn(self.inter_channels)
            )
        self.phi = nn.Sequential(
                conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0),
                bn(self.inter_channels)
            )
        self.shortcut = None

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            self.theta = nn.Sequential(self.theta, max_pool_layer)
            self.shortcut = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.out_channels,
                            kernel_size=1, stride=1, padding=0), 
                            max_pool_layer,
                            bn(self.out_channels))
        if self.in_channels != self.out_channels and not sub_sample:
            self.shortcut = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.out_channels,
                            kernel_size=1, stride=1, padding=0),
                            bn(self.out_channels))

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        if self.sub_sample:
            y = y.view(batch_size, self.inter_channels, x.shape[2]//2, x.shape[3]//2)
        else:
            y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        if self.shortcut is not None:
            x = self.shortcut(x)
        z = W_y + x
        if self.bn_layer:
            z = self.bn_norm(z)
        return z
class baseConvModule(nn.Module):
    """
    Embedding of visual tokens
    """
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, token_dim=64):
        super(baseConvModule, self).__init__()
        stride = img_size // 112
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, token_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        self.stage = nn.Sequential(
            Bottleneck(token_dim, 64, stride=2, downsample=True),
            Bottleneck(256, 64, stride=2, downsample=True),
            Bottleneck(256, 64, stride=stride, downsample=(stride==2)),
        )
        self.transition = nn.Conv2d(256, embed_dim, kernel_size=(1,1), stride=(1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        x = self.transition(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(1, 2)
        return x

class ResNetStyleModule(baseConvModule):
    """
    ResNet style
    """
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, token_dim=64):
        super(ResNetStyleModule, self).__init__(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim,
        )

class InceptionStyleModule(baseConvModule):
    """
    Inception style
    """
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, token_dim=64):
        super(InceptionStyleModule, self).__init__(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim
        )
        self.stage = nn.Sequential(
            InceptionBlock(in_channels=64, stride=2),
            InceptionBlock(in_channels=128, stride=2),
            InceptionBlock(in_channels=128, stride=1),
        )
        self.transition = nn.Conv2d(128, embed_dim, kernel_size=(1,1), stride=(1, 1))

class DenseNetStyleModule(baseConvModule):
    """
    DenseNet style
    """
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, token_dim=64):
        super(DenseNetStyleModule, self).__init__(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim
        )
        self.stage = nn.Sequential(
            DenseBlock(num_layers=6, num_input_features=64, bn_size=2, growth_rate=16),
            DenseTransition(num_input_features=160, num_output_features=128),
            DenseBlock(num_layers=6, num_input_features=128, bn_size=2, growth_rate=16),
            DenseTransition(num_input_features=224, num_output_features=128),
            DenseBlock(num_layers=6, num_input_features=128, bn_size=2, growth_rate=16),
            DenseTransition(num_input_features=224, num_output_features=128),
        )
        self.transition = nn.Conv2d(128, embed_dim, kernel_size=(1,1), stride=(1, 1))

class NonlocalStyleModule(baseConvModule):
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, token_dim=64):
        super(NonlocalStyleModule, self).__init__(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim
        )
        self.stage = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            _NonLocalBlockND(in_channels=64, inter_channels=64, out_channels=128, sub_sample=False, bn_layer=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            _NonLocalBlockND(in_channels=64, inter_channels=64, out_channels=256, sub_sample=False, bn_layer=True),
        )
        self.transition = nn.Conv2d(256, embed_dim, kernel_size=(1,1), stride=(1, 1))

class visualTokens(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=384,  visualTokenConfig={}):
        super(visualTokens, self).__init__()
        self.tokens_type = visualTokenConfig['type']
        self.token_dim = visualTokenConfig['token_dim']
        stride = img_size // 112
        if self.tokens_type == 'ResNet':
            self.tokens = baseConvModule(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, token_dim=self.token_dim)
        elif self.tokens_type == 'Inception':
            self.tokens = InceptionStyleModule(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, token_dim=self.token_dim)
        elif self.tokens_type == 'DenseNet':
            self.tokens = DenseNetStyleModule(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, token_dim=self.token_dim)
        elif self.tokens_type == 'Non-local':
            self.tokens = NonlocalStyleModule(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, token_dim=self.token_dim)
        
        self.num_patches = (img_size // (4 * 2 * stride)) * (img_size // (4 * 2 * stride))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        x = self.tokens(x)
        
        return x
