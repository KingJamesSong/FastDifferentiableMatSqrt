import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.senet import SEModule
from timm.models.inception_resnet_v2 import BasicConv2d
from timm.models.densenet import DenseBlock
from timm.models.densenet import DenseTransition

import numpy as np
from torch.nn import functional as F
from .model.classifier import Classifier
from .model.visualTokens import visualTokens, Bottleneck
from .model.vitBlock import ViTBlock, get_sinusoid_encoding


visualTokenConfig = dict(
    type='ResNet',
    token_dim = 64,
)

ViTConfig = dict(
    embed_dim=384, 
    depth=14, 
    num_heads=6, 
    mlp_ratio=3.,
    qkv_bias=False,
    qk_scale=None,
    attn_drop=0.,
    norm_layer=nn.LayerNorm,
    act_layer=nn.GELU,
)

representationConfig = dict(
    type='second-order',
    args=dict(
        cov_type='norm',
        remove_mean=True,
        dimension_reduction=[64, 64],
        input_dim=384,
    ),
    normalization=dict(
        type='PadeSqt',
        args=dict(
            #alpha=0.5,
            #iterNum=5,
            #svNum=1,
            #regular=nn.Dropout(0.5),
            #vec='full',
            input_dim=48,
        ),
    ),
)



class So_ViT(nn.Module):
    def __init__(self,
            img_size=224, 
            in_chans=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm, 
            visualTokenConfig=visualTokenConfig,
            ViTConfig=ViTConfig, 
            representationConfig=representationConfig):
        super(So_ViT, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = ViTConfig['embed_dim']
        self.depth = ViTConfig['depth']
        ViTConfig.pop('depth')
        #-------------
        # Build SO-ViT
        #-------------
        self.visual_tokens = visualTokens(img_size=img_size, in_chans=in_chans, embed_dim=self.embed_dim, visualTokenConfig=visualTokenConfig)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)] 
        self.blocks = nn.ModuleList([ViTBlock(drop=drop_rate, drop_path=dpr[i], **ViTConfig) for i in range(self.depth)])
        self.classifier = Classifier(num_classes=num_classes, input_dim=self.embed_dim, representationConfig=representationConfig)
        #-------------------------------------------
        # Prepare Class Token and Position Embedding
        #-------------------------------------------
        num_patches = self.visual_tokens.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=self.embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(self.embed_dim)
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        B = x.shape[0]
        x = self.visual_tokens(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x

@register_model
def So_vit_19(pretrained=False, **kwargs):  
    ViTConfig['embed_dim']=448
    ViTConfig['depth']=19
    ViTConfig['num_heads']=7
    ViTConfig['mlp_ratio']=3.
    representationConfig['args']['dimension_reduction']=[96, 48]
    representationConfig['args']['input_dim']=448
    representationConfig['normalization']['args']['input_dim']=representationConfig['args']['dimension_reduction']
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = So_ViT(visualTokenConfig=visualTokenConfig, ViTConfig=ViTConfig, representationConfig=representationConfig, **kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def So_vit_14(pretrained=False, **kwargs):
    representationConfig['normalization']['args']['input_dim']=representationConfig['args']['dimension_reduction']
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = So_ViT(visualTokenConfig=visualTokenConfig, ViTConfig=ViTConfig, representationConfig=representationConfig, **kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def So_vit_10(pretrained=False, **kwargs):  
    ViTConfig['embed_dim']=256
    ViTConfig['depth']=10
    ViTConfig['num_heads']=4
    ViTConfig['mlp_ratio']=2.
    representationConfig['args']['dimension_reduction']=[48, 48]
    representationConfig['args']['input_dim']=256
    #representationConfig['normalization']['args']['regular']=None
    representationConfig['normalization']['args']['input_dim']=representationConfig['args']['dimension_reduction']
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = So_ViT(visualTokenConfig=visualTokenConfig, ViTConfig=ViTConfig, representationConfig=representationConfig, **kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def So_vit_7(pretrained=False, **kwargs):  
    ViTConfig['embed_dim']=256
    ViTConfig['depth']=7
    ViTConfig['num_heads']=4
    ViTConfig['mlp_ratio']=2.
    representationConfig['args']['dimension_reduction']=[48, 48]
    representationConfig['args']['input_dim']=256
    #representationConfig['normalization']['args']['regular']=None
    representationConfig['normalization']['args']['input_dim']=representationConfig['args']['dimension_reduction']
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = So_ViT(visualTokenConfig=visualTokenConfig, ViTConfig=ViTConfig, representationConfig=representationConfig, **kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
