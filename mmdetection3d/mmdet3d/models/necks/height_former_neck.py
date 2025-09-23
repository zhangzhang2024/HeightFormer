
from .HeightTransformerBlock import HeightTransformerBlock
from einops import rearrange
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import os
from turtle import forward
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer, trunc_normal_init, build_conv_layer
from torch import nn
import torch.utils.checkpoint as cp
import torch
from mmdet.models import NECKS
from mmcv.runner import auto_fp16
from mmcv.cnn import xavier_init, constant_init
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

class ResModule2D(nn.Module):
    def __init__(self, n_channels, norm_cfg=dict(type='BN2d'), groups=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.activation = nn.ReLU(inplace=True)

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x

class ResModule3D(nn.Module):
    def __init__(self, n_channels, norm_cfg=dict(type='BN3d'), groups=1):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, 3, 1, 1),
            nn.BatchNorm3d(n_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, 3, 1, 1),
            nn.BatchNorm3d(n_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x

@NECKS.register_module()
class Height3DConvNeck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bev_decoder_num=2,
                 height_atten_layer_num=6,
                 norm_cfg=dict(type='BN2d'),
                 is_transpose=True):
        super().__init__()

        self.is_transpose = is_transpose
        model = nn.ModuleList()
        model.append(ResModule2D(in_channels, norm_cfg))
        model.append(ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))    
        for i in range(bev_decoder_num):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.model = nn.Sequential(*model)
    
        self.height_atten_layer_num = height_atten_layer_num
        self.height_atten = nn.ModuleList()
        for i in range(height_atten_layer_num):
            self.height_atten.append(
                ResModule3D(64),
                )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.Conv3d):
                m.reset_parameters()
                         
    @auto_fp16()
    def forward(self, x):

        def _inner_forward(x):
            out = self.model.forward(x)
            return out

        x = x.permute(0,1,4,2,3).contiguous()
        B, C, D, H, W = x.shape
        
        for i in range(self.height_atten_layer_num):
            # b,d,h,w,c
            x = self.height_atten[i](x)

        x = x.reshape(B,-1,H,W)
        x = _inner_forward(x)

        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            return x.transpose(-1, -2)
        else:
            return x

@NECKS.register_module()
class BEVAttenNeck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bev_decoder_num=2,
                 bev_atten_layer_num=4,
                 norm_cfg=dict(type='BN2d'),
                 is_transpose=True):
        super().__init__()

        self.is_transpose = is_transpose
        model = nn.ModuleList()
        model.append(ResModule2D(in_channels, norm_cfg))
        model.append(ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))    
        for i in range(bev_decoder_num):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        
        self.model = nn.Sequential(*model)

        self.bev_atten_layer_num = bev_atten_layer_num
        self.window_size = (1, 7, 7)
        self.bev_atten = nn.ModuleList()
        for i in range(bev_atten_layer_num):
            self.bev_atten.append(
                HeightTransformerBlock(
                    dim=64,
                    num_heads=1,
                    window_size=self.window_size,
                    shift_size=(0,0,0),
                    mlp_ratio=2,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.2,
                    attn_drop=0,
                    drop_path=0,
                    norm_layer=nn.LayerNorm,
                    use_checkpoint=False,
                ))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                m.reset_parameters()
                     
    @auto_fp16()
    def forward(self, x):

        def _inner_forward(x):
            out = self.model.forward(x)
            return out

        x = x.permute(0,1,4,2,3).contiguous()
        B, C, D, H, W = x.shape
        # b,n,c
        x = x.reshape(B,C,-1).permute(0,2,1).contiguous()
        x = x.reshape(B,D,H,W,C)
        
        for i in range(self.bev_atten_layer_num):
            # b,d,h,w,c
            x = self.bev_atten[i](x, None)
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = x.reshape(B,-1,H,W)
        x = _inner_forward(x)

        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            return x.transpose(-1, -2)
        else:
            return x

@NECKS.register_module()
class HeightAttenNeck(nn.Module):
    ''' HeightFormer Neck module.'''
    def __init__(self,
                 in_channels,
                 out_channels,
                 bev_decoder_num=2,
                 height_atten_layer_num=6,
                 height_size = (4,1,1),
                 norm_cfg=dict(type='BN2d'),
                 is_transpose=True):
        super().__init__()

        self.is_transpose = is_transpose
        model = nn.ModuleList()
        model.append(ResModule2D(in_channels, norm_cfg))
        model.append(ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))    
        for i in range(bev_decoder_num):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.model = nn.Sequential(*model)
    
        self.height_atten_layer_num = height_atten_layer_num
        self.height_size = height_size
        self.height_atten = nn.ModuleList()
        for i in range(height_atten_layer_num):
            self.height_atten.append(
                HeightTransformerBlock(
                    dim=in_channels//height_size[0],
                    num_heads=1,
                    window_size=self.height_size,
                    shift_size=(0,0,0), 
                    mlp_ratio=2,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.2,
                    attn_drop=0,
                    drop_path=0,
                    norm_layer=nn.LayerNorm,
                    use_checkpoint=False,
                ))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                m.reset_parameters()
                         
    @auto_fp16()
    def forward(self, x):

        def _inner_forward(x):
            out = self.model.forward(x)
            return out

        x = x.permute(0,1,4,2,3).contiguous()
        B, C, D, H, W = x.shape
        # b,n,c
        x = x.reshape(B,C,-1).permute(0,2,1).contiguous()
        x = x.reshape(B,D,H,W,C)
        
        for i in range(self.height_atten_layer_num):
            # b,d,h,w,c
            x = self.height_atten[i](x, None) + x
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = x.reshape(B,-1,H,W)
        x = _inner_forward(x)

        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            return x.transpose(-1, -2)
        else:
            return x
