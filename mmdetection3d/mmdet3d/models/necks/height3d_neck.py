import os
import torch
import math
from torch import nn
from torch.nn import BatchNorm2d
from torch.nn import functional as F

from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models import NECKS
from mmcv.runner import auto_fp16
from mmcv.cnn import  constant_init
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.cnn import build_conv_layer
from mmcv.cnn import (ConvModule, constant_init, is_norm, normal_init)

def fill_up_weights(up):
    """Simulated bilinear upsampling kernel.

    Args:
        up (nn.Module): ConvTranspose2d module.
    """
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

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

class Bottleneck(nn.Module):
    """Bottleneck block for DilatedEncoder used in `YOLOF.

    <https://arxiv.org/abs/2103.09460>`.

    The Bottleneck contains three ConvLayers and one residual connection.

    Args:
        in_channels (int): The number of input channels.
        mid_channels (int): The number of middle output channels.
        dilation (int): Dilation rate.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 dilation,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvModule(
            in_channels, mid_channels, 1, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            mid_channels,
            mid_channels,
            3,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg)
        self.conv3 = ConvModule(
            mid_channels, in_channels, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm, groups=1):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DilatedEncoder(nn.Module):
    """Dilated Encoder for YOLOF <https://arxiv.org/abs/2103.09460>`.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
              which are 1x1 conv + 3x3 conv
        - the dilated residual block

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        block_mid_channels (int): The number of middle block output channels
        num_residual_blocks (int): The number of residual blocks.
    """

    def __init__(self, 
                 in_channels, 
                 norm_cfg, 
                 num_residual_blocks):
        super(DilatedEncoder, self).__init__()
        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = [2, 4, 6, 8]
        self._init_layers()
        self._init_weights

    def _init_layers(self):
        
        BatchNorm=nn.BatchNorm2d
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                    _ASPPModule(
                        self.in_channels,
                        self.in_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        BatchNorm=BatchNorm))
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _init_weights(self):
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)


    def forward(self, feature):
        # out = self.lateral_norm(self.lateral_conv(feature))
        # out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(feature)

class GHPEncoder(nn.Module):

    def __init__(self, 
                 in_channels, 
                 norm_cfg,
                 num_residual_blocks):
        super(GHPEncoder, self).__init__()
        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = [1, 4, 6, 8]
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        
        BatchNorm=nn.BatchNorm2d
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            if dilation == 1:
                encoder_blocks.append(
                    _ASPPModule(
                        self.in_channels,
                        self.in_channels,
                        7,
                        padding=3,
                        dilation=dilation,
                        BatchNorm=BatchNorm,
                        groups=self.in_channels,))
            else: 
                encoder_blocks.append(
                    _ASPPModule(
                        self.in_channels,
                        self.in_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        BatchNorm=BatchNorm))
                
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _init_weights(self):
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

    def forward(self, feature):
        return self.dilated_encoder_blocks(feature)

class SACLayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Identity):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels//4, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels//4, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

@NECKS.register_module()
class GHPDCNSACNeck(nn.Module):
    """Neck for Height3D.
    """

    def __init__(self,
                 in_channels=192,
                 mid_channels=128,
                 out_channels=128,
                 num_layers=6,
                 height_grid_num=4,
                 norm_cfg=dict(type='BN2d'),
                 is_train_height=False,
                 ):
        super().__init__()

        self.is_train_height = is_train_height
        
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
        for i in range(num_layers):
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
        self.height_conv = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            GHPEncoder(mid_channels, norm_cfg=norm_cfg, num_residual_blocks=4),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels, height_grid_num, kernel_size=1, stride=1, padding=0)
            )
        self.bev_sac = SACLayer(in_channels)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                fill_up_weights(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()
   
    @auto_fp16()
    def forward(self, x):
       
        def _inner_forward(x):
            out = self.model.forward(x)
            return out

        if bool(os.getenv("DEPLOY", False)):
            N, X, Y, Z, C = x.shape
            x = x.reshape(N, X, Y, Z*C).permute(0, 3, 1, 2)
        else:

            B, C, X, Y, Z = x.shape
            # B,X,Y,Z,C 
            x = x.permute(0, 2, 3, 4, 1)
            # -> B,X,Y,Z*C -> B,Z*C,X,Y
            bev_init = x.reshape(B,X,Y,Z*C).permute(0,3,1,2).contiguous()

            # B,Z,X,Y
            height_feat = self.height_conv(bev_init)
            height_softmax = height_feat.softmax(dim=1)

            # B,X,Y,Z,1
            height_distribution = height_softmax.permute(0,2,3,1).unsqueeze(-1).contiguous()
            bev_feat = (x*height_distribution).reshape(B,X,Y,Z*C).permute(0,3,1,2).contiguous()
            
            bev_refine = self.bev_sac(bev_init, bev_feat)
           
            bev_out = _inner_forward(bev_refine)

            if self.is_train_height == True:
                return bev_out.transpose(-1, -2).contiguous(), height_distribution
            
            else:
                return bev_out.transpose(-1, -2).contiguous()

