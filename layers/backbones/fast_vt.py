# Copyright (c) Megvii Inc. All rights reserved.
import numpy as np

import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmseg.ops import resize
from ..feature_visualization import draw_feature_map
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn

from ops.voxel_pooling import voxel_pooling

__all__ = ['LSSFPN']



class FastViewTransformer(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, neck_fuse, height_net_conf):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            height_net_conf (dict): Config for height net.
        """

        super(FastViewTransformer, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound

        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        frustum = torch.stack(torch.meshgrid([
                    torch.arange(self.x_bound[0],self.x_bound[1],self.x_bound[2],dtype=torch.float32),
                    torch.arange(self.y_bound[0],self.y_bound[1],self.y_bound[2],dtype=torch.float32),
                    torch.arange(self.z_bound[0],self.z_bound[1],self.z_bound[2],dtype=torch.float32)
                    ])).view(3,-1)
        frustum = torch.cat((frustum,torch.ones_like(frustum[:1,:])))

        return frustum
    
    def get_geometry(self, sensor2ego_mat, sensor2virtual_mat, intrin_mat, extrin_mat, ida_mat, reference_heights, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        
        # undo post-transformation
        # B x N x D x H x W x 3
        
        points = self.frustum
        intrinsic = intrin_mat[..., :3, :3]
        intrinsic[..., :2, : ] = intrinsic[..., :2, : ] / self.downsample_factor
        extrinsic = extrin_mat[..., :3, : ]
        proj =  intrinsic @ extrinsic
        points = bda_mat @ points
        pixels = proj @ points.unsqueeze(1)
        pixels_u = (pixels[...,0, :] / pixels[...,2, :]).squeeze(1)
        pixels_v = (pixels[...,1, :] / pixels[...,2, :]).squeeze(1)
        
        return pixels_u, pixels_v

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        mlvl_feats = self.img_neck(self.img_backbone(imgs))
        
        multi_scale_id=[0]
        
        for msid in multi_scale_id:
            if getattr(self, f'neck_fuse_{msid}', None) is not None:
                fuse_feats = [mlvl_feats[msid]]
                for i in range(msid + 1, len(mlvl_feats)):
                    resized_feat = resize(
                        mlvl_feats[i], 
                        size=mlvl_feats[msid].size()[2:], 
                        mode="bilinear", 
                        align_corners=False)
                    fuse_feats.append(resized_feat)
            
                if len(fuse_feats) > 1:
                    fuse_feats = torch.cat(fuse_feats, dim=1)
                else:
                    fuse_feats = fuse_feats[0]
                fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
        img_feats = fuse_feats
        
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              is_return_height=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_height (bool, optional): Whether to return height.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        
        pixels_u, pixels_v = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['sensor2virtual_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['extrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict['reference_heights'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        
        
        B, _, _, C, H, W = img_feats.shape
        pixels_u = (pixels_u/W *2 -1)
        pixels_v = (pixels_v/H *2 -1)

        grids = torch.stack([pixels_u, pixels_v], dim=2).to(img_feats.device).unsqueeze(1)
        bev_feats = F.grid_sample(
                img_feats.squeeze(1).squeeze(1),
                grids,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True) 
        loc = torch.where(bev_feats != bev_feats)
        bev_feats[loc] = 0
        bev_feats = bev_feats.view(B, C, self.voxel_num[0], self.voxel_num[1], self.voxel_num[2])
        
        return bev_feats.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                is_return_height=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            is_return_height=is_return_height)

        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_height else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    is_return_height=False)
                ret_feature_list.append(feature_map)

        if is_return_height:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)
