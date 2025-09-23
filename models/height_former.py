import imp
from torch import nn
import torch
from layers.backbones.fast_vt import FastViewTransformer
from layers.heads.height_3d_head import Height3DHead 
from torch.nn import functional as F

__all__ = ['HeightFormer']


class HeightFormer(nn.Module):
    """
    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_height (bool): Whether to return height.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf=None, head_conf=None, is_train_height=False):
        super(HeightFormer, self).__init__()
        self.backbone = FastViewTransformer(**backbone_conf)
        self.head = Height3DHead(**head_conf)
        self.is_train_height = is_train_height

    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVHeight

        Args:
            x (Tensor): Input ferature map.
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
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_height and self.training:
            # x, height_pred = self.backbone(x,
            #                               mats_dict,
            #                               timestamps,
            #                               is_return_height=True)
            # preds = self.head(x)
            x = self.backbone(x, mats_dict, timestamps)
            
            preds, height_pred = self.head(x)
            
            return preds, height_pred
        else:
            x = self.backbone(x, mats_dict, timestamps)
            # import pdb;pdb.set_trace()
            preds = self.head(x)
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVHeight.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)
    
    def loss_height(self, gt_height, pred_height, Z, height_size):
        
        # import pdb;pdb.set_trace()
        # draw_feature_map(pred_height)
        # draw_feature_map(gt_height[0].unsqueeze(0))
        # draw_feature_map(pred_height.permute(0,3,2,1))
        # draw_feature_map(gt_height)
        
        pred_height = pred_height.squeeze(-1).permute(0,3,1,2).transpose(-1,-2).permute(0,2,3,1).reshape(-1,Z)
        gt_height = gt_height.squeeze(1).reshape(-1,1)

        fg_mask = torch.max(gt_height, dim=1).values > 0.0

        gt_height = gt_height[fg_mask]
        pred_height = pred_height[fg_mask]

        gt_height = (gt_height / height_size).floor().long()
        gt_height = torch.clamp(gt_height, 0, Z - 1).long()
        # if torch.isnan(pred_height).int().sum() != 0:
        #     import pdb;pdb.set_trace()
        gt_height = F.one_hot(gt_height, num_classes = Z).view(-1, Z).float()
        # import pdb;pdb.set_trace()
        loss_height = F.binary_cross_entropy(
                pred_height,
                gt_height,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        
        return loss_height * 2

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
