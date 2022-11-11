# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmpose.core import imshow_keypoints, imshow_multiview_keypoints_3d
from mmpose.core.camera import SimpleCamera, SimpleCameraTorch
from mmpose.core.post_processing.post_transforms import (
    affine_transform_torch, get_affine_transform)
from .. import builder
from ..builder import POSENETS
from ..utils.misc import torch_meshgrid_ij
from .base import BasePose

@POSENETS.register_module()
class DQTrasformerPoseDetector(BasePose):
    # TODO not init
    """Detect human center by 3D CNN on voxels.

    Please refer to the
    `paper <https://arxiv.org/abs/2004.06239>` for details.
    Args:
        image_size (list): input size of the 2D model.
        heatmap_size (list): output size of the 2D model.
        space_size (list): Size of the 3D space.
        cube_size (list): Size of the input volume to the 3D CNN.
        space_center (list): Coordinate of the center of the 3D space.
        center_net (ConfigDict): Dictionary to construct the center net.
        center_head (ConfigDict): Dictionary to construct the center head.
        train_cfg (ConfigDict): Config for training. Default: None.
        test_cfg (ConfigDict): Config for testing. Default: None.
    """
    # TODO not init
    def __init__(
        self,
        image_size,
        heatmap_size,
        space_size,
        cube_size,
        space_center,
        center_net,
        center_head,
        train_cfg=None,
        test_cfg=None,
    ):
        super(DQTrasformerPoseDetector, self).__init__()
        self.project_layer = ProjectLayer(image_size, heatmap_size)
        self.center_net = builder.build_backbone(center_net)
        self.center_head = builder.build_head(center_head)

        self.space_size = space_size
        self.cube_size = cube_size
        self.space_center = space_center

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    # TODO not init
    def assign2gt(self, center_candidates, gt_centers, gt_num_persons):
        """"Assign gt id to each valid human center candidate."""
        det_centers = center_candidates[..., :3]
        batch_size = center_candidates.shape[0]
        cand_num = center_candidates.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)

        for i in range(batch_size):
            cand = det_centers[i].view(cand_num, 1, -1)
            gt = gt_centers[None, i, :gt_num_persons[i]]

            dist = torch.sqrt(torch.sum((cand - gt)**2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt
            cand2gt[i][min_dist > self.train_cfg['dist_threshold']] = -1.0

        center_candidates[:, :, 3] = cand2gt

        return center_candidates
    # TODO not init
    def forward(self,
                img,
                img_metas,
                return_loss=True,
                feature_maps=None,
                targets_3d=None):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps width: W
            heatmaps height: H
        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            return_loss: Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            feature_maps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps.
        Returns:
            dict: if 'return_loss' is true, then return losses.
                Otherwise, return predicted poses
        """
        if return_loss:
            return self.forward_train(img, img_metas, feature_maps, targets_3d)
        else:
            return self.forward_test(img, img_metas, feature_maps)
    # TODO not init
    def forward_train(self,
                      img,
                      img_metas,
                      feature_maps=None,
                      targets_3d=None,
                      return_preds=False):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps width: W
            heatmaps height: H
        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            feature_maps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps.
            return_preds (bool): Whether to return prediction results
        Returns:
            dict: if 'return_pred' is true, then return losses
                and human centers. Otherwise, return losses only
        """
        initial_cubes, _ = self.project_layer(feature_maps, img_metas,
                                              self.space_size,
                                              [self.space_center],
                                              self.cube_size)
        center_heatmaps_3d = self.center_net(initial_cubes)
        center_heatmaps_3d = center_heatmaps_3d.squeeze(1)
        center_candidates = self.center_head(center_heatmaps_3d)

        device = center_candidates.device

        gt_centers = torch.stack([
            torch.tensor(img_meta['roots_3d'], device=device)
            for img_meta in img_metas
        ])
        gt_num_persons = torch.stack([
            torch.tensor(img_meta['num_persons'], device=device)
            for img_meta in img_metas
        ])
        center_candidates = self.assign2gt(center_candidates, gt_centers,
                                           gt_num_persons)

        losses = dict()
        losses.update(
            self.center_head.get_loss(center_heatmaps_3d, targets_3d))

        if return_preds:
            return center_candidates, losses
        else:
            return losses
    # TODO not init
    def forward_test(self, img, img_metas, feature_maps=None):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps width: W
            heatmaps height: H
        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            feature_maps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps.
        Returns:
            human centers
        """
        initial_cubes, _ = self.project_layer(feature_maps, img_metas,
                                              self.space_size,
                                              [self.space_center],
                                              self.cube_size)
        center_heatmaps_3d = self.center_net(initial_cubes)
        center_heatmaps_3d = center_heatmaps_3d.squeeze(1)
        center_candidates = self.center_head(center_heatmaps_3d)
        center_candidates[..., 3] = \
            (center_candidates[..., 4] >
             self.test_cfg['center_threshold']).float() - 1.0

        return center_candidates
    # TODO not init
    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError
    # TODO not init
    def forward_dummy(self, feature_maps):
        """Used for computing network FLOPs."""
        batch_size, num_channels, _, _ = feature_maps[0].shape
        initial_cubes = feature_maps[0].new_zeros(batch_size, num_channels,
                                                  *self.cube_size)
        _ = self.center_net(initial_cubes)
