# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import pdb
import numpy as np
import math
import cv2
import time
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from typing import Optional, Callable
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from vmdepth.layers import BackprojectDepth, Project3D, SSIM


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
 
    if pretrained:

        if num_layers == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1

        elif num_layers == 34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1

        elif num_layers == 50:
            weights = models.ResNet50_Weights.IMAGENET1K_V1

        else:
            raise ValueError(f"Unsupported ResNet layers: {num_layers}")

        loaded = weights.get_state_dict(progress=True)

        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images

        model.load_state_dict(loaded)

    return model


class ResnetEncoderMatching(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, num_layers, pretrained, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=100.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear'):

        super(ResnetEncoderMatching, self).__init__()
        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        encoder = resnets[num_layers](pretrained)
        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height,
                                   width=self.matching_width)
        self.ssim = SSIM()

        self.compute_depth_bins(min_depth_bin, max_depth_bin)

        self.reduce = nn.Sequential(nn.Conv2d(2 * self.num_depth_bins,
                                              out_channels=self.num_depth_bins,
                                              kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True)
                                    )
        
        self.reduce_conv = nn.Sequential(nn.Conv2d(self.num_ch_enc[1] + self.num_depth_bins,
                                                   out_channels=self.num_ch_enc[1],
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)
                                         )
        self.vmamba_block = nn.Sequential(
                                            VSSBlock(hidden_dim=64, d_state=32),
                                            VSSBlock(hidden_dim=64, d_state=32),
                                        )

    def compute_depth_bins(self, min_depth_bin, max_depth_bin):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)"""

        if self.depth_binning == 'inverse':
            self.depth_bins = 1 / np.linspace(1 / max_depth_bin,
                                              1 / min_depth_bin,
                                              self.num_depth_bins)[::-1]  # maintain depth order

        elif self.depth_binning == 'linear':
            self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)

        elif self.depth_binning == 'sid':
            self.depth_bins = np.array(
                [np.exp(np.log(min_depth_bin) + np.log(max_depth_bin / min_depth_bin) * i / (self.num_depth_bins - 1))
                 for i in range(self.num_depth_bins)])
        else:
            raise NotImplementedError
        self.depth_bins = torch.from_numpy(self.depth_bins).float()

        self.warp_depths = []
        for depth in self.depth_bins:
            depth = torch.ones((1, self.matching_height, self.matching_width)) * depth
            self.warp_depths.append(depth)
        self.warp_depths = torch.stack(self.warp_depths, 0).float()
        if self.is_cuda:
            self.warp_depths = self.warp_depths.cuda()

 
    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK, resflow, mode="train"):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.

        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).

        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it"""

        batch_cost_volume = []  # store all cost volumes of the batch
        batch_cost_volume_flow = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence
        cost_volume_masks_flow = []  # store locations of '0's in cost volume for confidence

        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width)
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            cost_volume_flow = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts_flow = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)

            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]

            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            _resflow = resflow[batch_idx:batch_idx + 1]
            world_points = self.backprojector(self.warp_depths, _invK)

            # loop through ref images adding to the current cost volume
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]  # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx]

                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
                pix_locs = self.projector(world_points, _K, lookup_pose)
 
                pix_locs_flow = pix_locs + _resflow.permute(0, 2, 3, 1)

                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                       align_corners=True)

                warped1 = F.grid_sample(lookup_feat, pix_locs_flow, padding_mode='zeros', mode='bilinear',
                                        align_corners=True)

                # mask values landing outside the image (and near the border)
                # we want to ignore edge pixels of the lookup images and the current image
                # because of zero padding in ResNet
                # Masking of ref image border
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                        self.matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

                edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                edge_mask = edge_mask.float()

                # masking of current image
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask

                diffs = (0.4 * self.ssim(warped, current_feats[batch_idx:batch_idx + 1]).mean(1)
                         + 0.6 * torch.where(
                            torch.abs(warped - current_feats[batch_idx:batch_idx + 1]) < torch.abs(
                                warped1 - current_feats[batch_idx:batch_idx + 1]),
                            torch.abs(warped - current_feats[batch_idx:batch_idx + 1]),
                            torch.abs(warped1 - current_feats[batch_idx:batch_idx + 1])).mean(1)) * edge_mask
                diffs1 = (0.4 * self.ssim(warped1, current_feats[batch_idx:batch_idx + 1]).mean(1)
                          + 0.6 * torch.where(
                            torch.abs(warped - current_feats[batch_idx:batch_idx + 1]) < torch.abs(
                                warped1 - current_feats[batch_idx:batch_idx + 1]),
                            torch.abs(warped - current_feats[batch_idx:batch_idx + 1]),
                            torch.abs(warped1 - current_feats[batch_idx:batch_idx + 1])).mean(1)) * edge_mask

                # integrate into cost volume
                cost_volume = cost_volume + diffs
                cost_volume_flow = cost_volume_flow + diffs1
                counts = counts + (diffs > 0).float()
                counts_flow = counts_flow + (diffs1 > 0).float()

            # average over lookup images
            cost_volume = cost_volume / (counts + 1e-7)
            cost_volume_flow = cost_volume_flow / (counts_flow + 1e-7)

            # if some missing values for a pixel location (i.e. some depths landed outside) then
            # set to max of existing values
            missing_val_mask = (cost_volume == 0).float()
            missing_val_mask_flow = (cost_volume_flow == 0).float()

            if self.set_missing_to_max:
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                              cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)

            if self.set_missing_to_max:
                cost_volume_flow = cost_volume_flow * (1 - missing_val_mask_flow) + \
                                   cost_volume_flow.max(0)[0].unsqueeze(0) * missing_val_mask_flow
            batch_cost_volume_flow.append(cost_volume_flow)
            cost_volume_masks_flow.append(missing_val_mask_flow)

        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        batch_cost_volume_flow = torch.stack(batch_cost_volume_flow, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)
        cost_volume_masks_flow = torch.stack(cost_volume_masks_flow, 0)

        return [batch_cost_volume, batch_cost_volume_flow], [cost_volume_masks, cost_volume_masks_flow]

    def forward(self, current_image, lookup_images, poses, K, invK, resflow,
                min_depth_bin=None, max_depth_bin=None, flag=None
                ):

        # feature extraction
        self.features = self.feature_extraction(current_image, return_all_feats=True)
        current_feats = self.features[-1]

        # feature extraction on lookup images - disable gradients to save memory
        with torch.no_grad():
            if self.adaptive_bins:
                self.compute_depth_bins(min_depth_bin, max_depth_bin)

            batch_size, num_frames, chns, height, width = lookup_images.shape
            lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)
            lookup_feats = self.feature_extraction(lookup_images,
                                                   return_all_feats=False)
            _, chns, height, width = lookup_feats.shape
            lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)

            # warp features to find cost volume
            cost_volume, missing_mask = \
                self.match_features(current_feats, lookup_feats, poses, K, invK, resflow)

            confidence_mask = self.compute_confidence_mask(cost_volume[0].detach() * (1 - missing_mask[0].detach()))
            confidence_mask1 = self.compute_confidence_mask(cost_volume[1].detach() * (1 - missing_mask[1].detach()))
 

        add_mask = (confidence_mask.bool() + confidence_mask1.bool()).float()
        mask_both = (confidence_mask.bool() & confidence_mask1.bool()).float()
        one_mask = (~(mask_both.bool()) * add_mask).float()
        mask_both = mask_both.unsqueeze(1)
        one_mask = one_mask.unsqueeze(1)
        conf_mask = mask_both + one_mask

        mask_both_96 = mask_both.repeat(1, self.num_depth_bins, 1, 1)
        one_mask_96 = one_mask.repeat(1, self.num_depth_bins, 1, 1)

        v_cost_volume = cost_volume[0] + cost_volume[1]
        v_cost_volume = torch.where(cost_volume[0] < cost_volume[1], cost_volume[0],
                                    cost_volume[1]) + v_cost_volume * one_mask_96

        # for visualisation - ignore 0s in cost volume for minimum
        viz_cost_vol = v_cost_volume.clone().detach()
        viz_cost_vol[viz_cost_vol == 0] = 100
        mins, argmin = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argmin)

        v_cost_volume = cost_volume[0] + cost_volume[1]
        vv_cost_volume = torch.where(cost_volume[0] * mask_both_96 < cost_volume[1] * mask_both_96,
                                     cost_volume[0] * mask_both_96,
                                     cost_volume[1] * mask_both_96) + v_cost_volume * one_mask_96

        volume_feature = self.reduce(torch.cat([cost_volume[0], cost_volume[1]], 1))

        volume_feature = volume_feature + vv_cost_volume

        post_matching_feats = self.reduce_conv(torch.cat([self.features[-1], volume_feature], 1))

        self.features.append(self.layer2(post_matching_feats))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))

        return self.features, lowest_cost, conf_mask.squeeze(1)


    def feature_extraction(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet"""

        image = (image - 0.45) / 0.225  # imagenet normalisation
        feats_0 = self.layer0(image)
        feats_1 = self.layer1(feats_0)

        if return_all_feats:
            return [feats_0, feats_1]
        else:
            return feats_1

    def indices_to_disparity(self, indices):
        """Convert cost volume indices to 1/depth for visualisation"""

        batch, height, width = indices.shape
        depth = self.depth_bins[indices.reshape(-1).cpu()]
        disp = 1 / depth.reshape((batch, height, width))
        return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask

    def cuda(self):
        super().cuda()
        self.backprojector.cuda()
        self.projector.cuda()
        self.is_cuda = True
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cuda()

    def cpu(self):
        super().cpu()
        self.backprojector.cpu()
        self.projector.cpu()
        self.is_cuda = False
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cpu()

    def to(self, device):
        if str(device) == 'cpu':
            self.cpu()
        elif str(device) == 'cuda':
            self.cuda()
        else:
            raise NotImplementedError
 
class ResnetEncoderMatching_FusedCostVolume_VMamba(nn.Module):

    def __init__(self, num_layers, pretrained, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=100.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear'):

        super(ResnetEncoderMatching_FusedCostVolume_VMamba, self).__init__()
        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_depth_bins = num_depth_bins
        self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        encoder = resnets[num_layers](pretrained)
        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height,
                                   width=self.matching_width)
        self.ssim = SSIM()

        self.compute_depth_bins(min_depth_bin, max_depth_bin)

        self.reduce = nn.Sequential(
            nn.Conv2d(2 * self.num_depth_bins, self.num_depth_bins, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(self.num_ch_enc[1] + self.num_depth_bins,
                      self.num_ch_enc[1], 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.vmamba_block = nn.Sequential(
            VSSBlock(hidden_dim=64, d_state=32),
            VSSBlock(hidden_dim=64, d_state=32),
        )

    def compute_depth_bins(self, min_depth_bin, max_depth_bin):

        if self.depth_binning == 'inverse':
            self.depth_bins = 1 / np.linspace(
                1 / max_depth_bin, 1 / min_depth_bin, self.num_depth_bins
            )[::-1]

        elif self.depth_binning == 'linear':
            self.depth_bins = np.linspace(min_depth_bin, max_depth_bin, self.num_depth_bins)

        elif self.depth_binning == 'sid':
            self.depth_bins = np.array([
                np.exp(np.log(min_depth_bin) +
                       np.log(max_depth_bin / min_depth_bin) * i / (self.num_depth_bins - 1))
                for i in range(self.num_depth_bins)
            ])
        else:
            raise NotImplementedError

        self.depth_bins = torch.from_numpy(self.depth_bins).float()

        self.warp_depths = []
        for depth in self.depth_bins:
            d = torch.ones((1, self.matching_height, self.matching_width)) * depth
            self.warp_depths.append(d)

        self.warp_depths = torch.stack(self.warp_depths, 0).float()

        if self.is_cuda:
            self.warp_depths = self.warp_depths.cuda()

    def match_features(self, current_feats, lookup_feats, relative_poses, K, invK, resflow, mode="train"):

        batch_cost_volume = []
        cost_volume_masks = []

        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width)
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)

            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = relative_poses[batch_idx:batch_idx + 1]

            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            _resflow = resflow[batch_idx:batch_idx + 1]

            world_points = self.backprojector(self.warp_depths, _invK)

            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]
                lookup_pose = _lookup_poses[:, lookup_idx]

                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])
                pix_locs = self.projector(world_points, _K, lookup_pose)
                pix_locs_flow = pix_locs + _resflow.permute(0, 2, 3, 1)

                warped_static = F.grid_sample(
                    lookup_feat, pix_locs, padding_mode='zeros',
                    mode='bilinear', align_corners=True
                )

                warped_dynamic = F.grid_sample(
                    lookup_feat, pix_locs_flow, padding_mode='zeros',
                    mode='bilinear', align_corners=True
                )

                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (self.matching_width - 1)
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

                edge_mask = (
                    (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) *
                    (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                ).float()

                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask *= current_mask

                diff_static = torch.abs(warped_static - current_feats[batch_idx:batch_idx + 1]).mean(1)
                diff_dynamic = torch.abs(warped_dynamic - current_feats[batch_idx:batch_idx + 1]).mean(1)

                diffs = torch.min(diff_static, diff_dynamic) * edge_mask

                cost_volume += diffs
                counts += (diffs > 0).float()

            cost_volume = cost_volume / (counts + 1e-7)

            missing_val_mask = (cost_volume == 0).float()

            if self.set_missing_to_max:
                cost_volume = cost_volume * (1 - missing_val_mask) + \
                              cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask

            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)

        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)

        return batch_cost_volume, cost_volume_masks

    def forward(self, current_image, lookup_images, poses, K, invK, resflow,
                min_depth_bin=None, max_depth_bin=None, flag=None):

        self.features = self.feature_extraction(current_image, return_all_feats=True)
        current_feats = self.features[-1]

        with torch.no_grad():
            if self.adaptive_bins:
                self.compute_depth_bins(min_depth_bin, max_depth_bin)

            batch_size, num_frames, chns, h, w = lookup_images.shape
            lookup_images = lookup_images.reshape(batch_size * num_frames, chns, h, w)

            lookup_feats = self.feature_extraction(lookup_images, return_all_feats=False)
            _, chns, h, w = lookup_feats.shape
            lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, h, w)

            cost_volume, missing_mask = self.match_features(
                current_feats, lookup_feats, poses, K, invK, resflow
            )

            confidence_mask = self.compute_confidence_mask(
                cost_volume.detach() * (1 - missing_mask.detach())
            )

        viz_cost_vol = cost_volume.clone().detach()
        viz_cost_vol[viz_cost_vol == 0] = 100
        _, argmin = torch.min(viz_cost_vol, 1)
        lowest_cost = self.indices_to_disparity(argmin)

        cost_volume *= confidence_mask.unsqueeze(1)
        combined_input = torch.cat([self.features[-1], cost_volume], dim=1)
        post_matching_feats = self.reduce_conv(combined_input)
        post_matching_feats = self.vmamba_block(post_matching_feats)
 

        self.features.append(self.layer2(post_matching_feats))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))

        return self.features, lowest_cost, confidence_mask

    def feature_extraction(self, image, return_all_feats=False):

        image = (image - 0.45) / 0.225
        feats_0 = self.layer0(image)
        feats_1 = self.layer1(feats_0)

        if return_all_feats:
            return [feats_0, feats_1]
        return feats_1

    def indices_to_disparity(self, indices):

        b, h, w = indices.shape
        depth = self.depth_bins[indices.reshape(-1).cpu()]
        disp = 1 / depth.reshape((b, h, w))
        return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins

        return ((cost_volume > 0).sum(1) == num_bins_threshold).float()

    def cuda(self):
        super().cuda()
        self.backprojector.cuda()
        self.projector.cuda()
        self.is_cuda = True
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cuda()

    def cpu(self):
        super().cpu()
        self.backprojector.cpu()
        self.projector.cpu()
        self.is_cuda = False
        if self.warp_depths is not None:
            self.warp_depths = self.warp_depths.cpu()

    def to(self, device):
        if str(device) == 'cpu':
            self.cpu()
        elif str(device) == 'cuda':
            self.cuda()
        else:
            raise NotImplementedError
 

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1, **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

 
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=0.5,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_core_windows

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core_windows(self, x: torch.Tensor, layer=1):
        return self.forward_corev0(x)

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W

        K = 4
        x_hwwh = torch.stack(
            [x.view(B, -1, L),
             torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1
        ).view(B, 2, -1, L)

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                             xs.view(B, K, -1, L),
                             self.x_proj_weight)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts.view(B, K, -1, L),
                           self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        wh_y = torch.transpose(
            out_y[:, 1].view(B, -1, W, H),
            dim0=2, dim1=3
        ).contiguous().view(B, -1, L)

        invwh_y = torch.transpose(
            inv_y[:, 1].view(B, -1, W, H),
            dim0=2, dim1=3
        ).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, layer=1, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y = self.forward_core(x, layer)
        y = y * F.silu(z)

        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        layer: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim 

        factor = 2.0 
        d_model = int(hidden_dim // factor)

        self.down = nn.Linear(hidden_dim, d_model)
        self.up = nn.Linear(d_model, hidden_dim)

        self.ln_1 = norm_layer(d_model)
        self.self_attention = SS2D(d_model=d_model, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.layer = layer

    def forward(self, x):

        B, C, H, W = x.shape

        assert C == self.hidden_dim, f"Expected {self.hidden_dim}, got {C}"

        x = x.permute(0, 2, 3, 1).contiguous()

        x_down = self.down(x)

        y = x_down + self.drop_path(self.self_attention(self.ln_1(x_down), self.layer))

        y = self.up(y) + x

        y = y.permute(0, 3, 1, 2).contiguous()

        return y