#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__golden__ = {
    "kernel": {"roi_align_grad": "roi_align_grad_golden"},
    "aclnn": {"aclnnRoiAlignV2Backward": "aclnn_roi_align_v2_backward_golden"},
}


def roi_align_grad_golden(
    ydiff,
    rois,
    rois_n=None,
    *,
    xdiff_shape,
    pooled_width,
    pooled_height,
    spatial_scale,
    sample_num=2,
    roi_end_mode=1,
    **kwargs,
):
    """
    Golden function for roi_align_grad.
    All the parameters (names and order) follow @roi_align_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        ydiff: numpy array, shape (N, C, pooled_H, pooled_W), backward gradient input
        rois: numpy array, shape (N, 5), ROI coordinates [batch_idx, x1, y1, x2, y2]
        rois_n: numpy array (optional), shape (B,), int32, number of ROIs per batch (unused in computation)
        xdiff_shape: tuple/list (B, C, H, W), output shape
        pooled_width: int, pooled output width
        pooled_height: int, pooled output height
        spatial_scale: float, spatial scale factor
        sample_num: int, sampling number, 0 means auto-compute
        roi_end_mode: int, alignment mode (0/1/2/3)
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        xdiff: numpy array, shape (B, C, H, W), input feature map gradient
    """
    B, C, H, W = xdiff_shape
    xdiff = np.zeros(xdiff_shape, dtype=np.float32)
    N = rois.shape[0]

    if N == 0 or ydiff.size == 0:
        return xdiff

    for n in range(N):
        batch_idx = int(rois[n, 0])
        if batch_idx < 0 or batch_idx >= B:
            continue

        x1 = rois[n, 1] * spatial_scale
        y1 = rois[n, 2] * spatial_scale
        x2 = rois[n, 3] * spatial_scale
        y2 = rois[n, 4] * spatial_scale

        # Alignment mode handling
        if roi_end_mode > 0:
            if roi_end_mode == 1:
                # TF mode
                x2 = x2 + spatial_scale
                y2 = y2 + spatial_scale
            else:
                # PyTorch/MMDetection mode (roi_end_mode == 2 or 3)
                x1 = x1 - 0.5
                y1 = y1 - 0.5
                x2 = x2 - 0.5
                y2 = y2 - 0.5

        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_end_mode < 2:
            roi_w = max(roi_w, 1.0)
            roi_h = max(roi_h, 1.0)

        bin_w = roi_w / pooled_width
        bin_h = roi_h / pooled_height

        # Sampling number
        if sample_num > 0:
            roi_bin_grid_h = sample_num
            roi_bin_grid_w = sample_num
        else:
            roi_bin_grid_h = int(np.ceil(roi_h / pooled_height))
            roi_bin_grid_w = int(np.ceil(roi_w / pooled_width))

        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)

        for c in range(C):
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    grad_val = ydiff[n, c, ph, pw]

                    for iy in range(roi_bin_grid_h):
                        y = y1 + ph * bin_h + (iy + 0.5) * bin_h / roi_bin_grid_h
                        for ix in range(roi_bin_grid_w):
                            x = x1 + pw * bin_w + (ix + 0.5) * bin_w / roi_bin_grid_w

                            # Bilinear interpolation gradient
                            if y < -1.0 or y > H or x < -1.0 or x > W:
                                continue

                            y = max(y, 0.0)
                            x = max(x, 0.0)

                            y_low = int(y)
                            x_low = int(x)

                            if y_low >= H - 1:
                                y_high = y_low = H - 1
                                y = float(y_low)
                            else:
                                y_high = y_low + 1

                            if x_low >= W - 1:
                                x_high = x_low = W - 1
                                x = float(x_low)
                            else:
                                x_high = x_low + 1

                            ly = y - y_low
                            lx = x - x_low
                            hy = 1.0 - ly
                            hx = 1.0 - lx

                            w1 = hy * hx
                            w2 = hy * lx
                            w3 = ly * hx
                            w4 = ly * lx

                            g = grad_val / count
                            xdiff[batch_idx, c, y_low, x_low] += g * w1
                            xdiff[batch_idx, c, y_low, x_high] += g * w2
                            xdiff[batch_idx, c, y_high, x_low] += g * w3
                            xdiff[batch_idx, c, y_high, x_high] += g * w4

    return xdiff


def aclnn_roi_align_v2_backward_golden(
    gradOutput,
    boxes,
    inputShape,
    pooledHeight,
    pooledWidth,
    spatialScale,
    samplingRatio,
    aligned,
    gradInput,
    **kwargs,
):
    """
    Aclnn golden for aclnnRoiAlignV2Backward.
    All the parameters (name & order) follow \
        function `aclnnRoiAlignV2BackwardGetWorkspaceSize` in \
        @aclnn_roi_align_v2_backward.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    """
    import torch

    # Convert tensors to numpy if needed
    if isinstance(gradOutput, torch.Tensor):
        gradOutput_np = gradOutput.numpy()
    else:
        gradOutput_np = gradOutput

    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.numpy()
    else:
        boxes_np = boxes

    # Map aclnn 'aligned' bool to kernel 'roi_end_mode' int
    # aligned=True -> roi_end_mode=2 (PyTorch aligned)
    # aligned=False -> roi_end_mode=0 (no alignment)
    roi_end_mode = 2 if aligned else 0

    # Convert inputShape to tuple if needed
    if isinstance(inputShape, (list, tuple)):
        xdiff_shape = tuple(inputShape)
    else:
        xdiff_shape = inputShape

    result = roi_align_grad_golden(
        gradOutput_np,
        boxes_np,
        xdiff_shape=xdiff_shape,
        pooled_width=pooledWidth,
        pooled_height=pooledHeight,
        spatial_scale=spatialScale,
        sample_num=samplingRatio,
        roi_end_mode=roi_end_mode,
    )

    return result
