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

__input__ = {"kernel": {"roi_align_grad": "roi_align_grad_input_func"}}


def _gen_rois(xdiff_shape, spatial_scale, rois, rois_n):
    """Generate semantically correct rois data.

    rois[:, 0] = batch_idx, integer in [0, B-1]
    rois[:, 1:5] = [x1, y1, x2, y2], float coordinates in feature map scale
    """
    B, C, H, W = xdiff_shape
    N = rois.shape[0]
    if N == 0:
        if rois_n is not None:
            return rois, rois_n
        return rois, None

    # batch_idx: uniform in [0, B-1]
    rois[:, 0] = np.random.randint(0, B, size=N).astype(rois.dtype)

    # coordinates in original image scale: [0, W/scale] x [0, H/scale]
    x_max = W / max(spatial_scale, 1e-6)
    y_max = H / max(spatial_scale, 1e-6)

    x1 = np.random.uniform(0, x_max, size=N)
    y1 = np.random.uniform(0, y_max, size=N)
    x2 = np.minimum(x1 + np.random.uniform(1, x_max, size=N), x_max)
    y2 = np.minimum(y1 + np.random.uniform(1, y_max, size=N), y_max)

    rois[:, 1] = x1
    rois[:, 2] = y1
    rois[:, 3] = x2
    rois[:, 4] = y2

    if rois_n is not None:
        rois_n[:] = N
        return rois, rois_n
    return rois, None


def roi_align_grad_input_func(
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
    input_func function for roi_align_grad.
    All the parameters (names and order) follow @roi_align_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        ydiff: numpy array, shape (N, C, pooled_H, pooled_W), backward gradient input
        rois: numpy array, shape (N, 5), ROI coordinates [batch_idx, x1, y1, x2, y2]
        rois_n: numpy array (optional), shape (N,), int32, number of valid ROIs
        xdiff_shape: list/tuple (B, C, H, W), output shape
        pooled_width: int, pooled output width
        pooled_height: int, pooled output height
        spatial_scale: float, spatial scale factor
        sample_num: int, sampling number, 0 means auto-compute
        roi_end_mode: int, alignment mode (0/1/2/3)
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        ydiff, rois, rois_n (or None if rois_n not provided)
    """
    # ydiff keeps framework-generated data (controlled by dataRange)
    # rois regenerated with semantically correct per-column values
    rois_out, rois_n_out = _gen_rois(
        xdiff_shape,
        spatial_scale,
        rois.copy(),
        rois_n.copy() if rois_n is not None else None,
    )

    if rois_n is not None:
        return ydiff, rois_out, rois_n_out
    return ydiff, rois_out, None
