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
    "kernel": {"anchor_response_flags": "anchor_response_flags_golden"}
}


def anchor_response_flags_golden(gt_bboxes, *, featmap_size, strides, num_base_anchors, **kwargs):
    '''
    Golden function for anchor_response_flags.
    All the parameters (names and order) follow @anchor_response_flags_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        gt_bboxes: numpy array, shape (N, 4), dtype float16/float32
                   Ground truth bounding box coordinates in [x1, y1, x2, y2] format.
        featmap_size: list of 2 ints, [feat_h, feat_w]
        strides: list of 2 ints, [stride_h, stride_w]
        num_base_anchors: int, number of base anchors per grid position
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor, shape (feat_h * feat_w * num_base_anchors,), dtype uint8
    '''
    feat_h, feat_w = featmap_size
    stride_h, stride_w = strides

    if stride_h == 0 or stride_w == 0:
        raise ZeroDivisionError("strides must not contain 0")

    N = gt_bboxes.shape[0]

    total_anchors = feat_h * feat_w * num_base_anchors
    flags = np.zeros(total_anchors, dtype=np.uint8)

    if N == 0:
        return flags

    # Cast to float32 for computation (avoid fp16 precision issues)
    gt_bboxes_f32 = gt_bboxes.astype(np.float32)

    cx = (gt_bboxes_f32[:, 0] + gt_bboxes_f32[:, 2]) * np.float32(0.5)
    cy = (gt_bboxes_f32[:, 1] + gt_bboxes_f32[:, 3]) * np.float32(0.5)

    bad_cx = np.isnan(cx) | np.isinf(cx)
    bad_cy = np.isnan(cy) | np.isinf(cy)
    cx[bad_cx] = 0.0
    cy[bad_cy] = 0.0

    raw_gx = np.floor(cx / np.float32(stride_h))
    raw_gy = np.floor(cy / np.float32(stride_w))

    safe_gx = np.where((raw_gx <= 0) | (raw_gx >= feat_w), 0.0, raw_gx)
    safe_gy = np.where((raw_gy <= 0) | (raw_gy >= feat_h), 0.0, raw_gy)

    grid_x = np.where(raw_gx <= 0, 0,
             np.where(raw_gx >= feat_w, feat_w - 1,
             safe_gx.astype(np.int64)))
    grid_y = np.where(raw_gy <= 0, 0,
             np.where(raw_gy >= feat_h, feat_h - 1,
             safe_gy.astype(np.int64)))
    grid_x[bad_cx] = 0
    grid_y[bad_cy] = 0

    # Step 4: Compute linear index
    grid_idx = grid_y * feat_w + grid_x

    # Step 5: Set response flags
    responsible_grid = np.zeros(feat_h * feat_w, dtype=np.uint8)
    responsible_grid[grid_idx] = 1

    # Step 6: Expand to all base anchors
    flags = np.repeat(responsible_grid, num_base_anchors)

    return flags
