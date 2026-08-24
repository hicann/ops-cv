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
    "kernel": {"yolox_bounding_box_decode": "yolox_bounding_box_decode_golden"}
}


def yolox_bounding_box_decode_golden(priors, bboxes, **kwargs):
    """
    Golden function for yolox_bounding_box_decode.
    All the input Tensors are numpy.ndarray.

    Computation:
        xys = bboxes[..., :2] * priors[:, 2:] + priors[:, :2]
        whs = exp(bboxes[..., 2:]) * priors[:, 2:] * 0.5
        tl  = xys - whs
        br  = xys + whs
        decoded_bboxes = concat([tl, br], -1)   # [tl_x, tl_y, br_x, br_y]

    Args:
        priors: (N, 4) float16/float32, [grid_x*stride_x, grid_y*stride_y, stride_x, stride_y]
        bboxes: (B, N, 4) float16/float32, [dx, dy, dw, dh]
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        decoded_bboxes: (B, N, 4) numpy.ndarray, dtype same as bboxes
    """
    import torch

    priors_torch = torch.from_numpy(np.ascontiguousarray(priors))
    bboxes_torch = torch.from_numpy(np.ascontiguousarray(bboxes))

    xys = bboxes_torch[..., :2] * priors_torch[:, 2:] + priors_torch[:, :2]
    # Keep exp -> stride -> 0.5 operation order.
    whs = torch.exp(bboxes_torch[..., 2:]) * priors_torch[:, 2:] * 0.5
    tl = xys - whs
    br = xys + whs
    decoded_bboxes = torch.cat([tl, br], dim=-1)

    return decoded_bboxes.numpy()
