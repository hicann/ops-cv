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
    "kernel": {"rgb2_yuv422": "rgb2_yuv422_golden"}
}


def _rgb2_yuv422_impl(x, compute_dtype=np.float32):
    """
    RGB → YUV422 (YUYV packed) conversion, aligned with canndev AIPP /256 coefficients.

    Args:
        x: numpy ndarray, RGB input, shape [..., H, W, 3] (NHWC), uint8
        compute_dtype: numpy dtype for computation

    Returns:
        y: numpy ndarray, YUV422 output, shape [..., H, W, 2], uint8
    """
    x = np.asarray(x, dtype=compute_dtype)

    if x.shape[-1] != 3:
        raise ValueError(f"Input channel dimension must be 3, got {x.shape[-1]}")

    orig_shape = x.shape
    h, w = orig_shape[-3], orig_shape[-2]

    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]

    # canndev AIPP CSC 系数 (/256 定点)：
    #   Y_R=77, Y_G=150, Y_B=29
    #   U_R=-43, U_G=-85, U_B=128, U_offset=+128
    #   V_R=128, V_G=-107, V_B=-21, V_offset=+128
    c00, c01, c02 = np.asarray([77.0/256.0, 150.0/256.0, 29.0/256.0], dtype=compute_dtype)
    c10, c11, c12 = np.asarray([-43.0/256.0, -85.0/256.0, 128.0/256.0], dtype=compute_dtype)
    c20, c21, c22 = np.asarray([128.0/256.0, -107.0/256.0, -21.0/256.0], dtype=compute_dtype)

    y_full = c00 * r + c01 * g + c02 * b
    u_full = c10 * r + c11 * g + c12 * b + np.asarray(128.0, dtype=compute_dtype)
    v_full = c20 * r + c21 * g + c22 * b + np.asarray(128.0, dtype=compute_dtype)

    # YUV422 YUYV packing (aligned with canndev):
    #   Even column: [Y_even, U_even]
    #   Odd column:  [Y_odd, V_odd]
    yuv422 = np.zeros(list(orig_shape[:-1]) + [2], dtype=compute_dtype)

    for j in range(0, w, 2):
        yuv422[..., j, 0] = y_full[..., j]
        yuv422[..., j, 1] = u_full[..., j]
        if j + 1 < w:
            yuv422[..., j + 1, 0] = y_full[..., j + 1]
            yuv422[..., j + 1, 1] = v_full[..., j + 1]

    return yuv422


def rgb2_yuv422_golden(x, **kwargs):
    orig_dtype = x.dtype
    x_input = x.astype(np.float32)
    result = _rgb2_yuv422_impl(x_input, compute_dtype=np.float32)
    result = np.clip(np.round(result), 0, 255).astype(np.uint8)
    return result
