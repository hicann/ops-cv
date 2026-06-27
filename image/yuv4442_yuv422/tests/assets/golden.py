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
    "kernel": {"yuv4442_yuv422": "yuv4442_yuv422_golden"}
}


def yuv4442_yuv422_golden(x, **kwargs):
    '''
    Golden function for yuv444_to_yuv422.
    All the parameters (names and order) follow @yuv444_to_yuv422_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: numpy.ndarray, shape (h, w, 4), dtype float16
           YUV444 input data, 4 channels per pixel: [Y, U, V, pad]
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        y: numpy.ndarray, shape (h, w, 2), dtype uint8
           YUV422 output data, YUYV packed
           Even column: [Y_even, U_even], Odd column: [Y_odd, V_odd]
           Direct selection chroma subsampling (cosited), aligned with canndev vreduce
    '''
    h, w, c = x.shape
    assert c == 4, f"Input channels must be 4, got {c}"

    x_f32 = x.astype(np.float32)

    # YUV444 input: 4 channels per pixel [Y, U, V, pad]
    y_ch = x_f32[:, :, 0]    # Y brightness channel
    u_ch = x_f32[:, :, 1]    # U chroma channel
    v_ch = x_f32[:, :, 2]    # V chroma channel
    # ch3 is pad, ignored

    # Direct selection chroma subsampling (no averaging, aligned with canndev):
    # Even column: pick Y_even, U_even
    # Odd column:  pick Y_odd, V_odd
    w_pairs = w // 2
    is_odd = (w % 2 == 1)
    w_even = w_pairs * 2

    # Even pixels: [Y_even, U_even]
    out_even = np.stack([y_ch[:, :w_even][:, 0::2], u_ch[:, :w_even][:, 0::2]], axis=-1)
    # Odd pixels: [Y_odd, V_odd]
    out_odd = np.stack([y_ch[:, :w_even][:, 1::2], v_ch[:, :w_even][:, 1::2]], axis=-1)

    y = np.empty((h, w, 2), dtype=np.float32)
    y[:, :w_even:2, :] = out_even
    y[:, 1:w_even:2, :] = out_odd

    # Handle orphan last column for odd width: output [Y_last, U_last]
    if is_odd:
        y[:, -1, 0] = y_ch[:, -1]
        y[:, -1, 1] = u_ch[:, -1]

    # Convert to uint8 (clip to [0, 255])
    y = np.clip(np.floor(y + 0.5), 0, 255).astype(np.uint8)

    return y
