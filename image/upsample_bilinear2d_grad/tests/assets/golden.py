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
    "kernel": {"upsample_bilinear2d_grad": "upsample_bilinear2d_grad_golden"},
    "aclnn": {"aclnnUpsampleBilinear2dGrad": "aclnn_upsample_bilinear2d_grad_golden"}
}


def upsample_bilinear2d_grad_golden(grad_output, *, output_size, input_size,
                                     align_corners=False, scales_h=0.0, scales_w=0.0,
                                     **kwargs):
    '''
    Golden function for upsample_bilinear2d_grad.
    All the parameters (names and order) follow @upsample_bilinear2d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        grad_output: numpy array, shape [N, C, H_out, W_out]
        output_size: list [H_out, W_out]
        input_size: list [N, C, H_in, W_in]
        align_corners: bool
        scales_h: float
        scales_w: float
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (grad_input)
    '''
    orig_dtype = grad_output.dtype
    if orig_dtype.name in ('bfloat16', 'float16'):
        grad_output = grad_output.astype(np.float32)

    N, C, H_out, W_out = grad_output.shape
    _, _, H_in, W_in = input_size

    # Compute scale factors
    if scales_h == 0.0:
        if align_corners:
            scales_h = (H_in - 1) / (H_out - 1) if H_out > 1 else 0.0
        else:
            scales_h = H_in / H_out if H_out > 0 else 0.0

    if scales_w == 0.0:
        if align_corners:
            scales_w = (W_in - 1) / (W_out - 1) if W_out > 1 else 0.0
        else:
            scales_w = W_in / W_out if W_out > 0 else 0.0

    # Initialize grad_input
    grad_input = np.zeros((N, C, H_in, W_in), dtype=np.float32)

    # Bilinear interpolation backward (scatter-add)
    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    if align_corners:
                        h_src = scales_h * h_out
                        w_src = scales_w * w_out
                    else:
                        h_src = scales_h * (h_out + 0.5) - 0.5
                        w_src = scales_w * (w_out + 0.5) - 0.5
                        h_src = max(h_src, 0.0)
                        w_src = max(w_src, 0.0)

                    h0 = int(np.floor(h_src))
                    h1 = h0 + 1
                    w0 = int(np.floor(w_src))
                    w1 = w0 + 1

                    h1_weight = h_src - h0
                    h0_weight = 1.0 - h1_weight
                    w1_weight = w_src - w0
                    w0_weight = 1.0 - w1_weight

                    grad_val = grad_output[n, c, h_out, w_out]

                    hw00 = h0_weight * w0_weight
                    hw01 = h0_weight * w1_weight
                    hw10 = h1_weight * w0_weight
                    hw11 = h1_weight * w1_weight
                    if hw00 > 0.0 and 0 <= h0 < H_in and 0 <= w0 < W_in:
                        grad_input[n, c, h0, w0] += grad_val * hw00
                    if hw01 > 0.0 and 0 <= h0 < H_in and 0 <= w1 < W_in:
                        grad_input[n, c, h0, w1] += grad_val * hw01
                    if hw10 > 0.0 and 0 <= h1 < H_in and 0 <= w0 < W_in:
                        grad_input[n, c, h1, w0] += grad_val * hw10
                    if hw11 > 0.0 and 0 <= h1 < H_in and 0 <= w1 < W_in:
                        grad_input[n, c, h1, w1] += grad_val * hw11

    if orig_dtype.name in ('bfloat16', 'float16'):
        grad_input = grad_input.astype(orig_dtype, copy=False)

    return grad_input


def aclnn_upsample_bilinear2d_grad_golden(gradOutput, outputSize, inputSize,
                                           alignCorners, scalesH, scalesW,
                                           gradInput, **kwargs):
    '''
    Aclnn golden for aclnnUpsampleBilinear2dGrad.
    All the parameters (name & order) follow \
        function `aclnnUpsampleBilinear2dGradGetWorkspaceSize` in @aclnn_upsample_bilinear2d_grad.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    if isinstance(gradOutput, np.ndarray):
        go = gradOutput
    else:
        go = gradOutput.cpu().numpy() if hasattr(gradOutput, 'cpu') else np.array(gradOutput)

    output_size = list(outputSize) if not isinstance(outputSize, list) else outputSize
    input_size = list(inputSize) if not isinstance(inputSize, list) else inputSize

    return upsample_bilinear2d_grad_golden(
        go,
        output_size=output_size,
        input_size=input_size,
        align_corners=bool(alignCorners),
        scales_h=float(scalesH),
        scales_w=float(scalesW)
    )
