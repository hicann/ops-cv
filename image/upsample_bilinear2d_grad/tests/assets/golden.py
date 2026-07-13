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

import torch
import torch.nn.functional as F
import numpy as np

__golden__ = {
    "kernel": {"upsample_bilinear2d_grad": "upsample_bilinear2d_grad_golden"},
    "aclnn": {"aclnnUpsampleBilinear2dGrad": "aclnn_upsample_bilinear2d_grad_golden"}
}

_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    np.float16: torch.float16,
    np.float32: torch.float32,
}

try:
    _TORCH_DTYPE[np.dtype("bfloat16")] = torch.bfloat16
except TypeError:
    pass


def _to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    return _TORCH_DTYPE.get(dtype, torch.float32)


def _to_numpy_dtype_name(dtype):
    if hasattr(dtype, 'name'):
        return dtype.name
    return str(dtype)


def _bilinear2d_backward_torch(grad_out_t, H_out, W_out, H_in, W_in,
                                align_corners, torch_dtype):
    N, C = grad_out_t.shape[0], grad_out_t.shape[1]
    x = torch.zeros(N, C, H_in, W_in, dtype=torch_dtype, requires_grad=True)
    y = F.interpolate(x, size=(H_out, W_out), mode='bilinear',
                      align_corners=align_corners)
    y.backward(grad_out_t)
    return x.grad.detach()


def upsample_bilinear2d_grad_golden(grad_output, *, output_size, input_size,
                                     align_corners=False, scales_h=0.0, scales_w=0.0,
                                     **kwargs):
    '''
    Golden function for upsample_bilinear2d_grad using PyTorch autograd.

    Args:
        grad_output: numpy array, shape [N, C, H_out, W_out]
        output_size: list [H_out, W_out]
        input_size: list [N, C, H_in, W_in]
        align_corners: bool
        scales_h: float (unused, PyTorch computes from sizes)
        scales_w: float (unused, PyTorch computes from sizes)

    Returns:
        Output tensor (grad_input) as numpy array
    '''
    orig_dtype = grad_output.dtype
    torch_dtype = _to_torch_dtype(orig_dtype)

    N, C, H_out, W_out = grad_output.shape
    _, _, H_in, W_in = input_size

    grad_out_t = torch.from_numpy(grad_output.astype(np.float32)).to(torch_dtype)
    grad = _bilinear2d_backward_torch(grad_out_t, H_out, W_out, H_in, W_in,
                                       align_corners, torch_dtype)
    result = grad.float().numpy()

    dtype_name = _to_numpy_dtype_name(orig_dtype)
    if dtype_name == 'float16':
        result = result.astype(np.float16)
    elif dtype_name == 'bfloat16':
        try:
            result = result.astype(np.dtype("bfloat16"))
        except TypeError:
            pass

    return result


def aclnn_upsample_bilinear2d_grad_golden(gradOutput, outputSize, inputSize,
                                           alignCorners, scalesH, scalesW,
                                           gradInput, **kwargs):
    '''
    Aclnn golden for aclnnUpsampleBilinear2dGrad using PyTorch autograd.

    Args:
        gradOutput: torch.Tensor or numpy array, shape [N, C, H_out, W_out]
        outputSize: list or aclIntArray [H_out, W_out]
        inputSize: list or aclIntArray [N, C, H_in, W_in]
        alignCorners: bool
        scalesH: float/double (unused, PyTorch computes from sizes)
        scalesW: float/double (unused, PyTorch computes from sizes)
        gradInput: output tensor (unused, for signature compatibility)

    Returns:
        Output tensor (grad_input) as torch.Tensor
    '''
    if isinstance(gradOutput, np.ndarray):
        torch_dtype = _to_torch_dtype(gradOutput.dtype)
        grad_out_t = torch.from_numpy(gradOutput.astype(np.float32)).to(torch_dtype)
    elif isinstance(gradOutput, torch.Tensor):
        torch_dtype = gradOutput.dtype
        grad_out_t = gradOutput.detach().clone()
    else:
        torch_dtype = torch.float32
        grad_out_t = torch.tensor(gradOutput, dtype=torch_dtype)

    output_size = list(outputSize) if not isinstance(outputSize, list) else outputSize
    input_size = list(inputSize) if not isinstance(inputSize, list) else inputSize

    H_out, W_out = output_size[0], output_size[1]
    H_in, W_in = input_size[2], input_size[3]

    return _bilinear2d_backward_torch(grad_out_t, H_out, W_out, H_in, W_in,
                                       bool(alignCorners), torch_dtype)
