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

"""TTK golden for ResizeLinearGrad (1D linear backward, NCL).

标杆标准: ops-precision-standard 浮点计算类 · 单标杆比对 · 优先级-1 高精度
CPU 实现 (PyTorch CPU aten 原生). 详见仓外 BENCHMARK_STANDARD.md.

正式 golden 直接调用 ``torch.ops.aten.upsample_linear1d_backward`` —— 前向
``upsample_linear1d`` 的原生反向, **使用实际上游梯度 grads**(非 ones_like),
不复刻 kernel、不依赖被测算子。

分支表 (单分支):
    条件      : 全部 (Linear 隐含 half_pixel = not align_corners)
    接口      : upsample_linear1d_backward(grad_NCL, [Lo], [N,C,Li], align_corners, scales)
    grad      : 实际上游梯度 grads (grad_output)
    input_size: 由 original_image 提供 [N,C,Li] (仅用形状)
    scales    : CANN ``scale`` 属性>0 时映射到 torch scales; 否则 None
    dtype     : float64 计算 → 回原 dtype
    实证偏差  : ≤6.66e-16 (adjoint 恒等 <fwd(x),g>==<x,bwd(g)> 成立, 含非均匀梯度)
    ALL_COPY  : L_out==L_in → 直返 grads(拷贝的反向仍是拷贝), 对齐 kernel
                ComputeKey ComputeDataCopy(忽略 scale)

不捕获异常静默回退: torch 调用失败即直接抛出。
"""

import numpy as np
import torch

__spec__ = {
    "resize_linear_grad": "ResizeLinearGradTestSpec",
}


def _scales_arg(scale):
    if scale is None:
        return None
    try:
        v = float(scale)
    except (TypeError, ValueError):
        return None
    return v if v > 0.0 else None


def _linear_grad(grads, input_size, align_corners, scale):
    """PyTorch CPU 高精度标杆: aten.upsample_linear1d_backward, float64。"""
    n, c, lo = grads.shape
    li = int(input_size[2])
    # kernel ComputeKey: lenSrcLOrUb_==lenDesL_ 时 ComputeDataCopy() 无条件纯拷贝
    # (resize_linear_grad_tiling_arch35.cpp:192-195)。拷贝前向的反向仍是拷贝,
    # grad 原样回传, 忽略 scale / align_corners。
    if int(lo) == li:
        return np.ascontiguousarray(grads, dtype=np.float64)
    g = torch.from_numpy(np.ascontiguousarray(grads, dtype=np.float64))
    y = torch.ops.aten.upsample_linear1d_backward(
        g,
        [int(lo)],
        [int(input_size[0]), int(input_size[1]), int(input_size[2])],
        bool(align_corners),
        _scales_arg(scale),
    )
    return y.numpy()


class ResizeLinearGradTestSpec:
    """TestSpec golden for ResizeLinearGrad."""

    @staticmethod
    def golden(grads, original_image, *, align_corners=False, scale=0.0, **kwargs):
        grads = np.ascontiguousarray(grads)
        original_image = np.ascontiguousarray(original_image)
        assert grads.ndim == 3 and original_image.ndim == 3, (
            "ResizeLinearGrad expects 3-D NCL tensors"
        )
        y = _linear_grad(grads, original_image.shape, bool(align_corners), scale)
        return [y.astype(original_image.dtype, copy=False)]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
