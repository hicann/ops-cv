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

"""TTK golden for ResizeLinear (1D linear forward, NCL).

标杆标准: ops-precision-standard 浮点计算类 · 单标杆比对 · 优先级-1 高精度
CPU 实现 (PyTorch CPU aten 原生). 详见仓外 BENCHMARK_STANDARD.md.

正式 golden 直接调用 ``torch.ops.aten.upsample_linear1d`` —— PyTorch 原生 1D
线性插值高精度 CPU 实现, 不复刻 kernel、不依赖被测算子。

分支表 (单分支):
    条件      : 全部 (Linear 隐含 half_pixel = not align_corners)
    接口      : torch.ops.aten.upsample_linear1d(x_NCL, [Lo], align_corners, scales)
    scales    : CANN ``scale`` 属性>0 时映射到 torch scales(=scale_attr); 否则 None
    布局      : NCL 原生, 无需转置
    dtype     : float64 计算 → 回原 dtype
    实证偏差  : ≤4.44e-16 (含 src=1 / dst=1 / 同尺寸 / 非整数缩放)
    ALL_COPY  : L_out==L_in → 直返输入, 对齐 kernel schId-1(DIM_1, 忽略 scale)

torch align_corners=False 恒采用 half_pixel 坐标, 与 Linear kernel
``isHalfPiex_ = alignCorners ? 0 : 1`` 一致, 故无需独立 half_pixel 开关。
不捕获异常静默回退: torch 调用失败即直接抛出。
"""

import numpy as np
import torch

__spec__ = {
    "resize_linear": "ResizeLinearTestSpec",
}


def _scales_arg(scale):
    """CANN scale 属性 → torch scales; <=0/缺省返回 None(用 size 隐含 src/dst)。"""
    if scale is None:
        return None
    try:
        v = float(scale)
    except (TypeError, ValueError):
        return None
    return v if v > 0.0 else None


def _linear_forward(x, lo, align_corners, scale):
    """PyTorch CPU 高精度标杆: aten.upsample_linear1d, float64 计算。"""
    li = int(x.shape[2])
    # kernel ComputeKey schId-1(DIM_1): L_out==L_in 时无条件纯搬运,
    # 忽略 scale / align_corners(resize_linear_tiling_arch35.cpp:73-74)。
    if int(lo) == li:
        return np.ascontiguousarray(x, dtype=np.float64)
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    y = torch.ops.aten.upsample_linear1d(
        t, [int(lo)], bool(align_corners), _scales_arg(scale)
    )
    return y.numpy()


class ResizeLinearTestSpec:
    """TestSpec golden for ResizeLinear."""

    @staticmethod
    def golden(x, size, *, align_corners=False, scale=0.0, **kwargs):
        x = np.ascontiguousarray(x)
        assert x.ndim == 3, "ResizeLinear expects a 3-D NCL input"
        dst = np.asarray(size).reshape(-1)
        assert dst.size >= 1, "size input must carry [L_out]"
        lo = int(dst[0])
        y = _linear_forward(x, lo, bool(align_corners), scale)
        return [y.astype(x.dtype, copy=False)]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
