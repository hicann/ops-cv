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

"""TTK golden for ResizeBilinearV2 (2D bilinear forward, NCHW).

标杆标准: ops-precision-standard 浮点计算类 · 单标杆比对 · 优先级-1 高精度 CPU
实现. 采用 **Torch + TensorFlow + NumPy 混合标杆**, 按 (align_corners,
half_pixel_centers, scales) 在调用框架**之前**做确定性分支选择, 不重新设计后端
选择规则, 不捕获 Torch/TF 异常静默回退 NumPy. 详见仓外 BENCHMARK_STANDARD.md.

分支表 (前向, NCHW):
  A (ac=1, hp=0)             : torch.ops.aten.upsample_bilinear2d(x,[Ho,Wo],True,None,None)   ~1e-16
  B (ac=0, hp=1)             : torch.ops.aten.upsample_bilinear2d(x,[Ho,Wo],False,sh,sw)      ~1e-16
                               sh=sw=scale_attr (scales>0) 否则 None
  C (ac=0, hp=0) 无显式scales: tf.raw_ops.ResizeBilinear(NHWC,[Ho,Wo],ac=False,hp=False)      ~1e-7
  D 其余(如 ac=0/hp=0+scales, : NumPy kernel 转写(§2 坐标映射, float64)  —— 成熟框架无法表达
     ac=1/hp=1)                  的组合, 标准允许的确定性 NumPy 分支
  ALL_COPY (in==out, 任意 hp)  : 直返输入, 对齐 kernel IsMatchAllCopy
                               (dtype 相等 + H/W 长度相等, 不判 half_pixel/scales)

torch align_corners=False 恒为 half_pixel, 故 (ac=0,hp=0) 无法用 torch 表达 → TF;
TF ResizeBilinear 只吃 size 不吃 scales, 故 (ac=0,hp=0)+scales → NumPy。
布局: torch 原生 NCHW; TF 需转 NHWC 再转回。dtype: float64 计算 → 回原 dtype。
"""

import math

import numpy as np
import torch

__spec__ = {
    "resize_bilinear_v2": "ResizeBilinearV2TestSpec",
}


# ---------------------------------------------------------------- 分支选择辅助
def _scale_attr(scales, idx):
    """scales[idx] as positive float, or None when absent/<=0."""
    if scales is None:
        return None
    try:
        v = float(scales[idx])
    except (IndexError, TypeError, ValueError):
        return None
    return v if v > 0.0 else None


def _has_explicit_scales(scales):
    return _scale_attr(scales, 0) is not None or _scale_attr(scales, 1) is not None


# ---------------------------------------------------------------- torch 分支 A/B
def _torch_forward(x, ho, wo, align_corners, scales):
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    if align_corners:
        sh = sw = None  # align_corners 忽略 scales(与 CANN 一致)
    else:
        sh = _scale_attr(scales, 0)
        sw = _scale_attr(scales, 1)
    y = torch.ops.aten.upsample_bilinear2d(
        t, [int(ho), int(wo)], bool(align_corners), sh, sw
    )
    return y.numpy()


# ---------------------------------------------------------------- TF 分支 C
def _tf_forward(x, ho, wo):
    import tensorflow as tf

    # TF ResizeBilinear/Grad 仅注册 float32(legacy TF 兼容语义即 float32 计算),
    # 与该分支对应的 kernel 语义一致; 结果升回 float64 容器保持统一输出契约。
    nhwc = np.ascontiguousarray(np.transpose(x.astype(np.float32), (0, 2, 3, 1)))
    y = tf.raw_ops.ResizeBilinear(
        images=tf.constant(nhwc),
        size=tf.constant([int(ho), int(wo)], dtype=tf.int32),
        align_corners=False,
        half_pixel_centers=False,
    ).numpy()
    return np.ascontiguousarray(np.transpose(y, (0, 3, 1, 2))).astype(np.float64)


# ---------------------------------------------------------------- NumPy 分支 D
def _compute_scale(src, dst, align_corners, scale_attr):
    if align_corners:
        return (src - 1.0) / (dst - 1.0) if dst > 1 else float(src) / float(dst)
    if scale_attr is not None and scale_attr > 0.0:
        return 1.0 / float(scale_attr)
    return float(src) / float(dst)


def _src_coord(dst_idx, scale, half_pixel):
    v = (dst_idx + 0.5) * scale - 0.5 if half_pixel else dst_idx * scale
    return v if v > 0.0 else 0.0


def _axis_weight(orig, src_len):
    left = int(math.floor(orig))
    if left > src_len - 1:
        left = src_len - 1
    right = left + 1
    if right > src_len - 1:
        right = src_len - 1
    delta = orig - left
    if delta < 0.0:
        delta = 0.0
    elif delta > 1.0:
        delta = 1.0
    return left, right, delta


def _numpy_forward(x, ho, wo, align_corners, half_pixel, scales):
    n, c, hi, wi = x.shape
    xf = x.astype(np.float64)
    sh = _compute_scale(hi, ho, align_corners, _scale_attr(scales, 0))
    sw = _compute_scale(wi, wo, align_corners, _scale_attr(scales, 1))
    wmap = [_axis_weight(_src_coord(ow, sw, half_pixel), wi) for ow in range(wo)]
    y = np.empty((n, c, ho, wo), np.float64)
    for oh in range(ho):
        ty, by, dy = _axis_weight(_src_coord(oh, sh, half_pixel), hi)
        top = xf[:, :, ty, :]
        bot = xf[:, :, by, :]
        for ow in range(wo):
            lx, rx, dx = wmap[ow]
            vt = top[:, :, lx] * (1.0 - dx) + top[:, :, rx] * dx
            vb = bot[:, :, lx] * (1.0 - dx) + bot[:, :, rx] * dx
            y[:, :, oh, ow] = vt * (1.0 - dy) + vb * dy
    return y


def _bilinear_forward(x, ho, wo, align_corners, half_pixel, scales):
    # ALL_COPY: kernel IsMatchAllCopy 仅判 dtype 相等 + H/W 长度相等
    # (resize_bilinear_v2_tiling_arch35.cpp:245-252), 不检查 half_pixel/scales。
    # in==out 时无论 ac/hp/scales 均原样搬运, dtype 由用例构造保证相等。
    n, c, hi, wi = x.shape
    if hi == ho and wi == wo:
        return x.astype(np.float64)
    # 确定性分支选择 (调用框架之前)
    if align_corners and not half_pixel:  # 分支 A
        return _torch_forward(x, ho, wo, True, None)
    if (not align_corners) and half_pixel:  # 分支 B
        return _torch_forward(x, ho, wo, False, scales)
    if (not align_corners) and (not half_pixel):  # 分支 C / D
        if _has_explicit_scales(scales):  # 分支 D
            return _numpy_forward(x, ho, wo, False, False, scales)
        return _tf_forward(x, ho, wo)  # 分支 C (legacy TF 兼容)
    # ac=1 & hp=1: 成熟框架无对应, NumPy kernel 转写 (分支 D)
    return _numpy_forward(x, ho, wo, True, True, scales)


class ResizeBilinearV2TestSpec:
    """TestSpec golden for ResizeBilinearV2."""

    @staticmethod
    def golden(
        x,
        size,
        *,
        align_corners=False,
        half_pixel_centers=False,
        dtype=0,
        scales=None,
        **kwargs,
    ):
        x = np.ascontiguousarray(x)
        assert x.ndim == 4, "ResizeBilinearV2 expects a 4-D NCHW input"
        dst = np.asarray(size).reshape(-1)
        assert dst.size >= 2, "size input must carry [H_out, W_out]"
        ho, wo = int(dst[0]), int(dst[1])
        y = _bilinear_forward(
            x, ho, wo, bool(align_corners), bool(half_pixel_centers), scales
        )
        return [y.astype(x.dtype, copy=False)]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
