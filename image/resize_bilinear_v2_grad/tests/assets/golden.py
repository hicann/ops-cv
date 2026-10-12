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

"""TTK golden for ResizeBilinearV2Grad (2D bilinear backward, NCHW).

标杆标准: ops-precision-standard 浮点计算类 · 单标杆比对 · 优先级-1 高精度 CPU
实现. 采用 **Torch + TensorFlow + NumPy 混合标杆**, 按 (align_corners,
half_pixel_centers, scales) 在调用框架**之前**做确定性分支选择, 不重新设计后端
选择规则, 不捕获 Torch/TF 异常静默回退 NumPy. 详见仓外 BENCHMARK_STANDARD.md.

反向是前向 gather 的转置(scatter): 对每个目标像素 (oh,ow), 用与前向一致的 4 邻
点 / 双线性权重把 **实际上游梯度 grads[oh,ow]**(非 ones_like) 散射回源网格。

分支表 (反向, NCHW):
  A (ac=1, hp=0)             : torch.ops.aten.upsample_bilinear2d_backward(g,[Ho,Wo],[N,C,Hi,Wi],True,None,None)
  B (ac=0, hp=1)             : torch...upsample_bilinear2d_backward(g,[Ho,Wo],[N,C,Hi,Wi],False,sh,sw)
                               sh=sw=scale_attr (scales>0) 否则 None
  C (ac=0, hp=0) 无显式scales: tf.raw_ops.ResizeBilinearGrad(grads=NHWC, original_image=NHWC, ac=False, hp=False)
  D 其余(ac=0/hp=0+scales,    : NumPy kernel 转写(§2 权重的转置散射, float64) —— 成熟框架
     ac=1/hp=1)                  无法表达的组合, 标准允许的确定性 NumPy 分支
  ALL_COPY (in==out, 任意 hp)  : 直返 grads, 对齐 kernel IsMatchAllCopy
                               (gradsDtype==yDtype + H/W 长度相等, 不判 half_pixel/scales)

torch align_corners=False 恒为 half_pixel, 故 (ac=0,hp=0) 无法用 torch 表达 → TF;
TF ResizeBilinearGrad 只吃 size 不吃 scales, 故 (ac=0,hp=0)+scales → NumPy。
布局: torch 原生 NCHW; TF 需 grads 与 original_image 均转 NHWC, 结果转回 NCHW。
dtype: float64 计算 → 回原 dtype。
"""

import math

import numpy as np
import torch

__spec__ = {
    "resize_bilinear_v2_grad": "ResizeBilinearV2GradTestSpec",
}


# ---------------------------------------------------------------- 分支选择辅助
def _scale_attr(scales, idx):
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
def _torch_grad(grads, hi, wi, align_corners, scales):
    n, c, ho, wo = grads.shape
    g = torch.from_numpy(np.ascontiguousarray(grads, dtype=np.float64))
    if align_corners:
        sh = sw = None  # align_corners 忽略 scales(与 CANN 一致)
    else:
        sh = _scale_attr(scales, 0)
        sw = _scale_attr(scales, 1)
    y = torch.ops.aten.upsample_bilinear2d_backward(
        g,
        [int(ho), int(wo)],
        [int(n), int(c), int(hi), int(wi)],
        bool(align_corners),
        sh,
        sw,
    )
    return y.numpy()


# ---------------------------------------------------------------- TF 分支 C
def _tf_grad(grads, original_image, hi, wi):
    import tensorflow as tf

    n, c, ho, wo = grads.shape
    # TF ResizeBilinearGrad 仅注册 float32(legacy TF 兼容语义即 float32 计算),
    # 与该分支对应的 kernel 语义一致; 结果升回 float64 容器保持统一输出契约。
    g_nhwc = np.ascontiguousarray(np.transpose(grads.astype(np.float32), (0, 2, 3, 1)))
    # original_image 仅用于形状/dtype, 内容不参与梯度; 传 float32 NHWC 零张量即可
    img_nhwc = np.zeros((n, hi, wi, c), np.float32)
    y = tf.raw_ops.ResizeBilinearGrad(
        grads=tf.constant(g_nhwc),
        original_image=tf.constant(img_nhwc),
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


def _numpy_grad(grads, hi, wi, align_corners, half_pixel, scales):
    n, c, ho, wo = grads.shape
    gf = grads.astype(np.float64)
    sh = _compute_scale(hi, ho, align_corners, _scale_attr(scales, 0))
    sw = _compute_scale(wi, wo, align_corners, _scale_attr(scales, 1))
    wmap = [_axis_weight(_src_coord(ow, sw, half_pixel), wi) for ow in range(wo)]
    y = np.zeros((n, c, hi, wi), np.float64)
    for oh in range(ho):
        ty, by, dy = _axis_weight(_src_coord(oh, sh, half_pixel), hi)
        wy_top = 1.0 - dy
        wy_bot = dy
        for ow in range(wo):
            lx, rx, dx = wmap[ow]
            wx_left = 1.0 - dx
            wx_right = dx
            g = gf[:, :, oh, ow]
            y[:, :, ty, lx] += g * (wy_top * wx_left)
            y[:, :, ty, rx] += g * (wy_top * wx_right)
            y[:, :, by, lx] += g * (wy_bot * wx_left)
            y[:, :, by, rx] += g * (wy_bot * wx_right)
    return y


def _bilinear_grad(grads, hi, wi, align_corners, half_pixel, scales):
    # ALL_COPY: kernel IsMatchAllCopy 仅判 gradsDtype==yDtype + H/W 长度相等
    # (resize_bilinear_v2_grad_tiling_arch35.cpp:310-312), 不检查 half_pixel/scales。
    # src==dst 时无论 ac/hp/scales 梯度原样回传(拷贝前向的转置仍是拷贝)。
    n, c, ho, wo = grads.shape
    if hi == ho and wi == wo:
        return grads.astype(np.float64)
    # 确定性分支选择 (调用框架之前)
    if align_corners and not half_pixel:  # 分支 A
        return _torch_grad(grads, hi, wi, True, None)
    if (not align_corners) and half_pixel:  # 分支 B
        return _torch_grad(grads, hi, wi, False, scales)
    if (not align_corners) and (not half_pixel):  # 分支 C / D
        if _has_explicit_scales(scales):  # 分支 D
            return _numpy_grad(grads, hi, wi, False, False, scales)
        return _tf_grad(grads, None, hi, wi)  # 分支 C (legacy TF 兼容)
    # ac=1 & hp=1: 成熟框架无对应, NumPy kernel 转写 (分支 D)
    return _numpy_grad(grads, hi, wi, True, True, scales)


class ResizeBilinearV2GradTestSpec:
    """TestSpec golden for ResizeBilinearV2Grad."""

    @staticmethod
    def golden(
        grads,
        original_image,
        *,
        align_corners=False,
        half_pixel_centers=False,
        scales=None,
        **kwargs,
    ):
        grads = np.ascontiguousarray(grads)
        original_image = np.ascontiguousarray(original_image)
        assert grads.ndim == 4 and original_image.ndim == 4, (
            "ResizeBilinearV2Grad expects 4-D NCHW tensors"
        )
        hi, wi = int(original_image.shape[2]), int(original_image.shape[3])
        y = _bilinear_grad(
            grads, hi, wi, bool(align_corners), bool(half_pixel_centers), scales
        )
        return [y.astype(original_image.dtype, copy=False)]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
