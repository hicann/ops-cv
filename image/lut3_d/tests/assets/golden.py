#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np

__golden__ = {
    "kernel": {"lut3_d": "lut3_d_golden"},
}


def _lut3_d_compute(img, lut_table):
    """LUT3D 三线性插值核心计算（numpy 实现，来自 SE 文档 §6）。"""
    # 统一转为 float32 计算
    img_f32 = img.astype(np.float32)
    lut_table_f32 = lut_table.astype(np.float32)

    lut_n = lut_table_f32.shape[0]  # 查找表边长 N

    # 获取图像像素数和通道数
    orig_shape = img_f32.shape
    if len(orig_shape) == 3:
        H, W, C = orig_shape
        N_batch = 1
    elif len(orig_shape) == 4:
        N_batch, H, W, C = orig_shape
    else:
        raise ValueError(f"img dims should be 3 or 4, got {len(orig_shape)}")

    assert C == 3, f"img last dim must be 3 (RGB), got {C}"
    assert lut_table_f32.shape == (lut_n, lut_n, lut_n, 3), (
        f"lut_table shape must be [N,N,N,3], got {lut_table_f32.shape}"
    )

    # 展平为 [num_pixels, 3] 以便向量化计算
    pixels = img_f32.reshape(-1, 3)  # [num_pixels, 3]

    # 步骤1: 像素值缩放 [0, 255] → [0, N-1]
    # 与 kernel 对齐: kernel 用 fp32 标量 (N-1)/255. 做一次 vmuls,
    # 这里强制 fp32 标量, 避免 numpy 默认 fp64 promote 导致舍入不一致。
    scale = np.float32((lut_n - 1)) / np.float32(255.0)
    img_scaled = pixels * scale  # [num_pixels, 3], 通道顺序 B, G, R

    # 步骤2: 计算 floor/ceil 索引和插值权重
    # 与 kernel 对齐: kernel 用 vconv("floor")/vconv("ceil") 直接取整, 不做 clip。
    floor_idx = np.floor(img_scaled).astype(np.int32)
    ceil_idx = np.ceil(img_scaled).astype(np.int32)

    fract = (img_scaled - floor_idx.astype(np.float32)).astype(np.float32)
    # 与 kernel 对齐: kernel 用 vadds(-1)+vabs 实现 fract_1 = abs(fract - 1),
    # 而非直接 1.0 - fract。两者在 IEEE754 下舍入路径不同, 会有 1 ULP 差异。
    fract_1 = np.abs((fract - np.float32(1.0))).astype(np.float32)

    b_floor = floor_idx[:, 0]
    g_floor = floor_idx[:, 1]
    r_floor = floor_idx[:, 2]
    b_ceil = ceil_idx[:, 0]
    g_ceil = ceil_idx[:, 1]
    r_ceil = ceil_idx[:, 2]

    fract_b = fract[:, 0]
    fract_g = fract[:, 1]
    fract_r = fract[:, 2]
    fract_b_1 = fract_1[:, 0]
    fract_g_1 = fract_1[:, 1]
    fract_r_1 = fract_1[:, 2]

    # 步骤3: 计算 8 个角点的 LUT 线性索引
    # 与 kernel 对齐: kernel 在 fp32 下计算 b*N*N, g*N (vmuls), 再 vadd 两步累加,
    # 最后 vconv("round") 转 int32。这里复刻: fp32 索引 -> round -> int32。
    lut_n_fp32 = np.float32(lut_n)
    b_floor_f = (b_floor.astype(np.float32) * (lut_n_fp32 * lut_n_fp32)).astype(
        np.float32
    )
    g_floor_f = (g_floor.astype(np.float32) * lut_n_fp32).astype(np.float32)
    b_ceil_f = (b_ceil.astype(np.float32) * (lut_n_fp32 * lut_n_fp32)).astype(
        np.float32
    )
    g_ceil_f = (g_ceil.astype(np.float32) * lut_n_fp32).astype(np.float32)
    r_floor_f = r_floor.astype(np.float32)
    r_ceil_f = r_ceil.astype(np.float32)

    def _mk_idx(b_f, g_f, r_f):
        bg = (b_f + g_f).astype(np.float32)
        bgr = (bg + r_f).astype(np.float32)
        return np.round(bgr).astype(np.int32)

    idx_0 = _mk_idx(b_floor_f, g_floor_f, r_floor_f)
    idx_1 = _mk_idx(b_ceil_f, g_floor_f, r_floor_f)
    idx_2 = _mk_idx(b_floor_f, g_ceil_f, r_floor_f)
    idx_3 = _mk_idx(b_ceil_f, g_ceil_f, r_floor_f)
    idx_4 = _mk_idx(b_floor_f, g_floor_f, r_ceil_f)
    idx_5 = _mk_idx(b_ceil_f, g_floor_f, r_ceil_f)
    idx_6 = _mk_idx(b_floor_f, g_ceil_f, r_ceil_f)
    idx_7 = _mk_idx(b_ceil_f, g_ceil_f, r_ceil_f)

    # 步骤4: 查表获取 8 个角点的值
    lut_flat = lut_table_f32.reshape(-1, 3)

    v0 = lut_flat[idx_0]
    v1 = lut_flat[idx_1]
    v2 = lut_flat[idx_2]
    v3 = lut_flat[idx_3]
    v4 = lut_flat[idx_4]
    v5 = lut_flat[idx_5]
    v6 = lut_flat[idx_6]
    v7 = lut_flat[idx_7]

    # 步骤5: 三线性插值
    # 与 kernel 对齐: kernel 对每步插值用 vmul+vmul+vadd (先两次独立乘法再一次加法,
    # 非 FMA), 顺序 R->G->B。这里显式拆成临时变量, 避免 numpy/FMA 融合。
    r1 = (v0 * fract_r_1[:, np.newaxis]).astype(np.float32)
    r2 = (v4 * fract_r[:, np.newaxis]).astype(np.float32)
    vr0 = (r1 + r2).astype(np.float32)
    r1 = (v1 * fract_r_1[:, np.newaxis]).astype(np.float32)
    r2 = (v5 * fract_r[:, np.newaxis]).astype(np.float32)
    vr1 = (r1 + r2).astype(np.float32)
    r1 = (v2 * fract_r_1[:, np.newaxis]).astype(np.float32)
    r2 = (v6 * fract_r[:, np.newaxis]).astype(np.float32)
    vr2 = (r1 + r2).astype(np.float32)
    r1 = (v3 * fract_r_1[:, np.newaxis]).astype(np.float32)
    r2 = (v7 * fract_r[:, np.newaxis]).astype(np.float32)
    vr3 = (r1 + r2).astype(np.float32)

    g1 = (vr0 * fract_g_1[:, np.newaxis]).astype(np.float32)
    g2 = (vr2 * fract_g[:, np.newaxis]).astype(np.float32)
    vg0 = (g1 + g2).astype(np.float32)
    g1 = (vr1 * fract_g_1[:, np.newaxis]).astype(np.float32)
    g2 = (vr3 * fract_g[:, np.newaxis]).astype(np.float32)
    vg1 = (g1 + g2).astype(np.float32)

    b1 = (vg0 * fract_b_1[:, np.newaxis]).astype(np.float32)
    b2 = (vg1 * fract_b[:, np.newaxis]).astype(np.float32)
    lut_img_flat = (b1 + b2).astype(np.float32)

    # 恢复原始 shape
    lut_img = lut_img_flat.reshape(orig_shape).astype(np.float32)

    return lut_img


def lut3_d_golden(img, lut_table, **kwargs):
    """
    Golden function for lut3_d.
    All the parameters (names and order) follow @lut3_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        img: numpy.ndarray, input image, shape [H,W,3] or [N,H,W,3],
             dtype uint8 or float32, value range [0, 255]
        lut_table: numpy.ndarray, 3D lookup table, shape [lut_n, lut_n, lut_n, 3],
                   dtype uint8 or float32
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (numpy.ndarray, dtype float32)
    """
    return _lut3_d_compute(img, lut_table)
