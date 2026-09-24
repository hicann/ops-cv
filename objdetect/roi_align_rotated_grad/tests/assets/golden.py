#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Golden TestSpec for roi_align_rotated_grad (ascend950) — golden_new 修订版。

与 golden.py 的差异：golden() 与 ThirdPartyImpl 都不再使用小算子拼接复刻，
统一调用 MMCV ``roi_align_rotated`` 正向算子的 backward（autograd 触发
mmcv ext 的 roi_align_rotated_backward kernel，CPU/CUDA 由 device registry
自动派发）。因此本文件要求执行环境预装完整版 mmcv（mmcv-lite 不含 ops）。

功能唯一真值：MMCV ``roi_align_rotated_backward``（SE 文档 §7，v1.1 修订版）。
布局契约（910B 原型）：
  - x_grad: [N, pooled_h, pooled_w, C]（C-last）
  - rois:   [6, N]（转置布局，第 0~5 行 = batch_ind/cx/cy/w/h/angle，angle 弧度）
  - y_grad: [B, H, W, C]（shape 由属性 y_grad_shape 透传）
MMCV 布局契约（NCHW 原型）：
  - input:  [B, C, H, W]，rois: [N, 6]，grad_output: [N, C, pooled_h, pooled_w]
  两侧布局在共享核内做 permute/转置转换，公式语义不变。

注意：mmcv CUDA backward 为 atomicAdd 实现，存在 run-to-run 抖动；
mmcv CPU backward 为单线程串行累加，结果确定。容差走 cross_check L0。
"""

import numpy as np
import torch

__spec__ = {
    "roi_align_rotated_grad": "RoiAlignRotatedGradTestSpec",
}


def _mmcv_roi_align_rotated_grad(
    x_grad,
    rois,
    y_grad_shape,
    pooled_h,
    pooled_w,
    spatial_scale,
    sampling_ratio,
    aligned,
    clockwise,
):
    """共享 MMCV 计算核心：正向算子 + autograd backward。

    golden wrapper 与 ThirdPartyImpl 复用本核心；核心只做布局转换并调用
    MMCV 正向算子的 backward，不做 numpy 转换或 provider 选择。

    Args:
        x_grad: torch.Tensor [N, pooled_h, pooled_w, C]（C-last，910B 布局）
        rois:   torch.Tensor [6, N]（转置布局，910B 约定）
        y_grad_shape: list[int] 长度 4，语义 [B, H, W, C]
        pooled_h / pooled_w / spatial_scale / sampling_ratio / aligned / clockwise:
            算子属性（与 REG_OP 定义一致）
    Returns:
        y_grad: torch.Tensor [B, H, W, C]（未覆盖位置为 0）
    """
    from mmcv.ops import roi_align_rotated

    batch_size, height, width, channels = (
        int(y_grad_shape[0]),
        int(y_grad_shape[1]),
        int(y_grad_shape[2]),
        int(y_grad_shape[3]),
    )

    # 输出预置零（对标 MMCV new_zeros 语义）；n_rois==0 时无贡献直接返回
    if x_grad.shape[0] == 0:
        return torch.zeros(
            batch_size,
            height,
            width,
            channels,
            dtype=x_grad.dtype,
            device=x_grad.device,
        )

    # MMCV NCHW 输入特征图：零初始化并挂 autograd，backward 后 .grad 即
    # roi_align_rotated_backward 的输出（等价 MMCV new_zeros 语义）
    feature = torch.zeros(
        batch_size,
        channels,
        height,
        width,
        dtype=x_grad.dtype,
        device=x_grad.device,
        requires_grad=True,
    )
    # [6, N] -> [N, 6]（MMCV 行式 rois 布局）
    rois_mmcv = rois.t().contiguous()

    # 位置参数调用：部分 mmcv 版本将 roi_align_rotated 直接导出为
    # RoIAlignRotatedFunction.apply，不接受关键字参数
    output = roi_align_rotated(
        feature,
        rois_mmcv,
        (pooled_h, pooled_w),
        spatial_scale,
        sampling_ratio,
        aligned,
        clockwise,
    )

    # C-last grad_output -> MMCV NCHW grad_output
    grad_output = x_grad.permute(0, 3, 1, 2).contiguous()
    output.backward(grad_output)

    # NCHW 输入梯度 -> C-last 输出契约 [B, H, W, C]
    return feature.grad.permute(0, 2, 3, 1).contiguous()


class RoiAlignRotatedGradTestSpec:
    """One TestSpec shared by Kernel and GEIR for roi_align_rotated_grad."""

    def golden(
        x_grad,
        rois,
        y_grad_shape,
        pooled_h,
        pooled_w,
        spatial_scale,
        sampling_ratio=0,
        aligned=True,
        clockwise=False,
        **kwargs,
    ):
        """Kernel/GEIR Golden：numpy 输入 -> numpy 输出（list）。

        参数名与顺序与 roi_align_rotated_grad_def.cpp / REG_OP 一致：
        输入 x_grad, rois 在前；属性 y_grad_shape, pooled_h, pooled_w,
        spatial_scale, sampling_ratio, aligned, clockwise 按关键字传递。
        """
        x_t = torch.from_numpy(np.ascontiguousarray(x_grad))
        rois_t = torch.from_numpy(np.ascontiguousarray(rois))
        result = _mmcv_roi_align_rotated_grad(
            x_t,
            rois_t,
            y_grad_shape,
            pooled_h,
            pooled_w,
            spatial_scale,
            sampling_ratio,
            aligned,
            clockwise,
        )
        return [result.detach().cpu().numpy()]

    class ThirdPartyImpl:
        """torch provider third-party：调用 MMCV 正向算子的 backward。

        通过 autograd 触发 mmcv ext 的 roi_align_rotated_backward kernel。
        直接消费原 dtype provider Tensor，不做 .cpu()/.numpy()/固定 dtype cast。
        """

        def __init__(
            self,
            y_grad_shape,
            pooled_h,
            pooled_w,
            spatial_scale,
            sampling_ratio=0,
            aligned=True,
            clockwise=False,
            **kwargs,
        ):
            self.y_grad_shape = y_grad_shape
            self.pooled_h = pooled_h
            self.pooled_w = pooled_w
            self.spatial_scale = spatial_scale
            self.sampling_ratio = sampling_ratio
            self.aligned = aligned
            self.clockwise = clockwise

        def __call__(self, x_grad, rois, **kwargs):
            result = _mmcv_roi_align_rotated_grad(
                x_grad,
                rois,
                self.y_grad_shape,
                self.pooled_h,
                self.pooled_w,
                self.spatial_scale,
                self.sampling_ratio,
                self.aligned,
                self.clockwise,
            )
            return [result]

    # GEIR remote dispatch needs an explicit provider mapping.
    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L0"},
    }
