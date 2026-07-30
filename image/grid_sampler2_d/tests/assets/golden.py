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
import torch

# 属性值映射表（string -> int 索引，与 SE 文档第6章一致）
_INTERP_MODES = ["bilinear", "nearest", "bicubic"]
_PADDING_MODES = ["zeros", "border", "reflection"]


def _resolve_interp_mode(value):
    """将 interpolation_mode 兼容 string/int 输入为 int 索引"""
    if isinstance(value, str):
        return _INTERP_MODES.index(value)
    return int(value)


def _resolve_padding_mode(value):
    """将 padding_mode 兼容 string/int 输入为 int 索引"""
    if isinstance(value, str):
        return _PADDING_MODES.index(value)
    return int(value)


class GridSampler2DComposeSpec:
    """Class form — small-op composition for grid_sampler2_d"""

    # -- golden — function form --
    @staticmethod
    def golden(
        x, grid, *, interpolation_mode=0, padding_mode=0, align_corners=False, **kwargs
    ):
        """
        Golden function for grid_sampler2_d.
        All the parameters (names and order) follow SE doc prototype definition without outputs.
        All the input Tensors are numpy.ndarray.

        Args:
            x: input feature map, shape (N, C, H_in, W_in)
            grid: sampling grid, shape (N, H_out, W_out, 2)
            interpolation_mode: 0=bilinear, 1=nearest, 2=bicubic (or string)
            padding_mode: 0=zeros, 1=border, 2=reflection (or string)
            align_corners: bool, coordinate alignment
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            Output tensor, shape (N, C, H_out, W_out)
        """
        interp_idx = _resolve_interp_mode(interpolation_mode)
        pad_idx = _resolve_padding_mode(padding_mode)

        orig_dtype = x.dtype
        # 避免 float16/bfloat16 精度丢失或下溢，统一在 float32 下计算
        if orig_dtype.name in ("bfloat16", "float16"):
            x = x.astype(np.float32)
            grid = grid.astype(np.float32)

        x_torch = torch.from_numpy(np.ascontiguousarray(x))
        grid_torch = torch.from_numpy(np.ascontiguousarray(grid))

        result = torch.nn.functional.grid_sample(
            x_torch,
            grid_torch,
            mode=_INTERP_MODES[interp_idx],
            padding_mode=_PADDING_MODES[pad_idx],
            align_corners=align_corners,
        )
        result_np = result.numpy()

        # 还原为原始数据类型
        if orig_dtype.name in ("bfloat16", "float16"):
            result_np = result_np.astype(orig_dtype, copy=False)

        return result_np

    # -- third_party — dict multi-vendor --
    class ThirdPartyImpl:
        def __init__(
            self, *, interpolation_mode=0, padding_mode=0, align_corners=False, **kwargs
        ):
            self.interpolation_mode = interpolation_mode
            self.padding_mode = padding_mode
            self.align_corners = align_corners

        def __call__(self, x, grid, **kwargs):
            interp_idx = _resolve_interp_mode(self.interpolation_mode)
            pad_idx = _resolve_padding_mode(self.padding_mode)

            # 与 golden 对齐: fp16/bf16 输入升 fp32 计算, 避免 bicubic/reflection
            # 等精度敏感模式在 fp16 下中间计算丢精度 (与 golden 不同精度级别导致
            # cross_check 时 |b-g| 异常大)。
            orig_dtype = x.dtype if isinstance(x, torch.Tensor) else None
            if orig_dtype is not None and orig_dtype in (torch.float16, torch.bfloat16):
                x = x.to(torch.float32)
                grid = grid.to(torch.float32)

            output = torch.nn.functional.grid_sample(
                x,
                grid,
                mode=_INTERP_MODES[interp_idx],
                padding_mode=_PADDING_MODES[pad_idx],
                align_corners=self.align_corners,
            )

            if orig_dtype is not None and orig_dtype in (torch.float16, torch.bfloat16):
                output = output.to(orig_dtype)
            return output

    third_party = {
        "torch": ThirdPartyImpl,
    }

    # -- tolerance --
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "grid_sampler2_d": "GridSampler2DComposeSpec",
}
