#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
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

__spec__ = {
    "grid_unnormal": "GridUnnormalKernelSpec",
}
__golden__ = {"kernel": {"grid_unnormal": "grid_unnormal_golden"}}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}


def _normalize_attr_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _grid_unnormal_torch(grid, assist, align_corners):
    ori_dtype = grid.dtype
    torch_dtype = (
        torch.float16 if ori_dtype == np.float16 else torch.from_numpy(grid).dtype
    )
    grid_tensor = torch.from_numpy(grid).to(torch_dtype)
    assist_tensor = torch.from_numpy(assist).to(torch_dtype)
    diff, position = _grid_unnormal_compute(grid_tensor, assist_tensor, align_corners)
    return [
        diff.cpu().numpy().astype(ori_dtype, copy=False),
        position.cpu().numpy(),
    ]


def _grid_unnormal_compute(grid_tensor, assist_tensor, align_corners):
    out_dtype = grid_tensor.dtype
    compute_dtype = (
        torch.float32
        if grid_tensor.dtype in (torch.float16, torch.bfloat16)
        else grid_tensor.dtype
    )
    grid_compute = grid_tensor.to(compute_dtype)
    assist_compute = assist_tensor.to(compute_dtype)
    normalized = (grid_compute + 1.0) * 0.5
    if align_corners:
        pos_base = normalized * (assist_compute - 1.0)
    else:
        pos_base = normalized * assist_compute - 0.5
    floor = torch.floor(pos_base)
    position = floor.to(torch.int32)
    diff = (pos_base - floor).to(out_dtype)
    return [diff, position]


class _GridUnnormalCompose:
    def __init__(self, align_corners=False, **kwargs):
        self.align_corners = _normalize_attr_bool(align_corners)

    def __call__(self, grid, assist, **kwargs):
        grid_tensor = (
            grid
            if isinstance(grid, torch.Tensor)
            else torch.from_numpy(np.asarray(grid))
        )
        assist_tensor = (
            assist
            if isinstance(assist, torch.Tensor)
            else torch.from_numpy(np.asarray(assist))
        )
        return _grid_unnormal_compute(grid_tensor, assist_tensor, self.align_corners)


class GridUnnormalKernelSpec:
    """kernel + GEIR TestSpec entry. GridUnnormal has no aclnn/e2e path."""

    @staticmethod
    def golden(grid, assist, *, align_corners=False, **kwargs):
        return _grid_unnormal_torch(
            np.asarray(grid), np.asarray(assist), _normalize_attr_bool(align_corners)
        )

    third_party = {"torch": _GridUnnormalCompose}
    tolerance = _TOL


def grid_unnormal_golden(grid, assist, *, align_corners=False, **kwargs):
    """
    Golden for GridUnnormal（参数名/顺序对齐算子原型: grid, assist）。

        t        = (grid + 1) * 0.5
        pos_base = align_corners ? t * (assist - 1) : t * assist - 0.5
        position = floor(pos_base)            (int32)
        diff     = pos_base - floor(pos_base) (grid.dtype)

    依据需求规格公式使用 PyTorch 基础算子拼接作为独立参考。
    本轮 A5 规格要求 fp16/fp32 输入均提升到 fp32 计算，diff 回写 grid dtype。
    """
    align_corners = _normalize_attr_bool(align_corners)
    return _grid_unnormal_torch(grid, assist, align_corners)


# Not registered in __spec__:
# - aclnn/e2e: OpDef is aclnn_exclude and no torch_npu binding is delivered.
# - TensorFlow/ONNX/fusion: no parser or graph pass is delivered for this op.
