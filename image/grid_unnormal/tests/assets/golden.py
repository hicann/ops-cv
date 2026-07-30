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

__golden__ = {"kernel": {"grid_unnormal": "grid_unnormal_golden"}}


def _normalize_attr_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _grid_unnormal_torch(grid, assist, align_corners):
    ori_dtype = grid.dtype
    torch_dtype = torch.float16 if ori_dtype == np.float16 else torch.float32
    grid_tensor = torch.from_numpy(grid).to(torch_dtype).to(torch.float32)
    assist_tensor = torch.from_numpy(assist).to(torch_dtype).to(torch.float32)
    normalized = (grid_tensor + 1.0) * 0.5
    if align_corners:
        pos_base = normalized * (assist_tensor - 1.0)
    else:
        pos_base = normalized * assist_tensor - 0.5
    floor = torch.floor(pos_base)
    position = floor.to(torch.int32).cpu().numpy()
    diff = (
        (pos_base - floor).to(torch_dtype).cpu().numpy().astype(ori_dtype, copy=False)
    )
    return [diff, position]


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
