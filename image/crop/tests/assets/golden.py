#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


__golden__ = {"kernel": {"crop": "crop_golden"}}


def crop_golden(x, size, *, axis=2, offsets=None, **kwargs):
    """
    Golden function for crop.
    All the parameters (names and order) follow SE doc prototype definition without outputs.
    All the input Tensors are numpy.ndarray.

    Crop 算子 Golden 实现：从 x 中按 axis 和 offsets 裁剪出与 size 相同 shape 的区域。

    Args:
        x: 输入张量 (numpy array)
        size: 参考裁剪张量 (numpy array，仅使用其 shape)
        axis: 裁剪起始维度，默认 2
        offsets: 各维度偏移量列表

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    x_shape = list(x.shape)
    size_shape = list(size.shape)
    rank = len(x_shape)

    # @constraint: axis < 0 时转换为正值 axis += rank(x)
    if axis < 0:
        axis += rank

    # @constraint: offsets 长度为1时，从 axis 开始所有维度使用同一偏移；
    #              长度>1时必须等于 rank(x) - axis，前面补0
    if offsets is None:
        offsets = [0] * rank
    elif len(offsets) == 1:
        offsets = [0] * axis + [offsets[0]] * (rank - axis)
    else:
        offsets = [0] * axis + list(offsets)

    # 构建切片索引：前 axis 维保持完整，axis 及之后按 offsets 偏移裁剪到 size 对应维度
    slices = []
    for i in range(rank):
        if i < axis:
            slices.append(slice(None))
        else:
            start = offsets[i]
            end = start + size_shape[i]
            slices.append(slice(start, end))

    y = x[tuple(slices)].copy()
    return y
