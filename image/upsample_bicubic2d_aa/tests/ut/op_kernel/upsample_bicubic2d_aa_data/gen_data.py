#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np
import torch


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list


def to_file(data, filename, d_type):
    """按 dtype 落盘；bfloat16 走 torch 真实转换（2 字节/元素，IEEE bf16 位布局）。"""
    if d_type == "bfloat16":
        t = torch.from_numpy(np.ascontiguousarray(data)).to(torch.bfloat16)
        t.view(torch.uint16).numpy().tofile(filename)
    else:
        data.astype(_np_dtype(d_type)).tofile(filename)


def _np_dtype(d_type):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    if d_type not in d_type_dict:
        raise ValueError(
            f"unsupported dtype: {d_type}, expected one of {list(d_type_dict)} or 'bfloat16'"
        )
    return d_type_dict[d_type]


def _to_np_of_dtype(data, d_type):
    """把 fp32 中间数据转成目标 dtype 的 numpy 数组（bfloat16 经 torch 真实转换）。"""
    if d_type == "bfloat16":
        return (
            torch.from_numpy(np.ascontiguousarray(data))
            .to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
        )
    return data.astype(_np_dtype(d_type))


def gen_data_and_golden(
    input_shape_str, output_size_str, d_type="float32", align_corners=False
):
    input_shape, _ = parse_str_to_shape_list(input_shape_str)
    _, output_size = parse_str_to_shape_list(output_size_str)
    # 输入先生成 fp32，再转换到目标 dtype（保证量化语义与真实算子输入一致）
    tmp_input = np.random.random(input_shape).astype(np.float32)
    if d_type == "bfloat16":
        input_typed = torch.from_numpy(tmp_input).to(torch.bfloat16)
        x_tensor = input_typed.to(torch.float32)  # 中间计算按 fp32
        tmp_input = input_typed.view(torch.uint16).numpy()
    elif d_type == "float16":
        input_typed = torch.from_numpy(tmp_input).to(torch.float16)
        x_tensor = input_typed.to(torch.float32)
        tmp_input = input_typed.numpy()
    else:
        x_tensor = torch.from_numpy(tmp_input)
    y_golden = torch.nn.functional.interpolate(
        x_tensor,
        output_size,
        mode="bicubic",
        align_corners=align_corners,
        antialias=True,
    )
    tmp_golden = _to_np_of_dtype(np.array(y_golden), d_type)

    tmp_input.tofile(f"{d_type}_input_bicubic2d_aa.bin")
    tmp_golden.tofile(f"{d_type}_golden_bicubic2d_aa.bin")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print(
            "Param num must be 4 or 5: <input_shape> <output_size> <dtype> [align_corners]"
        )
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    ac = sys.argv[4].lower() in ("true", "1") if len(sys.argv) == 5 else False
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], ac)
