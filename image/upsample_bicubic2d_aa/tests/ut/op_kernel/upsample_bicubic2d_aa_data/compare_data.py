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
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def load_bin(path, d_type):
    """读取 golden/output bin；bfloat16 以 uint16 读入后经 torch 转换（2 字节/元素）。"""
    if d_type == "bfloat16":
        import torch

        raw = np.fromfile(path, np.uint16)
        return torch.from_numpy(raw).view(torch.bfloat16).to(torch.float32).numpy()
    return np.fromfile(path, _np_dtype(d_type))


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


def compare_data(golden_file_lists, output_file_lists, d_type):
    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = load_bin(out, d_type)
        tmp_gold = load_bin(gold, d_type)
        diff_res = np.isclose(tmp_out, tmp_gold, 0.01, 0.01, True)
        diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    compare_data(golden_file_lists, output_file_lists, d_type)


if __name__ == "__main__":
    process(sys.argv[1])
