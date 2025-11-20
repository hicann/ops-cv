#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import os
import numpy as np
import re
import torch
from torch.nn.functional import interpolate


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list

def gen_data_and_golden(input_shape_str, output_size_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    np_type = d_type_dict[d_type]
    grad_input_shape, _ = parse_str_to_shape_list(input_shape_str)
    _, grad_output_size = parse_str_to_shape_list(output_size_str)

    size = np.prod(grad_input_shape)
    tmp_input = np.random.random(size).reshape(grad_input_shape).astype(np_type)
    input_tensor = torch.tensor(tmp_input, dtype=torch.float32).requires_grad_(True)
    grad_output_tensor = interpolate(input_tensor, grad_output_size, mode='nearest-exact')
    grad_output_tensor.backward(grad_output_tensor)
    grad_input_golden = np.array(input_tensor.grad).astype(np_type)
    grad_output = grad_output_tensor.detach().numpy()

    grad_output.astype(np_type).tofile(f"{d_type}_input_upsample_nearest_exact3d_grad.bin")
    grad_input_golden.astype(np_type).tofile(f"{d_type}_golden_upsample_nearest_exact3d_grad.bin")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])