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

import torch
import torch_npu
import ascend_ops

supported_dtypes = {torch.float16, torch.bfloat16, torch.float}
for data_type in supported_dtypes:
    print(f"DataType = <{data_type}>")
    x = torch.randn(1, 3, 8, 20, 45).to(data_type)
    print(f"Tensor x = {x}")
    size = (16, 40, 90)
    if data_type == torch.float :
        cpu_result = torch.nn.functional.interpolate(x, size=size, mode='nearest')
    else :
        cpu_result = torch.nn.functional.interpolate(x.float(), size=size, mode='nearest').to(data_type)
    print(f"cpu: upsamplenearest3d(x, size) = {cpu_result}")
    x_npu = x.npu()
    npu_result = torch.ops.ascend_ops.upsample_nearest3d(x_npu, size).cpu()
    print(f"[OK] torch.ops.ascend_ops.upsample_nearest3d<{data_type}> successfully!")
    print(f"npu: upsamplenearest3d(x, size) = {npu_result}")
    print(f"compare CPU Result vs NPU Result: {torch.allclose(cpu_result, npu_result)}\n\n")
