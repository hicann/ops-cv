#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import numpy as np

def gen_golden(case_info: dict):
    input_desc = case_info["input_desc"]
    
    self_tensor = torch.from_numpy(input_desc[0]["value"])
    output_size_desc = input_desc[1]["value"]
    align_corners = bool(input_desc[2]["value"])
    scales_h = input_desc[3]["value"]
  
    output = torch._C._nn.upsample_bilinear2d(self_tensor, output_size_desc, align_corners, scales_h, scales_h)
  
    return output.numpy()