#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import numpy as np

def gen_golden(case_info: dict):
  input_desc = case_info["input_desc"]
  input_dtype = input_desc[0].get("dtype")
  self_desc = input_desc[0]["value"]
  output_size_desc = input_desc[1]["value"]
  align_coreners = input_desc[2]["value"]
  if align_coreners == 1:
      align_coreners = True
  else:
      align_coreners = False
  scales_h = input_desc[3]["value"]
  scales_w = input_desc[4]["value"]

  if input_dtype == "float16":
      self_tensor = torch.from_numpy(self_desc.astype(np.float32))
  else:
      self_tensor = torch.from_numpy(self_desc)

  output = torch._C._nn.upsample_bicubic2d(self_tensor, output_size_desc, align_coreners, scales_h, scales_w)
  if input_dtype == "float16":
      output = output.half()

  return output.numpy()