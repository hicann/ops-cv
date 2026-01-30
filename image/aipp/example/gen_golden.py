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
from dataclasses import dataclass
from typing import List
import re
import torch
import numpy as np


@dataclass
class AippConfig:
    start_offset_h: int
    start_offset_w: int
    output_size_h: int
    output_size_w: int
    mean_chn: List[float]
    min_chn: List[float]
    var_reci_chn: List[float]


def aipp_golden(input_tensor: torch.Tensor, config: AippConfig) -> torch.Tensor:
    cropped = input_tensor[:, 
                           config.start_offset_h:config.start_offset_h + config.output_size_h,
                           config.start_offset_w:config.start_offset_w + config.output_size_w, :]
    cropped_float32 = cropped.half()
    channel_num = cropped_float32.size(-1)
    for c in range(channel_num):
        cropped_float32[:, :, :, c] = (cropped_float32[:, :, :, c] - config.mean_chn[c] -
            config.min_chn[c]) * config.var_reci_chn[c]
    output_tensor = cropped_float32.half()
    return output_tensor


def load_aipp_params(config_file):
    with open(config_file, 'r') as f:
        text = f.read()
    params = {}
    pattern = r'(\w+)\s*:\s*([^;\n]+)'
    for key, value in re.findall(pattern, text):
        value = value.strip()
        if value.replace('.', '').isdigit():
            if '.' in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        elif value in ['true', 'false']:
            params[key] = value == 'true'
        else:
            params[key] = value
    
    return params


if __name__ == "__main__":
    config_file = "aipp.cfg"
    config = load_aipp_params(config_file)
    input_height = config['src_image_size_h']
    input_width = config['src_image_size_w']
    input_tensor = torch.randint(0, 256, (2, input_height, input_width, 3), dtype=torch.uint8)
    start_offset_h = config['load_start_pos_h']
    start_offset_w = config['load_start_pos_w']
    output_size_h = config['crop_size_h']
    output_size_w = config['crop_size_w']
    min_chn = [config['min_chn_0'], config['min_chn_1'], config['min_chn_2']]
    mean_chn = [0.0, 0.0, 0.0]
    var_reci_chn = [config['var_reci_chn_0'], config['var_reci_chn_1'], config['var_reci_chn_2']]
    aipp_config = AippConfig(
        start_offset_h=start_offset_h,
        start_offset_w=start_offset_w,
        output_size_h=output_size_h,
        output_size_w=output_size_w,
        mean_chn=mean_chn,
        min_chn=min_chn,
        var_reci_chn=var_reci_chn
    )
    output_tensor = aipp_golden(input_tensor, aipp_config)
    input_np = input_tensor.cpu().numpy().astype(np.uint8)
    input_np.tofile("input_tensor.bin")
    output_np = output_tensor.cpu().numpy().astype(np.float16)
    output_np.tofile("output_tensor.bin")
