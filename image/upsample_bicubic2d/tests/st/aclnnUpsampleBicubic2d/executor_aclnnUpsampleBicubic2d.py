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

from abc import ABC

import torch

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.function_api import FunctionApi
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset
logging = Logger().get_logger()


@register("upsample_bicubic2d_aa")
class FunctionTorchUpsampleBicubic2dApi(FunctionApi):
    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            del input_data.kwargs['mode']
            del input_data.kwargs['antialias']
            del input_data.kwargs['recompute_scale_factor']
            del input_data.kwargs['format']
            if 'scale_factor' in input_data.kwargs and isinstance(input_data.kwargs['scale_factor'],tuple):
                input_data.kwargs['scalesH_double'] = input_data.kwargs['scale_factor'][0]
                input_data.kwargs['scalesW_double'] = input_data.kwargs['scale_factor'][1]
                #scale_factor场景下通过计算给outputsize赋值
                input_data.kwargs['outputSize_int'][0] = int(input_data.kwargs['input'].shape[2]*input_data.kwargs['scale_factor'][0])
                input_data.kwargs['outputSize_int'][1] = int(input_data.kwargs['input'].shape[3]*input_data.kwargs['scale_factor'][1])
                del input_data.kwargs['scale_factor']
            elif 'scale_factor' in  input_data.kwargs and not isinstance(input_data.kwargs['scale_factor'],tuple):
                input_data.kwargs['scalesH_double'] = input_data.kwargs['scale_factor']
                input_data.kwargs['scalesW_double'] = input_data.kwargs['scale_factor']
                # scale_factor场景下通过计算给outputsize赋值
                input_data.kwargs['outputSize_int'][0] = int(input_data.kwargs['input'].shape[2]*input_data.kwargs['scale_factor'])
                input_data.kwargs['outputSize_int'][1] = int(input_data.kwargs['input'].shape[3]*input_data.kwargs['scale_factor'])
                del input_data.kwargs['scale_factor']
            elif 'size' in input_data.kwargs and isinstance(input_data.kwargs['size'],tuple):
                input_data.kwargs['outputSize_int'][0] = input_data.kwargs['size'][0]
                input_data.kwargs['outputSize_int'][1] = input_data.kwargs['size'][1]
                #size场景下scalesW，scalesH设置为0
                input_data.kwargs['scalesH_double'] = 0
                input_data.kwargs['scalesW_double'] = 0
                del input_data.kwargs['size']
            elif 'size' in input_data.kwargs and not isinstance(input_data.kwargs['size'], tuple):
                input_data.kwargs['outputSize_int'][0] = input_data.kwargs['size'][0]
                input_data.kwargs['outputSize_int'][1] = input_data.kwargs['size'][1]
                #size场景下scalesW，scalesH设置为0
                input_data.kwargs['scalesH_double'] = 0
                input_data.kwargs['scalesW_double'] = 0
                del input_data.kwargs['size']
        elif self.device in ["cpu", "gpu", "npu"]:
            del input_data.kwargs['scalesH_double']
            del input_data.kwargs['scalesW_double']
            del input_data.kwargs['outputSize_int']
            del input_data.kwargs['format']

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        origin_dtype = input_data.kwargs["input"].dtype
        input = input_data.kwargs["input"]
        align_corners = input_data.kwargs["align_corners"]
        mode = input_data.kwargs["mode"]
        size = input_data.kwargs["size"]
        antialias = input_data.kwargs["antialias"]
        recompute_scale_factor = input_data.kwargs["recompute_scale_factor"]
        if self.device == 'cpu' and (input.dtype == torch.float16 or input.dtype == torch.bfloat16):
            output = torch.nn.functional.interpolate(input.to(torch.float32), size=size, mode=mode, align_corners=align_corners, 
                    antialias=antialias,recompute_scale_factor=recompute_scale_factor).to(origin_dtype)
        else:
            output = torch.nn.functional.interpolate(input, size=size, mode=mode, align_corners=align_corners, 
                    antialias=antialias,recompute_scale_factor=recompute_scale_factor)
        return output