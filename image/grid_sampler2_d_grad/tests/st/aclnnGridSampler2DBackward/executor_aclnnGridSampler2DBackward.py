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
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

@register("ascend_method_torch_grid_sampler_2d_backward")
class FunctionGridSampler2DBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionGridSampler2DBackwardApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradOutput = input_data.kwargs["gradOutput"]
        input = input_data.kwargs["input"]
        grid = input_data.kwargs["grid"]
        origin_dtype = input_data.kwargs["input"].dtype

        interpolation_mode = input_data.kwargs["interpolationMode"]
        padding_mode = input_data.kwargs["paddingMode"]
        alignCorners = input_data.kwargs["alignCorners"]
        outputMask = input_data.kwargs["outputMask"]
        if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
            output = torch.ops.aten.grid_sampler_2d_backward(gradOutput.to(torch.float32), input.to(torch.float32), grid.to(torch.float32), interpolation_mode, padding_mode, alignCorners, outputMask)
            output = (output[0].to(origin_dtype),output[1].to(origin_dtype))
        else:
            output = torch.ops.aten.grid_sampler_2d_backward(gradOutput, input, grid, interpolation_mode, padding_mode, alignCorners, outputMask)
        return output
