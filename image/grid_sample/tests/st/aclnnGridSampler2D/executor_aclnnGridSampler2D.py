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
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@register("aclnn_gridsampler2d")
class FunctionTorchGridSampler2DApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionTorchGridSampler2DApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        origin_dtype = input_data.kwargs["input"].dtype

        input = input_data.kwargs["input"]
        grid = input_data.kwargs["grid"]

        if input_data.kwargs["mode"] == 0:
            interpolation_mode = "bilinear"
        elif input_data.kwargs["mode"] == 1:
            interpolation_mode = "nearest"

        if input_data.kwargs["padding_mode"] == 0:
            paddingMode = "zeros"
        elif input_data.kwargs["padding_mode"] == 1:
            paddingMode = "border"
        elif input_data.kwargs["padding_mode"] == 2:
            paddingMode = "reflection"

        alignCorners = input_data.kwargs["align_corners"]

        if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
            output = torch.nn.functional.grid_sample(input.to(torch.float32), grid.to(torch.float32),
                                                     interpolation_mode, paddingMode, alignCorners).to(origin_dtype)
        else:
            output = torch.nn.functional.grid_sample(input, grid, interpolation_mode, paddingMode, alignCorners)

        return output

        