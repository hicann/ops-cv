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

@register("aclnn_gridsampler3dgrad")
class FunctionTorchGridSampler3DApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionTorchGridSampler3DApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        origin_dtype = input_data.kwargs["input"].dtype

        gradOutput = input_data.kwargs["gradOutput"]
        input = input_data.kwargs["input"]
        grid = input_data.kwargs["grid"]

        interpolation_mode = input_data.kwargs["mode"]
        paddingMode = input_data.kwargs["padding_mode"]
        alignCorners = input_data.kwargs["align_corners"]
        outputMask = input_data.kwargs["outputMask"]

        # 升精度到fp32
        if gradOutput.dtype == torch.float16 or gradOutput.dtype == torch.bfloat16:
            res = torch.ops.aten.grid_sampler_3d_backward(gradOutput.to(torch.float32), input.to(torch.float32), grid.to(torch.float32),
                                                             interpolation_mode, paddingMode, alignCorners, outputMask)
            output = tuple(tensor.to(origin_dtype) for tensor in res)
        else:
            output = torch.ops.aten.grid_sampler_3d_backward(gradOutput, input, grid, interpolation_mode, paddingMode, alignCorners, outputMask)

        return output

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if input_data.kwargs["format"] == "NCDHW":
            return AclFormat.ACL_FORMAT_NCDHW
        return AclFormat.ACL_FORMAT_NDHWC
