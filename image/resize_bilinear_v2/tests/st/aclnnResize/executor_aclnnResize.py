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

import logging
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.function_api import FunctionApi

from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@register("function_Resize")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.format = None

    def init_by_input_data(self, input_data: InputDataset):
        self.format = input_data.kwargs["format"]

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs["x"]
        input_scales = input_data.kwargs["scales"]
        input_mode = input_data.kwargs["mode"]
        input_format = self.format

        if input_format == "NCHW":
            input_scales[0] = 1.0
            input_scales[1] = 1.0
        elif input_format == "NHWC":
            input_scales[0] = 1.0
            input_scales[3] = 1.0

        if input_x.dtype == torch.float16:
            input_x = input_x.to(torch.float32)
            output_tensor = torch.nn.functional.interpolate(input_x, scale_factor=input_scales[2:],
                                                            mode=input_mode).to(torch.float16)
        else:
            output_tensor = torch.nn.functional.interpolate(input_x, scale_factor=input_scales[2:], mode=input_mode)
        return output_tensor

    def get_format(self, input_data: InputDataset, index=None, name=None):
        """
        :param input_data: 参数列表
        :param index: 参数位置
        :param name: 参数名字
        :return:
        format at this index or name
        """
        if input_data.kwargs["format"] == "NCHW":
            return AclFormat.ACL_FORMAT_NCHW
        if input_data.kwargs["format"] == "NHWC":
            return AclFormat.ACL_FORMAT_NHWC
        return AclFormat.ACL_FORMAT_ND
