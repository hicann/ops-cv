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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
logging = Logger().get_logger()


@register("upsample_nearest_1d")
class FunctionTorchUpsampleNearest1dApi(FunctionApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        origin_dtype = input_data.kwargs["input"].dtype
        x_in = input_data.kwargs["input"]
        size = input_data.kwargs["size"]
        if self.device == 'cpu' and (x_in.dtype == torch.float16 or x_in.dtype == torch.bfloat16):
            output = torch.nn.functional.interpolate(x_in.to(torch.float32), size=size, mode='nearest').to(origin_dtype)
        else:
            output = torch.nn.functional.interpolate(x_in, size=size, mode='nearest')
        return output

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            del input_data.kwargs['format']
            if 'size' in input_data.kwargs and isinstance(input_data.kwargs['size'], tuple):
                input_data.kwargs['outputSize_int'][0] = input_data.kwargs['size'][0]
                del input_data.kwargs['size']
            elif 'size' in input_data.kwargs and not isinstance(input_data.kwargs['size'], tuple):
                input_data.kwargs['outputSize_int'][0] = input_data.kwargs['size']
                del input_data.kwargs['size']
        else:
            del input_data.kwargs['outputSize_int']
            del input_data.kwargs['format']

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_NCL