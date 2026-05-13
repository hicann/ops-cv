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
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

logging = Logger().get_logger()

try:
    import torch_npu
except ModuleNotFoundError as e:
    logging.warning("import torch_npu failed!!!")


@register("upsample_nearest3d_backward")
class FunctionTorchUpsampleNearest3dBackwardApi(FunctionApi):
    def __init__(self, task_result: TaskResult):
        self.task_result = task_result
        self.api_name = task_result.case_config.name
        self.output = task_result.case_config.outputs
        self.device = task_result.backend
        self.name = task_result.name
        self.device_id = task_result.device_id
        self.change_flag = False
        OpsDataset.seed_everything()
        self.forward_out = None
        self.outshape = None
        self.outdtype = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 标杆正向计算
        output = None
        input_tensor = input_data.kwargs['input'].requires_grad_(True)

        ##标量tensor和标量tensor场景
        if input_tensor.shape == torch.Size([]) or (0 in input_tensor.shape):  
            output = input_tensor
        else:
            if not with_output:
                eval(self.api_name)(*input_data.args, **input_data.kwargs)
                return output
            self.change_flag = False
            if self.device == "cpu":
                args = self.change_data_dtype(input_data.args, input_tensor.dtype, torch.float32)
                kwargs = self.change_data_dtype(input_data.kwargs, input_tensor.dtype, torch.float32)
            else:
                args = input_data.args
                kwargs = input_data.kwargs

            if self.output is None:
                output = eval(self.api_name)(*args, **kwargs)
            else:
                eval(self.api_name)(*args, **kwargs)
                if isinstance(self.output, int):
                    output = args[self.output]
                elif isinstance(self.output, str):
                    output = kwargs[self.output]
                else:
                    raise ValueError(f"self.output {self.output} value is " f"error")

            if self.change_flag:
                output = self.change_data_dtype(output, torch.float32, input_tensor.dtype)
        #使用gradOutput = torch.randn(output.shape, dtype=output.dtype)不能保证随机种子一致，先使用torch.ones测试验证
        gradoutput = torch.ones(output.shape, dtype=output.dtype)
        output.backward(gradoutput)
        backmarkgradoutput = input_tensor.grad
        return backmarkgradoutput

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            input_data.kwargs['intputSize_int'] = list(input_data.kwargs['input'].shape)
            self.outshape = list(input_data.kwargs['input'].shape)
            self.outdtype = input_data.kwargs['input'].dtype
            del input_data.kwargs['mode']
            del input_data.kwargs['antialias']
            del input_data.kwargs['format']
            del input_data.kwargs['recompute_scale_factor']
            if 'scale_factor' in input_data.kwargs and isinstance(input_data.kwargs['scale_factor'], tuple):
                input_data.kwargs['scalesD_double'] = input_data.kwargs['scale_factor'][0]
                input_data.kwargs['scalesH_double'] = input_data.kwargs['scale_factor'][1]
                input_data.kwargs['scalesW_double'] = input_data.kwargs['scale_factor'][2]
                input_data.kwargs['outputSize_int'][0] = int(
                    input_data.kwargs['input'].shape[2]*input_data.kwargs['scale_factor'][0])
                input_data.kwargs['outputSize_int'][1] = int(
                    input_data.kwargs['input'].shape[3]*input_data.kwargs['scale_factor'][1])
                input_data.kwargs['outputSize_int'][2] = int(
                    input_data.kwargs['input'].shape[4]*input_data.kwargs['scale_factor'][2])
                del input_data.kwargs['scale_factor']
            elif 'scale_factor' in input_data.kwargs and not isinstance(input_data.kwargs['scale_factor'], tuple):
                scale_factor_temp = input_data.kwargs['scale_factor']
                input_data.kwargs['scalesD_double'] = input_data.kwargs['scalesH_double'] = scale_factor_temp
                input_data.kwargs['scalesW_double'] = input_data.kwargs['scale_factor']
                input_data.kwargs['outputSize_int'][0] = int(input_data.kwargs['input'].shape[2]*scale_factor_temp)
                input_data.kwargs['outputSize_int'][1] = int(input_data.kwargs['input'].shape[3]*scale_factor_temp)
                input_data.kwargs['outputSize_int'][2] = int(input_data.kwargs['input'].shape[4]*scale_factor_temp)
                del input_data.kwargs['scale_factor']
            elif 'size' in input_data.kwargs and isinstance(input_data.kwargs['size'], tuple):
                input_data.kwargs['outputSize_int'][0] = input_data.kwargs['size'][0]
                input_data.kwargs['outputSize_int'][1] = input_data.kwargs['size'][1]
                input_data.kwargs['outputSize_int'][2] = input_data.kwargs['size'][2]
                input_data.kwargs['scalesD_double'] = input_data.kwargs['scalesH_double'] = 0
                input_data.kwargs['scalesW_double'] = 0
                del input_data.kwargs['size']
            elif 'size' in input_data.kwargs and not isinstance(input_data.kwargs['size'], tuple):
                input_data.kwargs['outputSize_int'][0] = input_data.kwargs['size'][0]
                input_data.kwargs['outputSize_int'][1] = input_data.kwargs['size'][1]
                input_data.kwargs['outputSize_int'][2] = input_data.kwargs['size'][2]
                input_data.kwargs['scalesD_double'] = input_data.kwargs['scalesH_double'] = 0
                input_data.kwargs['scalesW_double'] = 0
                del input_data.kwargs['size']
            self.outshape[-3:] = input_data.kwargs['outputSize_int']
            gradoutput = torch.ones(self.outshape, dtype=self.outdtype).npu()
            input_data.kwargs['input'] = gradoutput
        elif self.device in ["cpu", "gpu", "npu"]:
            del input_data.kwargs['scalesD_double']
            del input_data.kwargs['scalesH_double']
            del input_data.kwargs['scalesW_double']
            del input_data.kwargs['outputSize_int']
            del input_data.kwargs['intputSize_int']
            del input_data.kwargs['format']

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_NCDHW
