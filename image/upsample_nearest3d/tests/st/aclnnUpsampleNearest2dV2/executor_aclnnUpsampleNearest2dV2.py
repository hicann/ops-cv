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
import math

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.function_api import FunctionApi
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

logging = Logger().get_logger()

try:
    import torch_npu
except Exception as e:
    logging.warning("import torch_npu failed!!!")

@register("DFXUpsampleNearest2dV2")
class FunctionDFXUpsampleNearest2dV2Api(FunctionApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.format= None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 标杆正向计算
        output = None
        input_tensor = input_data.kwargs['input']
        dtype_list = [torch.float32, torch.bfloat16, torch.float16, torch.float64, torch.uint8 ]

        ## 标量tensor和标量tensor场景
        if (
            input_tensor.shape == torch.Size([]) or 
            (0 in input_tensor.shape) or
        #dtype 限制
            input_tensor.dtype not in dtype_list or 
        #shape 不合规
            len(input_tensor.shape) != 4
        # 空指针
            # or input_data.kwargs['nullptr_mode'] 
            ):  
            output = input_tensor
            print(" skip ")
        #elif 2147483649 in  input_data.kwargs['input'].shape :
        #    output = torch.zeros(out_shape)
        else:
            # del input_data.kwargs['nullptr_mode']
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

        if self.format == "NHWC" and len(output.shape)==4:
            output = output.permute(0, 2, 3, 1)
        return output

    def init_by_input_data(self, input_data: InputDataset):
        if self.device in [ "pyaclnn"]:
            input_shape = input_data.kwargs['input'].shape
            if 2147483649 in input_shape:
                input_data.kwargs['scalesH'] = 1.0
                input_data.kwargs['scalesW'] = 1.0
            if len(input_shape) == 4 :
                scalesH = input_data.kwargs['scalesH']
                scalesW = input_data.kwargs['scalesW']

                if input_data.kwargs["format"] == "NCHW":
                    out_shape = [math.floor(scalesH* input_shape[2]),math.floor(scalesW* input_shape[3])]
                elif input_data.kwargs["format"] == "NHWC":
                    out_shape = [math.floor(scalesH* input_shape[1]),math.floor(scalesW* input_shape[2])]
                else :
                    out_shape = [8,8]
            else:
                out_shape = [8,8]
            input_data.kwargs['size'] = out_shape
        if  self.device in ["npu","cpu","gpu" ]:
            scalesH = input_data.kwargs['scalesH']
            scalesW = input_data.kwargs['scalesW']
            if 2147483649 in input_data.kwargs['input'].shape:
                scalesH = 1.0
                scalesW = 1.0
            del input_data.kwargs['scalesH']
            del input_data.kwargs['scalesW']
            del input_data.kwargs['size']
            input_data.kwargs['scale_factor'] = [scalesH,scalesW]
            self.format = input_data.kwargs["format"]
            if self.format == "NHWC" and len(input_data.kwargs['input'].shape)==4:
                input_data.kwargs['input'] = input_data.kwargs['input'].permute(0, 3, 1, 2)

@register("aclnnDFXUpsampleNearest2dV2")
class FunctionAclnnDFXUpsampleNearest2dV2Api(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
            del input_data.kwargs['mode']
            del input_data.kwargs['scale_factor']
            input_args, output_packages = super().init_by_input_data(input_data)
            return input_args, output_packages

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