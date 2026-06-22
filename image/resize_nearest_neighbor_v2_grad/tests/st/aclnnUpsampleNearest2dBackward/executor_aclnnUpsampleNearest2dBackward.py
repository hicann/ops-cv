#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
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
except Exception as e:
    logging.warning("import torch_npu failed!!!")


@register("upsample_nearest_2d_grad")
class FunctionTorchUpsampleNearest2dGradApi(FunctionApi):
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
        origin_dtype = input_data.kwargs["input"].dtype
        input_tensor = input_data.kwargs["input"].requires_grad_(True)

        if input_tensor.shape == torch.Size([]) or (0 in input_tensor.shape):
            output = input_tensor
        else:
            if not with_output:
                eval(self.api_name)(*input_data.args, **input_data.kwargs)
                return None
            self.change_flag = False

            if self.device == "cpu" and (
                origin_dtype == torch.float16 or origin_dtype == torch.bfloat16
            ):
                x = input_tensor.to(torch.float32)
                size = input_data.kwargs.get("size", None)
                scale_factor = input_data.kwargs.get("scale_factor", None)
                if size is not None:
                    output = torch.nn.functional.interpolate(
                        x, size=size, mode="nearest"
                    ).to(origin_dtype)
                elif scale_factor is not None:
                    output = torch.nn.functional.interpolate(
                        x, scale_factor=scale_factor, mode="nearest"
                    ).to(origin_dtype)
                else:
                    output = torch.nn.functional.interpolate(x, mode="nearest").to(
                        origin_dtype
                    )
            else:
                size = input_data.kwargs.get("size", None)
                scale_factor = input_data.kwargs.get("scale_factor", None)
                if size is not None:
                    output = torch.nn.functional.interpolate(
                        input_tensor, size=size, mode="nearest"
                    )
                elif scale_factor is not None:
                    output = torch.nn.functional.interpolate(
                        input_tensor, scale_factor=scale_factor, mode="nearest"
                    )
                else:
                    output = torch.nn.functional.interpolate(
                        input_tensor, mode="nearest"
                    )

            gradoutput = torch.ones_like(output).to(output.dtype)
            output.backward(gradoutput)
            backmarkgradoutput = input_tensor.grad
            return backmarkgradoutput

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            input_data.kwargs["intputSize_int"] = list(input_data.kwargs["input"].shape)
            self.outshape = list(input_data.kwargs["input"].shape)
            self.outdtype = input_data.kwargs["input"].dtype

            del input_data.kwargs["mode"]
            del input_data.kwargs["format"]

            if "size" in input_data.kwargs and isinstance(
                input_data.kwargs["size"], tuple
            ):
                input_data.kwargs["outputSize_int"][0] = input_data.kwargs["size"][0]
                input_data.kwargs["outputSize_int"][1] = input_data.kwargs["size"][1]
                del input_data.kwargs["size"]
            elif "size" in input_data.kwargs and not isinstance(
                input_data.kwargs["size"], tuple
            ):
                input_data.kwargs["outputSize_int"][0] = input_data.kwargs["size"][0]
                input_data.kwargs["outputSize_int"][1] = input_data.kwargs["size"][1]
                del input_data.kwargs["size"]

            if "scale_factor" in input_data.kwargs and isinstance(
                input_data.kwargs["scale_factor"], tuple
            ):
                input_data.kwargs["scalesH_double"] = input_data.kwargs["scale_factor"][
                    0
                ]
                input_data.kwargs["scalesW_double"] = input_data.kwargs["scale_factor"][
                    1
                ]
                del input_data.kwargs["scale_factor"]
            elif "scale_factor" in input_data.kwargs and not isinstance(
                input_data.kwargs["scale_factor"], tuple
            ):
                input_data.kwargs["scalesH_double"] = input_data.kwargs["scale_factor"]
                input_data.kwargs["scalesW_double"] = input_data.kwargs["scale_factor"]
                del input_data.kwargs["scale_factor"]

            self.outshape[-2:] = input_data.kwargs["outputSize_int"]
            gradoutput = torch.ones(self.outshape, dtype=self.outdtype).npu()
            input_data.kwargs["input"] = gradoutput
        else:
            del input_data.kwargs["outputSize_int"]
            del input_data.kwargs["intputSize_int"]
            if "scalesH_double" in input_data.kwargs:
                del input_data.kwargs["scalesH_double"]
            if "scalesW_double" in input_data.kwargs:
                del input_data.kwargs["scalesW_double"]
            del input_data.kwargs["format"]

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_NCHW
