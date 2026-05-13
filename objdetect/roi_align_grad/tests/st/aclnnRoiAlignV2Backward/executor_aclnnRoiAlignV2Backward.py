#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
=======
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
>>>>>>> 8b32882 (新增st)
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
<<<<<<< HEAD
# -----------------------------------------------------------------------------------------------------------
=======
# ----------------------------------------------------------------------------
>>>>>>> 8b32882 (新增st)

import torch
import random
import numpy as np
<<<<<<< HEAD
import torchvision
=======
>>>>>>> 8b32882 (新增st)
import ctypes
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
 
@register("function_aclnnRoiAlignV2Backward")
class AsceneRoiAlignV2BackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None
 
    def init_by_input_data(self, input_data: InputDataset):
        random.seed(self.task_result.case_config.id)
        ori_dtype = input_data.kwargs["boxes"].dtype
        if ori_dtype != torch.float16:
            input_data.kwargs["boxes"] = input_data.kwargs["boxes"].to(torch.float16)
        input_data.kwargs["spatialScale"] = random.uniform(0.1,1)
        for i in range(len(input_data.kwargs["boxes"])):
            input_data.kwargs["boxes"][i][0] = random.uniform(0, int(input_data.kwargs["inputShape"][0]-0.1))
            input_data.kwargs["boxes"][i][1] = random.uniform(0, input_data.kwargs["inputShape"][3] / input_data.kwargs["spatialScale"])
            input_data.kwargs["boxes"][i][2] = random.uniform(0, input_data.kwargs["inputShape"][2] / input_data.kwargs["spatialScale"])
            input_data.kwargs["boxes"][i][3] = random.uniform(float(input_data.kwargs["boxes"][i][1]+1), input_data.kwargs["inputShape"][3] / input_data.kwargs["spatialScale"])
            input_data.kwargs["boxes"][i][4] = random.uniform(float(input_data.kwargs["boxes"][i][2]+1), input_data.kwargs["inputShape"][2] / input_data.kwargs["spatialScale"])
        input_data.kwargs["boxes"] = input_data.kwargs["boxes"].to(ori_dtype)
 
    def __call__(self, input_data: InputDataset, with_output: bool = False):
 
 
        def roi_align_grad_compute(y_diff, rois, x_diff_shape, spatial_scale, sample_num, aligned):
            yN, C, pooled_height, pooled_width = y_diff.shape
            xN, _, xH, xW = x_diff_shape
            res = np.zeros(shape=(xN, C, xH, xW), dtype=np.float64)
            roi_end_mode = 2 if aligned == True else 0
 
            roi_batch_idx = rois[:, 0]
            roi_start_w = rois[:, 1] * spatial_scale
            roi_start_h = rois[:, 2] * spatial_scale
            roi_end_w = rois[:, 3] * spatial_scale
            roi_end_h = rois[:, 4] * spatial_scale
            if roi_end_mode > 0:
                if roi_end_mode == 1:
                    roi_end_w, roi_end_h = roi_end_w + spatial_scale, roi_end_h + spatial_scale
                else:
                    roi_start_w, roi_start_h = roi_start_w - 0.5, roi_start_h - 0.5
                    roi_end_w, roi_end_h = roi_end_w - 0.5, roi_end_h - 0.5
 
            roi_width = roi_end_w - roi_start_w
            roi_height = roi_end_h - roi_start_h
            if roi_end_mode < 2:
                roi_width = np.maximum(roi_width, 1)
                roi_height = np.maximum(roi_height, 1)
 
            bin_width = roi_width / pooled_width
            bin_height = roi_height / pooled_height
            if sample_num > 0:
                roi_grid_w = np.ones(bin_width.shape, dtype="int32")
                roi_grid_h = np.ones(bin_height.shape, dtype="int32")
                roi_grid_w = roi_grid_w * sample_num
                roi_grid_h = roi_grid_h * sample_num
            else:
                roi_grid_w = np.ceil(bin_width).astype("int32")
                roi_grid_h = np.ceil(bin_height).astype("int32")
            counts = roi_grid_h * roi_grid_w
 
            for n in range(yN):
                start_h, start_w = roi_start_h[n], roi_start_w[n]
                grid_h, grid_w = roi_grid_h[n], roi_grid_w[n]
                size_h, size_w = bin_height[n], bin_width[n]
                fm_batch_idx = int(roi_batch_idx[n])
                cnt = counts[n]
                if fm_batch_idx < 0 or fm_batch_idx >= xN:
                    continue
                for ph in range(pooled_height):
                    for pw in range(pooled_width):
                        grad_bin = y_diff[n, :, ph, pw]
                        grad_bin = grad_bin / cnt
 
                        for gh in range(grid_h):
                            y = start_h + ph * size_h + (gh + 0.5) * size_h / grid_h
                            for gw in range(grid_w):
                                x = start_w + pw * size_w + (gw + 0.5) * size_w / grid_w
<<<<<<< HEAD
                                # w1, w2, w3, w4, x_low, x_high, y_low, y_high = bilinear_interpolate_gradient(xH, xW,
                                #                                                                              y, x)
=======
>>>>>>> 8b32882 (新增st)
                                w1, w2, w3, w4, x_low, x_high, y_low, y_high = bilinear_interpolate_gradient(xH, xW,
                                                                                                    y, x, y_diff.dtype.name)
                                g1, g2, g3, g4 = grad_bin * w1, grad_bin * w2, grad_bin * w3, grad_bin * w4
                                if x_low >= 0 and x_high >= 0 and y_low >= 0 and y_high >= 0:
                                    res[fm_batch_idx, :, y_low, x_low] += g1
                                    res[fm_batch_idx, :, y_low, x_high] += g2
                                    res[fm_batch_idx, :, y_high, x_low] += g3
                                    res[fm_batch_idx, :, y_high, x_high] += g4
            return torch.from_numpy(res.astype(y_diff.dtype, copy=False))
 
<<<<<<< HEAD
        def bilinear_interpolate_gradient_ori(height, width, y, x):
            # deal with cases that inverse elements are out of feature map boundary
            if y < -1.0 or y > height or x < -1.0 or x > width:
                w1 = w2 = w3 = w4 = np.float32(0.)
                x_low = x_high = y_low = y_high = -1
                return w1, w2, w3, w4, x_low, x_high, y_low, y_high
 
            if y <= 0:
                y = np.float32(0)
            if x <= 0:
                x = np.float32(0)
 
            x_low = int(x)
            y_low = int(y)
 
            if y_low >= height - 1:
                y_high = y_low = height - 1
                y = np.float32(y_low)
            else:
                y_high = y_low + 1
 
            if x_low >= width - 1:
                x_high = x_low = width - 1
                x = np.float32(x_low)
            else:
                x_high = x_low + 1
 
            ly = y - np.float32(y_low)
            lx = x - np.float32(x_low)
            hy = np.float32(1.) - ly
            hx = np.float32(1.) - lx
 
            w1 = hy * hx
            w2 = hy * lx
            w3 = ly * hx
            w4 = ly * lx
 
            return w1, w2, w3, w4, x_low, x_high, y_low, y_high
 
=======
>>>>>>> 8b32882 (新增st)
        def bilinear_interpolate_gradient(height, width, y, x, dtype):
            # deal with cases that inverse elements are out of feature map boundary
            if y < -1.0 or y > height or x < -1.0 or x > width:
                w1 = w2 = w3 = w4 = getattr(np, dtype)(0.)
                x_low = x_high = y_low = y_high = -1
                return w1, w2, w3, w4, x_low, x_high, y_low, y_high
 
            if y <= 0:
                y = getattr(np, dtype)(0)
            if x <= 0:
                x = getattr(np, dtype)(0)
 
            x_low = int(x)
            y_low = int(y)
 
            if y_low >= height - 1:
                y_high = y_low = height - 1
                y = getattr(np, dtype)(y_low)
            else:
                y_high = y_low + 1
 
            if x_low >= width - 1:
                x_high = x_low = width - 1
                x = getattr(np, dtype)(x_low)
            else:
                x_high = x_low + 1
 
            ly = y - getattr(np, dtype)(y_low)
            lx = x - getattr(np, dtype)(x_low)
            hy = getattr(np, dtype)(1.) - ly
            hx = getattr(np, dtype)(1.) - lx
 
            w1 = hy * hx
            w2 = hy * lx
            w3 = ly * hx
            w4 = ly * lx
 
            return w1, w2, w3, w4, x_low, x_high, y_low, y_high
 
<<<<<<< HEAD
 
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
            output = torch.ops.torchvision._roi_align_backward(input_data.kwargs["gradOutput"].to(device), input_data.kwargs["boxes"].to(device),
                                                                spatial_scale=input_data.kwargs["spatialScale"], pooled_height=input_data.kwargs["pooledHeight"], 
                                                                pooled_width=input_data.kwargs["pooledWidth"], batch_size=input_data.kwargs["inputShape"][0], 
                                                                channels=input_data.kwargs["inputShape"][1], height=input_data.kwargs["inputShape"][2], 
                                                                width=input_data.kwargs["inputShape"][3], sampling_ratio=input_data.kwargs["samplingRatio"], 
                                                                aligned=input_data.kwargs["aligned"])
            return output
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
            gradInput = roi_align_grad_compute(input_data.kwargs["gradOutput"].numpy(), input_data.kwargs["boxes"].numpy(),
                                               input_data.kwargs["inputShape"], input_data.kwargs["spatialScale"],
                                               input_data.kwargs["samplingRatio"], input_data.kwargs["aligned"])
            return gradInput
=======
        gradInput = roi_align_grad_compute(input_data.kwargs["gradOutput"].numpy(), input_data.kwargs["boxes"].numpy(),
                                            input_data.kwargs["inputShape"], input_data.kwargs["spatialScale"],
                                            input_data.kwargs["samplingRatio"], input_data.kwargs["aligned"])
        return gradInput
>>>>>>> 8b32882 (新增st)

@register("aclnn_function")
class aclnnfunctionExecutor(AclnnBaseApi):
        
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages= super().init_by_input_data(input_data)
        input_args[5] = ctypes.c_float(input_args[5].value)
        return input_args, output_packages       
 
    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name == "gradOutput": # self为json中的name字段，此时处理self输入的format
            return AclFormat.ACL_FORMAT_NCHW
        if index == 0: # 0为输出或输出的位置，此时处理第0位置的出参
            return AclFormat.ACL_FORMAT_NCHW
        return AclFormat.ACL_FORMAT_ND
