#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import random
import numpy as np
import torchvision
import ctypes
import copy
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

@register("ascend_method_torch_roi_align")
class MethodTorchRoiAlignApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchRoiAlignApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.spatialScale = None
        self.boxes_ori = None
        self.boxes = None

    def init_by_input_data(self, input_data: InputDataset):
        random.seed(self.task_result.case_config.id)
        ori_dtype = input_data.kwargs["rois"].dtype
        ori_dtype_int = input_data.kwargs["batch_indices"].dtype
        if ori_dtype != torch.float16:
            input_data.kwargs["rois"] = input_data.kwargs["rois"].to(torch.float16)
        input_data.kwargs["spatialScale"] = random.uniform(0.1,1)
        for i in range(len(input_data.kwargs["rois"])):
            input_data.kwargs["batch_indices"][i] = int(random.uniform(0, int(input_data.kwargs["x"].shape[0]-0.1)))
            input_data.kwargs["rois"][i][0] = random.uniform(0, input_data.kwargs["x"].shape[3] / input_data.kwargs["spatialScale"])
            input_data.kwargs["rois"][i][1] = random.uniform(0, input_data.kwargs["x"].shape[2] / input_data.kwargs["spatialScale"])
            input_data.kwargs["rois"][i][2] = random.uniform(float(input_data.kwargs["rois"][i][0]+1), input_data.kwargs["x"].shape[3] / input_data.kwargs["spatialScale"])
            input_data.kwargs["rois"][i][3] = random.uniform(float(input_data.kwargs["rois"][i][1]+1), input_data.kwargs["x"].shape[2] / input_data.kwargs["spatialScale"])
        input_data.kwargs["rois"] = input_data.kwargs["rois"].to(ori_dtype)
        input_data.kwargs["batch_indices"] = input_data.kwargs["batch_indices"].to(ori_dtype_int)


    def __call__(self, input_data: InputDataset, with_output: bool = False):

        def bilinear_interpolate(bottom_data, width, height, c, x, y, cnt):
            val_point = np.zeros((c), bottom_data.dtype)
            if y < -1.0 or y > height or x < -1.0 or x > width:
                return val_point

            if y <= 0:
                y = 0
            if x <= 0:
                x = 0
            y_low = int(y)
            x_low = int(x)
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

            x_low_fp = np.float32(x_low)
            y_low_fp = np.float32(y_low)

            lx = x - x_low_fp
            ly = y - y_low_fp
            hx = np.float32(1.0) - lx
            hy = np.float32(1.0) - ly
            lt = bottom_data[:, y_low, x_low]
            rt = bottom_data[:, y_low, x_high]
            lb = bottom_data[:, y_high, x_low]
            rb = bottom_data[:, y_high, x_high]
            w1 = hx * hy
            w2 = hy * lx
            w3 = ly * hx
            w4 = lx * ly

            if bottom_data.dtype.name == 'float16':
                w1 = np.float16(w1)
                w2 = np.float16(w2)
                w3 = np.float16(w3)
                w4 = np.float16(w4)

            # average here to avoid float overflow when summed
            val_point = (w1 * lt + w2 * rt + w3 * lb + w4 * rb) / cnt
            return val_point

        def bilinear_interpolate_keep_precision(bottom_data, width, height, c, x, y, cnt):
            val_point = np.zeros((c), bottom_data.dtype)
            if y < -1.0 or y > height or x < -1.0 or x > width:
                return val_point

            if y <= 0:
                y = 0
            if x <= 0:
                x = 0
            y_low = int(y)
            x_low = int(x)
            if y_low >= height - 1:
                y_high = y_low = height - 1
                y = getattr(np, bottom_data.dtype.name)(y_low)
            else:
                y_high = y_low + 1
            if x_low >= width - 1:
                x_high = x_low = width - 1
                x = getattr(np, bottom_data.dtype.name)(x_low)
            else:
                x_high = x_low + 1

            x_low_fp = getattr(np, bottom_data.dtype.name)(x_low)
            y_low_fp = getattr(np, bottom_data.dtype.name)(y_low)

            lx = x - x_low_fp
            ly = y - y_low_fp
            hx = getattr(np, bottom_data.dtype.name)(1.0) - lx
            hy = getattr(np, bottom_data.dtype.name)(1.0) - ly
            lt = bottom_data[:, y_low, x_low]
            rt = bottom_data[:, y_low, x_high]
            lb = bottom_data[:, y_high, x_low]
            rb = bottom_data[:, y_high, x_high]
            w1 = hx * hy
            w2 = hy * lx
            w3 = ly * hx
            w4 = lx * ly

            if bottom_data.dtype.name == 'float16':
                w1 = np.float16(w1)
                w2 = np.float16(w2)
                w3 = np.float16(w3)
                w4 = np.float16(w4)

            val_point = (w1 * lt + w2 * rt + w3 * lb + w4 * rb) / cnt
            return val_point

        def roi_align_golden(feature_map, batchIndices, rois, pooled_height, pooled_width, spatial_scale, 
            sample_num, pool_mode, roi_end_mode):
            feature_map = feature_map.numpy()

            batchIndices_ori = copy.deepcopy(batchIndices).to(rois.dtype).numpy()
            batchIndices_ori_expand = batchIndices_ori.reshape(-1, 1)
            rois_ori = copy.deepcopy(rois).numpy()
            self.boxes_ori = np.concatenate([batchIndices_ori_expand, rois_ori], axis=1)

            rois = rois.numpy().astype("float32", copy=True)
            batchIndices = batchIndices.numpy().astype("float32", copy=True)
            batchIndices_expand = batchIndices.reshape(-1, 1)
            self.boxes = np.concatenate([batchIndices_expand, rois], axis=1)

            fm_n, C, fm_h, fm_w = feature_map.shape
            y_n = self.boxes.shape[0]
            res = np.zeros(shape=(y_n, C, pooled_height, pooled_width), dtype=feature_map.dtype)

            if roi_end_mode is None:
                roi_end_mode = 0

            roi_batch_idx = self.boxes[:, 0]
            roi_start_w = self.boxes[:, 1] * spatial_scale
            roi_start_h = self.boxes[:, 2] * spatial_scale
            roi_end_w = self.boxes[:, 3] * spatial_scale
            roi_end_h = self.boxes[:, 4] * spatial_scale
            if roi_end_mode == 1:
                roi_end_w, roi_end_h = roi_end_w + spatial_scale, roi_end_h + spatial_scale
            elif roi_end_mode > 1:
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

            for n in range(y_n):
                start_h, start_w = roi_start_h[n], roi_start_w[n]
                grid_h, grid_w = roi_grid_h[n], roi_grid_w[n]
                size_h, size_w = bin_height[n], bin_width[n]
                fm_batch_idx = int(roi_batch_idx[n])
                if fm_batch_idx < 0 or fm_batch_idx >= fm_n:
                    continue

                bottom_data = feature_map[fm_batch_idx, :, :, :]
                cnt = counts[n] if pool_mode == 'avg' else 1
                for ph in range(pooled_height):
                    for pw in range(pooled_width):
                        if pool_mode == 'avg':
                            val = np.zeros(shape=(C), dtype=feature_map.dtype)
                        else:
                            min_v = -3.4028235e+38 if feature_map.dtype == "float32" else -65504
                            val = np.ones(shape=(C), dtype=feature_map.dtype) * min_v
                        for gh in range(grid_h):
                            y = start_h + ph * size_h + (gh + 0.5) * size_h / grid_h
                            for gw in range(grid_w):
                                x = start_w + pw * size_w + (gw + 0.5) * size_w / grid_w
                                # vg = bilinear_interpolate(bottom_data, fm_w, fm_h, C, x, y, cnt)
                                vg = bilinear_interpolate_keep_precision(bottom_data, fm_w, fm_h, C, x, y, cnt)
                                if pool_mode == 'avg':
                                    val += vg
                                else:
                                    val = np.maximum(val, vg)
                        res[n, :, ph, pw] = val
            return torch.from_numpy(res)

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
            output_size = [input_data.kwargs["outputHeight"], input_data.kwargs["outputWidth"]]
            RoIAlign = torchvision.ops.RoIAlign(output_size=output_size, sampling_ratio=input_data.kwargs["samplingRatio"],
                                                spatial_scale=input_data.kwargs["spatialScale"], aligned=false)
            output = RoIAlign(input_data.kwargs["x"].to(device), torch.from_numpy(self.boxes_ori).to(device))
            return output
        elif self.device == "npu" and self.device_id == 0:
            import torch_npu
            torch.npu.set_compile_mode(jit_compile=True)
            device = f"{self.device}:{self.device_id}"
            roi_end_mode = 0
            res = torch_npu.npu_roi_align(input=input_data.kwargs["x"], rois=torch.from_numpy(self.boxes_ori),
                                          spatial_scale=input_data.kwargs["spatialScale"],
                                          pooled_height=input_data.kwargs["outputHeight"],
                                          pooled_width=input_data.kwargs["outputWidth"],
                                          sample_num=input_data.kwargs["samplingRatio"],
                                          roi_end_mode=roi_end_mode)
            return res
        else:
            device = "cpu"
            roi_end_mode = 0
            output = roi_align_golden(input_data.kwargs["x"], input_data.kwargs["batch_indices"], input_data.kwargs["rois"], 
                                      input_data.kwargs["outputHeight"], input_data.kwargs["outputWidth"], input_data.kwargs["spatialScale"],
                                      input_data.kwargs["samplingRatio"], input_data.kwargs["mode"], roi_end_mode)
            return output

@register("aclnn_roi_align")
class RoiAlignAclnnApi(AclnnBaseApi):  

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name == "x": # x为json中的name字段，此时处理x输入的format
            return AclFormat.ACL_FORMAT_NCHW
        if index == 0: # 0为输出或输出的位置，此时处理第0位置的出参
            return AclFormat.ACL_FORMAT_NCHW
        return AclFormat.ACL_FORMAT_ND
