#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import math
import numpy as np
import torch

def generate_rois(roi_shape, feature_shape, spatial_scale):
    num_boxes = roi_shape[0]
    rois = torch.Tensor(6, num_boxes)
    rois[0] = torch.randint(0, feature_shape[0], (num_boxes,))
    rois[1].uniform_(0, feature_shape[2]) / spatial_scale
    rois[2].uniform_(0, feature_shape[3]) / spatial_scale
    rois[3].uniform_(0, feature_shape[2]) / spatial_scale
    rois[4].uniform_(0, feature_shape[3]) / spatial_scale
    rois[5].uniform_(0, math.pi)

    return rois.transpose(0, 1).contiguous()

def generate_grad_output(grad_output_shape):
    grad_output = torch.rand(grad_output_shape)
    return grad_output

def gen_golden_data_simple(batch_size, channel, h, w, rois_num, roi_length):
    rois = generate_rois([rois_num, roi_length], [batch_size, channel, h, w], 0.5)
    grad_output = generate_grad_output([rois_num, channel, h, w])

    rois.tofile("./rois.bin")
    grad_output.tofile("./grad_output.bin")

if __name__ == "__main__":
    gen_golden_data_simple(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))