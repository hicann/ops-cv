#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

import sys
import math
import numpy as np
import torch
def generate_features(feature_shape):
    features = torch.rand(feature_shape)
    return features

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
def gen_golden_data_simple(batch_size, channel, h, w, rois_num, roi_length):
    features = generate_features([batch_size, h, w, channel])
    rois = generate_rois([rois_num, roi_length], [batch_size, channel, h, w], 0.5)

    features.tofile("./input.bin")
    rois.tofile("./rois.bin")

if __name__ == "__main__":
    gen_golden_data_simple(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))