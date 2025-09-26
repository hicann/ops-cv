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
import copy
import torch
import numpy as np


def stack_group_points_forward(
    features, features_batch_cnt, indices, indices_batch_cnt, output, b, m, c, nsample
):
    for index in range(m * c * nsample):
        cur_features = features
        cur_idx = indices
        sample_idx = index % nsample
        c_idx = (index // nsample) % c
        pt_idx = index // (nsample * c)
        if pt_idx >= m or c_idx >= c or sample_idx >= nsample:
            return output.view(m, c, nsample)
        bs_idx = 0
        pt_cnt = indices_batch_cnt[0]
        pt_cnt = copy.deepcopy(pt_cnt)
        for k in range(1, b):
            if pt_idx >= pt_cnt:
                pt_cnt += indices_batch_cnt[k]
                bs_idx = k
        features_batch_start_idx = 0
        features_batch_end_idx = features_batch_cnt[0]
        for k in range(bs_idx):
            features_batch_start_idx += features_batch_cnt[k]
            features_batch_end_idx = (
                features_batch_start_idx + features_batch_cnt[k + 1]
            )
        cur_features = cur_features[features_batch_start_idx * c:]
        cur_idx = cur_idx[pt_idx * nsample + sample_idx]
        in_idx = cur_idx * c + c_idx
        out_idx = pt_idx * c * nsample + c_idx * nsample + sample_idx
        if in_idx < features_batch_end_idx * c and in_idx < len(cur_features):
            output[out_idx] = cur_features[in_idx]
    return output.view(m, c, nsample)


def stack_group_points_golden(m, nsample, n, c, b, np_dtype):
    if np_dtype == "np.float32":
        features = torch.rand(n, c, dtype=torch.float32)
    else:
        features = torch.rand(n, c, dtype=torch.float16)
    features_batch_cnt = torch.randint(0, 10, (b + 1,), dtype=torch.int32)
    indices = torch.randint(0, 10, (m, nsample), dtype=torch.int32)
    indices_batch_cnt = torch.randint(0, 10, (b,), dtype=torch.int32)
    output = features.new_zeros((m, c, nsample))
    features = features.view(-1)
    features_batch_cnt = features_batch_cnt.view(-1)
    indices = indices.view(-1)
    indices_batch_cnt = indices_batch_cnt.view(-1)
    output = output.view(-1)

    out = stack_group_points_forward(
        features,
        features_batch_cnt,
        indices,
        indices_batch_cnt,
        output,
        b,
        m,
        c,
        nsample,
    )
    golden = out.numpy().astype(eval(np_dtype))
    input_features = features.numpy().astype(eval(np_dtype))
    input_fbc = features_batch_cnt.numpy().astype(np.int32)
    input_ibc = indices_batch_cnt.numpy().astype(np.int32)
    input_indices = indices.numpy().astype(np.int32)
    input_features.tofile("./features.bin")
    input_fbc.tofile("./features_batch_cnt.bin")
    input_indices.tofile("./indices.bin")
    input_ibc.tofile("./indices_batch_cnt.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    M = int(sys.argv[1])
    nsample = int(sys.argv[2])
    N = int(sys.argv[3])
    C = int(sys.argv[4])
    B = int(sys.argv[5])
    np_dtype = sys.argv[6]
    stack_group_points_golden(M, nsample, N, C, B, np_dtype)
