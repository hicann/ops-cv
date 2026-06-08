/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_upsample_trilinear_tiling_data.h
 * \brief tiling data struct for arch35
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_ARCH35_TILING_DATA_H
#define RESIZE_UPSAMPLE_TRILINEAR_ARCH35_TILING_DATA_H

struct ResizeUpsampleTrilinearArch35TilingData {
    uint32_t elements_per_thread;
    uint32_t block_count;
    uint32_t used_core_num;
    uint32_t base_elements_per_block;
    uint32_t tail_elements;
    uint64_t total_elements;
    int64_t batch_count;
    int64_t input_d;
    int64_t input_h;
    int64_t input_w;
    int64_t output_d;
    int64_t output_h;
    int64_t output_w;
    float scale_d;
    float scale_h;
    float scale_w;
    int32_t align_corners;
    int32_t use_int32;
};
#endif