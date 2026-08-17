/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dilation2_d_backprop_filter_tiling_data.h
 * \brief Tiling data struct for dilation2_d_backprop_filter operator
 *
 * MDE v2.1 §3.1: 18 fields, no threadNum (compile-time constant),
 * no magic/shift (computed in kernel Process and passed via UB).
 *
 * v2.1 change: deterministic accumulation (per-core buffer + final reduce)
 *   - Added perCoreBufElems: per-core buffer size in elements (128B aligned)
 *   - ComputeSimt atomic_add targets per-core workspace, not yGm directly
 *   - New Phase 3: sequential reduce of per-core buffers to yGm
 */

#ifndef DILATION2D_BACKPROP_FILTER_TILING_DATA_H_
#define DILATION2D_BACKPROP_FILTER_TILING_DATA_H_

struct Dilation2DBackpropFilterTilingData {
    int64_t totalElements = 0;   // batch × outH × outW × depth
    int64_t filterSize = 0;      // filterH × filterW × depth (ZeroOut element count)
    int32_t needCoreNum = 0;     // actual launched core count
    int64_t perCoreBufElems = 0; // per-core buffer size in elements (128B aligned), v2.1
    int32_t batch = 0;           // N
    int32_t inputH = 0;          // H_in
    int32_t inputW = 0;          // W_in
    int32_t depth = 0;           // C
    int32_t filterH = 0;         // filter height
    int32_t filterW = 0;         // filter width
    int32_t outH = 0;            // H_out
    int32_t outW = 0;            // W_out
    int32_t strideH = 0;         // strides[1]
    int32_t strideW = 0;         // strides[2]
    int32_t rateH = 0;           // rates[1]
    int32_t rateW = 0;           // rates[2]
    int32_t padTop = 0;          // top padding
    int32_t padLeft = 0;         // left padding
    int32_t padInputH = 0;       // padded input H (inputH + padTop + padBottom), v2.3
    int32_t padInputW = 0;       // padded input W (inputW + padLeft + padRight), v2.3
    int32_t isNCHW = 0;          // 0=NHWC, 1=NCHW (v2.5: NCHW support)
    // v2.0: fp32AccumOffset removed (T=float, direct accumulation on yGm)
};

#endif // DILATION2D_BACKPROP_FILTER_TILING_DATA_H_
