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
 * \file crop_tiling_data.h
 * \brief Tiling data struct for crop operator
 */

#ifndef CROP_TILING_DATA_H_
#define CROP_TILING_DATA_H_

// 最大支持维度数，xStrides/yStrides 数组容量上限
static constexpr int32_t MAX_NDIM = 8;

struct CropTilingData {
    int32_t needCoreNum = 0;
    int64_t totalElements = 0;
    int32_t rank = 0;
    int32_t axis = 0;
    int64_t xStrides[MAX_NDIM] = {};
    int64_t yStrides[MAX_NDIM] = {};
    int64_t mainBlockFactor = 0;
    int64_t tailBlockFactor = 0;
    // offsets 预计算结果，kernel 中用于偏移输入指针
    int64_t baseOffset = 0;
};

#endif // CROP_TILING_DATA_H_
