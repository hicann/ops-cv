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
 * \file test_resize_nearest_neighbor_v2.cpp
 * \brief resize_nearest_neighbor_v2 kernel UT test - direct Simt class invocation (bypass ASCENDC_TPL macros)
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

// Directly include ResizeNearestNeighborV2Simt class (bypass template function with ASCENDC_TPL macros)
#include "../../../op_kernel/arch35/resize_nearest_neighbor_v2_simt.h"
#include "resize_nearest_neighbor_v2_tiling.h"

using namespace std;
using namespace AscendC;
using namespace ResizeNearestNeighborV2;

constexpr int TPL_SCH_MODE_SIMT_INPUT_EQ_OUTPUT = 4;  // Input size == Output size

class resize_nearest_neighbor_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "resize_nearest_neighbor_v2 SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "resize_nearest_neighbor_v2 TearDown\n" << endl;
    }
};

TEST_F(resize_nearest_neighbor_v2_test, test_simt_input_eq_output_fp16)
{
    // Direct Simt class invocation (bypass ASCENDC_TPL template macros)
    // Test TPL_SCH_MODE_SIMT_INPUT_EQ_OUTPUT (mode=4) branch: input == output size
    
    size_t inputByteSize = 1 * 1 * 2 * 2 * sizeof(half);
    size_t sizeByteSize = 2 * sizeof(int32_t);
    size_t outputByteSize = 1 * 1 * 2 * 2 * sizeof(half);
    size_t tilingDataSize = sizeof(ResizeNearestNeighborV2TilingData);
    
    uint8_t *x = (uint8_t *)GmAlloc(inputByteSize);
    uint8_t *size_tensor = (uint8_t *)GmAlloc(sizeByteSize);
    uint8_t *y = (uint8_t *)GmAlloc(outputByteSize);
    
    uint8_t *tiling = (uint8_t *)GmAlloc(tilingDataSize);
    
    // Initialize input data to fixed value for reliability
    memset(x, 0, inputByteSize);
    
    // Prepare size data
    int32_t* sizeData = reinterpret_cast<int32_t*>(size_tensor);
    sizeData[0] = 2;  // output H
    sizeData[1] = 2;  // output W
    
    // Manually construct TilingData
    ResizeNearestNeighborV2TilingData* tilingData = reinterpret_cast<ResizeNearestNeighborV2TilingData*>(tiling);
    tilingData->realCoreNum = 1;
    tilingData->ubSize = 262144;
    tilingData->alignCorners = 0;
    tilingData->halfPixelCenters = 0;
    tilingData->lenN = 1;
    tilingData->lenC = 1;
    tilingData->lenSrcH = 2;
    tilingData->lenSrcW = 2;
    tilingData->lenDesH = 2;
    tilingData->lenDesW = 2;
    tilingData->condition = 0;
    tilingData->switchParams = 0;
    tilingData->splitBlockFactor = 4;
    tilingData->splitBlockTailFactor = 0;
    tilingData->lenCAlign = 1;
    tilingData->hwcNum = 4;
    tilingData->dstHwcNum = 4;
    tilingData->wcNum = 2;
    tilingData->dstWcNum = 2;
    tilingData->nLoop = 1;
    tilingData->nLoopTimesBefore = 1;
    tilingData->nLoopTimesLast = 1;
    tilingData->nLoopTailLast = 0;
    tilingData->wcLoop = 1;
    tilingData->wcLoopTimesBefore = 1;
    tilingData->wcLoopTailBefore = 0;
    tilingData->wcLoopTimesLast = 1;
    tilingData->wcLoopTailLast = 0;
    tilingData->splitBlockFullCount = 1;
    tilingData->splitFactorDesH = 1;
    tilingData->splitFactorTailDesH = 0;
    tilingData->splitCountDesH = 1;
    tilingData->splitFactorDesW = 1;
    tilingData->splitFactorTailDesW = 0;
    tilingData->splitCountDesW = 1;
    tilingData->scaleW = 1.0f;
    tilingData->scaleH = 1.0f;
    
    // Directly instantiate ResizeNearestNeighborV2Simt class (no ICPU_RUN_KF needed)
    // Template params: T=half, T_IDX=uint32_t, format=FORMAT_NCHW, mode=INPUT_EQ_OUTPUT, align_corner=false, half_pixel=false
    ResizeNearestNeighborV2Simt<half, uint32_t, FORMAT_NCHW, TPL_SCH_MODE_SIMT_INPUT_EQ_OUTPUT, false, false> op;
    op.Init(x, size_tensor, y, tilingData);
    op.Process();
    
    GmFree(x);
    GmFree(size_tensor);
    GmFree(y);
    GmFree(tiling);
}