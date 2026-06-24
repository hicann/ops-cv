/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_resize_nearest_neighbor_v2_grad.cpp
 * \brief resize_nearest_neighbor_v2_grad kernel UT test - AllCopy branch (no Simt dependency)
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

#include "../../../op_kernel/arch35/resize_nearest_neighbor_v2_grad_simt_base.h"
#include "../../../op_kernel/arch35/resize_nearest_neighbor_v2_grad_all_copy.h"
#include "resize_nearest_neighbor_v2_grad_tiling.h"

using namespace std;
using namespace AscendC;
using namespace ResizeNearestNeighborV2Grad;

class resize_nearest_neighbor_v2_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "resize_nearest_neighbor_v2_grad SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "resize_nearest_neighbor_v2_grad TearDown\n" << endl; }
};

TEST_F(resize_nearest_neighbor_v2_grad_test, test_resize_nearest_neighbor_v2_grad_all_copy_fp16)
{
    size_t gradsByteSize = 1 * 1 * 2 * 2 * sizeof(half);
    size_t sizeByteSize = 2 * sizeof(int32_t);
    size_t yByteSize = 1 * 1 * 2 * 2 * sizeof(half);
    size_t tilingDataSize = sizeof(ResizeNearestNeighborV2GradTilingData);

    uint8_t* grads = (uint8_t*)GmAlloc(gradsByteSize);
    uint8_t* size_tensor = (uint8_t*)GmAlloc(sizeByteSize);
    uint8_t* y = (uint8_t*)GmAlloc(yByteSize);

    uint8_t* tiling = (uint8_t*)GmAlloc(tilingDataSize);

    memset(grads, 0, gradsByteSize);

    int32_t* sizeData = reinterpret_cast<int32_t*>(size_tensor);
    sizeData[0] = 2;
    sizeData[1] = 2;

    ResizeNearestNeighborV2GradTilingData* tilingData = reinterpret_cast<ResizeNearestNeighborV2GradTilingData*>(
        tiling);
    tilingData->ubSize = 262144;
    tilingData->lenN = 1;
    tilingData->lenC = 1;
    tilingData->lenSrcH = 2;
    tilingData->lenSrcW = 2;
    tilingData->lenDstH = 2;
    tilingData->lenDstW = 2;
    tilingData->ubCFactor = 4;
    tilingData->scaleH = 1.0f;
    tilingData->scaleW = 1.0f;
    tilingData->inverseScaleH = 1.0f;
    tilingData->inverseScaleW = 1.0f;
    tilingData->initYRealCoreNum = 0;
    tilingData->initYSplitBlockFactor = 0;
    tilingData->initYSplitBlockTailFactor = 0;
    tilingData->realCoreNum = 1;
    tilingData->splitBlockFactor = 4;
    tilingData->splitBlockTailFactor = 0;

    TPipe pipe;
    ResizeNearestNeighborV2GradAllCopy<half, half> op;
    op.Init(grads, size_tensor, y, &pipe, tilingData);
    op.Process();

    GmFree(grads);
    GmFree(size_tensor);
    GmFree(y);
    GmFree(tiling);
}

TEST_F(resize_nearest_neighbor_v2_grad_test, test_resize_nearest_neighbor_v2_grad_all_copy_fp32)
{
    size_t gradsByteSize = 1 * 1 * 4 * 4 * sizeof(float);
    size_t sizeByteSize = 2 * sizeof(int32_t);
    size_t yByteSize = 1 * 1 * 4 * 4 * sizeof(float);
    size_t tilingDataSize = sizeof(ResizeNearestNeighborV2GradTilingData);

    uint8_t* grads = (uint8_t*)GmAlloc(gradsByteSize);
    uint8_t* size_tensor = (uint8_t*)GmAlloc(sizeByteSize);
    uint8_t* y = (uint8_t*)GmAlloc(yByteSize);

    uint8_t* tiling = (uint8_t*)GmAlloc(tilingDataSize);

    memset(grads, 0, gradsByteSize);

    int32_t* sizeData = reinterpret_cast<int32_t*>(size_tensor);
    sizeData[0] = 4;
    sizeData[1] = 4;

    ResizeNearestNeighborV2GradTilingData* tilingData = reinterpret_cast<ResizeNearestNeighborV2GradTilingData*>(
        tiling);
    tilingData->ubSize = 262144;
    tilingData->lenN = 1;
    tilingData->lenC = 1;
    tilingData->lenSrcH = 4;
    tilingData->lenSrcW = 4;
    tilingData->lenDstH = 4;
    tilingData->lenDstW = 4;
    tilingData->ubCFactor = 16;
    tilingData->scaleH = 1.0f;
    tilingData->scaleW = 1.0f;
    tilingData->inverseScaleH = 1.0f;
    tilingData->inverseScaleW = 1.0f;
    tilingData->initYRealCoreNum = 0;
    tilingData->initYSplitBlockFactor = 0;
    tilingData->initYSplitBlockTailFactor = 0;
    tilingData->realCoreNum = 1;
    tilingData->splitBlockFactor = 16;
    tilingData->splitBlockTailFactor = 0;

    TPipe pipe;
    ResizeNearestNeighborV2GradAllCopy<float, float> op;
    op.Init(grads, size_tensor, y, &pipe, tilingData);
    op.Process();

    GmFree(grads);
    GmFree(size_tensor);
    GmFree(y);
    GmFree(tiling);
}
