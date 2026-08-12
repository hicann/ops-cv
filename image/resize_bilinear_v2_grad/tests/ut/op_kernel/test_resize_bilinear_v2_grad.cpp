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
 * \file test_resize_bilinear_v2_grad.cpp
 * \brief resize_bilinear_v2_grad kernel UT test - selective compilation (ALL_COPY branch only)
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

// Selective compilation: only include ALL_COPY and POINT_COPY branches
// Avoid including resize_bilinear_v2_grad_apt.cpp which instantiates all branches (including Simt)
#include "../../../op_kernel/arch35/resize_bilinear_v2_grad_base.h"
#include "../../../op_kernel/arch35/resize_bilinear_v2_grad_all_copy.h"
#include "../../../op_kernel/arch35/resize_bilinear_v2_grad_point_copy.h"
#include "resize_bilinear_v2_grad_tiling.h"

using namespace std;
using namespace AscendC;
using namespace ResizeBilinearV2Grad;

class resize_bilinear_v2_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "resize_bilinear_v2_grad SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "resize_bilinear_v2_grad TearDown\n" << endl; }
};

TEST_F(resize_bilinear_v2_grad_test, test_resize_bilinear_v2_grad_950_all_copy_fp16)
{
    // Test TILING_KEY_ALL_COPY (30000): grads and y sizes identical
    // Input: grads (1, 1, 2, 2) NCHW, FP16
    // Input: originalImage (not used in ALL_COPY branch)
    // Output: y (1, 1, 2, 2) identical

    size_t gradsByteSize = 1 * 1 * 2 * 2 * sizeof(half);
    size_t originalImageByteSize = 1 * 1 * 2 * 2 * sizeof(half);
    size_t yByteSize = 1 * 1 * 2 * 2 * sizeof(half);
    size_t tilingDataSize = sizeof(ResizeBilinearV2GradTilingData);

    uint8_t* grads = (uint8_t*)GmAlloc(gradsByteSize);
    uint8_t* originalImage = (uint8_t*)GmAlloc(originalImageByteSize);
    uint8_t* y = (uint8_t*)GmAlloc(yByteSize);

    uint8_t* tiling = (uint8_t*)GmAlloc(tilingDataSize);

    // Initialize input data to fixed value for reliability
    memset(grads, 0, gradsByteSize);

    // Manually construct TilingData
    ResizeBilinearV2GradTilingData* tilingData = reinterpret_cast<ResizeBilinearV2GradTilingData*>(tiling);
    tilingData->tilingKey = 30000;
    tilingData->ubSize = 262144;
    tilingData->alignCorners = 0;
    tilingData->halfPixelCenters = 0;
    tilingData->lenN = 1;
    tilingData->lenC = 1;
    tilingData->lenSrcH = 2;
    tilingData->lenSrcW = 2;
    tilingData->lenDesH = 2;
    tilingData->lenDesW = 2;
    tilingData->nFactor = 1;
    tilingData->hFactor = 1;
    tilingData->wFactor = 1;
    tilingData->cFactor = 1;
    tilingData->hwFactor = 1;
    tilingData->ubNFactor = 1;
    tilingData->ubHFactor = 1;
    tilingData->ubWFactor = 1;
    tilingData->ubCFactor = 4;
    tilingData->ubHWFactor = 4;
    tilingData->initYRealCoreNum = 1;
    tilingData->initYSplitBlockFactor = 4;
    tilingData->initYSplitBlockTailFactor = 0;
    tilingData->realCoreNum = 1;
    tilingData->splitBlockFactor = 4;
    tilingData->splitBlockTailFactor = 0;
    tilingData->scaleH = 1.0f;
    tilingData->scaleW = 1.0f;
    tilingData->inverseScaleH = 1.0f;
    tilingData->inverseScaleW = 1.0f;

    // Directly instantiate ResizeBilinearV2GradAllCopy class (no Simt dependency)
    TPipe pipe;
    ResizeBilinearV2GradAllCopy<half, half> op;
    op.Init(grads, originalImage, y, &pipe, tilingData);
    op.Process();

    GmFree(grads);
    GmFree(originalImage);
    GmFree(y);
    GmFree(tiling);
}

// Regression test for the integer divide-by-zero fix in ResizeBilinearV2GradPointCopy::CalcTile.
// Problem scenario (issue): alignCorners=true and lenDesH==1 && lenDesW==1 selects the PointCopy path;
// CalcTile computes (lenSrcH-1)/(lenDesH-1) where lenDesH-1==0 -> integer divide by zero (SIGFPE).
// Before the fix this test crashes; after the fix it completes normally.
TEST_F(resize_bilinear_v2_grad_test, test_resize_bilinear_v2_grad_950_point_copy_align_corners_desHW_1)
{
    // grads (dest) NHWC (1, 1, 1, 64): lenDesH==1 && lenDesW==1
    // y     (src)  NHWC (1, 4, 4, 64): lenSrcH==4 && lenSrcW==4
    // lenC==64, FP16 -> lenC * sizeof(half) == 128 == MIN_C_SIZE (PointCopy condition satisfied)
    const int64_t lenN = 1;
    const int64_t lenC = 64;
    const int64_t lenSrcH = 4;
    const int64_t lenSrcW = 4;
    const int64_t lenDesH = 1;
    const int64_t lenDesW = 1;

    size_t gradsByteSize = lenN * lenDesH * lenDesW * lenC * sizeof(half);
    size_t originalImageByteSize = gradsByteSize;
    size_t yByteSize = lenN * lenSrcH * lenSrcW * lenC * sizeof(half);
    size_t tilingDataSize = sizeof(ResizeBilinearV2GradTilingData);

    uint8_t* grads = (uint8_t*)GmAlloc(gradsByteSize);
    uint8_t* originalImage = (uint8_t*)GmAlloc(originalImageByteSize);
    uint8_t* y = (uint8_t*)GmAlloc(yByteSize);
    uint8_t* tiling = (uint8_t*)GmAlloc(tilingDataSize);

    memset(grads, 0, gradsByteSize);
    memset(y, 0, yByteSize);

    ResizeBilinearV2GradTilingData* tilingData = reinterpret_cast<ResizeBilinearV2GradTilingData*>(tiling);
    tilingData->tilingKey = 30001; // TILING_KEY_POINT_COPY
    tilingData->ubSize = 262144;
    tilingData->alignCorners = 1; // triggers the (lenSrcH-1)/(lenDesH-1) branch
    tilingData->halfPixelCenters = 0;
    tilingData->lenN = lenN;
    tilingData->lenC = lenC;
    tilingData->lenSrcH = lenSrcH;
    tilingData->lenSrcW = lenSrcW;
    tilingData->lenDesH = lenDesH; // == 1, was the divide-by-zero source
    tilingData->lenDesW = lenDesW; // == 1
    tilingData->nFactor = 1;
    tilingData->hFactor = 1;
    tilingData->wFactor = 1;
    tilingData->cFactor = lenC;
    tilingData->hwFactor = 1;
    tilingData->ubNFactor = 1;
    tilingData->ubHFactor = 1;
    tilingData->ubWFactor = 1;
    tilingData->ubCFactor = lenC;
    tilingData->ubHWFactor = lenC;
    tilingData->initYRealCoreNum = 1;
    tilingData->initYSplitBlockFactor = lenSrcH * lenSrcW * lenC; // clear the whole y
    tilingData->initYSplitBlockTailFactor = 0;
    tilingData->realCoreNum = 1;
    tilingData->splitBlockFactor = lenSrcH * lenSrcW * lenC;
    tilingData->splitBlockTailFactor = 0;
    tilingData->scaleH = 0.0f;
    tilingData->scaleW = 0.0f;
    tilingData->inverseScaleH = 0.0f;
    tilingData->inverseScaleW = 0.0f;

    TPipe pipe;
    ResizeBilinearV2GradPointCopy<half, half> op;
    op.Init(grads, originalImage, y, &pipe, tilingData);
    op.Process(); // must not crash with divide-by-zero

    GmFree(grads);
    GmFree(originalImage);
    GmFree(y);
    GmFree(tiling);
}
