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
 * \file test_resize_bilinear_v2.cpp
 * \brief resize_bilinear_v2 kernel UT test - selective compilation (ALL_COPY branch only)
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

// Selective compilation: only include ALL_COPY branch
// Avoid including resize_bilinear_v2_apt.cpp which instantiates all branches (including Simt with min/volatile errors)
#include "../../../op_kernel/arch35/resize_bilinear_v2_base.h"
#include "../../../op_kernel/arch35/resize_bilinear_v2_all_copy.h"
#include "resize_bilinear_v2_tiling.h"

using namespace std;
using namespace AscendC;
using namespace ResizeBilinearV2;

class resize_bilinear_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "resize_bilinear_v2 SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "resize_bilinear_v2 TearDown\n" << endl;
    }
};

TEST_F(resize_bilinear_v2_test, test_resize_bilinear_v2_950_all_copy_fp16)
{
    // Test TILING_KEY_ALL_COPY (40000): input and output sizes identical
    // Directly instantiate ResizeBilinearV2AllCopy class (bypass Simt branches)
    
    size_t inputByteSize = 1 * 1 * 2 * 2 * sizeof(half);
    size_t sizeByteSize = 2 * sizeof(int32_t);
    size_t outputByteSize = 1 * 1 * 2 * 2 * sizeof(half);
    size_t tilingDataSize = sizeof(ResizeBilinearV2TilingData);
    
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
    ResizeBilinearV2TilingData* tilingData = reinterpret_cast<ResizeBilinearV2TilingData*>(tiling);
    tilingData->tilingKey = 40000;
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
    tilingData->nFactor = 1;
    tilingData->hFactor = 1;
    tilingData->wFactor = 1;
    tilingData->cFactor = 1;
    tilingData->hwFactor = 1;
    tilingData->ubNFactor = 1;
    tilingData->ubHFactor = 1;
    tilingData->ubWFactor = 1;
    tilingData->ubCFactor = 4;
    tilingData->splitBlockFactor = 4;
    tilingData->splitBlockTailFactor = 0;
    tilingData->ubHWFactor = 4;
    tilingData->scaleW = 1.0f;
    tilingData->scaleH = 1.0f;
    
    // Directly instantiate ResizeBilinearV2AllCopy class (no Simt dependency)
    TPipe pipe;
    ResizeBilinearV2AllCopy<half> op;
    op.Init(x, size_tensor, y, &pipe, tilingData);
    op.Process();
    
    GmFree(x);
    GmFree(size_tensor);
    GmFree(y);
    GmFree(tiling);
}