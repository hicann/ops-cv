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
 * \file test_resize_linear.cpp
 * \brief resize_linear kernel UT test - direct class invocation (bypass ASCENDC_TPL template macros)
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

// Directly include ResizeLinearSimtNCL class (bypass template function)
#include "../../../op_kernel/arch35/resize_linear_simt_ncl.h"
#include "resize_linear_tiling.h"

using namespace std;
using namespace AscendC;
using namespace ResizeLinear;

class resize_linear_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "resize_linear SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "resize_linear TearDown\n" << endl;
    }
};

TEST_F(resize_linear_test, test_resize_linear_950_mode1_fp32)
{
    // Test mode=1 (pure copy): input and output sizes identical
    // Directly instantiate ResizeLinearSimtNCL class (bypass template function with ASCENDC_TPL macros)
    
    size_t inputByteSize = 1 * 1 * 10 * sizeof(float);
    size_t sizeByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 1 * 1 * 10 * sizeof(float);
    size_t tilingDataSize = sizeof(ResizeLinearTilingData);
    
    uint8_t *x = (uint8_t *)GmAlloc(inputByteSize);
    uint8_t *size_tensor = (uint8_t *)GmAlloc(sizeByteSize);
    uint8_t *y = (uint8_t *)GmAlloc(outputByteSize);
    
    uint8_t *tiling = (uint8_t *)GmAlloc(tilingDataSize);
    
    // Initialize input data to fixed value for reliability
    memset(x, 0, inputByteSize);
    
    // Prepare size data
    int32_t* sizeData = reinterpret_cast<int32_t*>(size_tensor);
    sizeData[0] = 10;  // output L
    
    // Manually construct TilingData
    ResizeLinearTilingData* tilingData = reinterpret_cast<ResizeLinearTilingData*>(tiling);
    tilingData->realCoreNum = 1;
    tilingData->blkProcessNum = 10;  // N * C * L_out
    tilingData->splitBlockTailFactor = 0;
    tilingData->lenSrcL = 10;
    tilingData->lenDesL = 10;
    tilingData->scaleL = 1.0f;
    
    // Directly instantiate ResizeLinearSimtNCL class (no ICPU_RUN_KF needed)
    // Template parameters: T1=float, T2=uint32_t, halfPixel=0, mode=1 (pure copy)
    ResizeLinearSimtNCL<float, uint32_t, 0, 1> op;
    op.Init(x, size_tensor, y, tilingData);
    op.Process();
    
    GmFree(x);
    GmFree(size_tensor);
    GmFree(y);
    GmFree(tiling);
}