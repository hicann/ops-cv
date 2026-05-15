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
 * \file test_resize_linear_grad.cpp
 * \brief resize_linear_grad kernel UT test - direct class invocation (bypass ASCENDC_TPL template macros)
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

// Directly include ResizeLinearGradTensorMove class (bypass template function with ASCENDC_TPL macros)
#include "../../../op_kernel/arch35/resize_linear_grad_tensor_move.h"
#include "resize_linear_grad_tiling.h"

using namespace std;
using namespace AscendC;
using namespace ResizeLinearGrad;

class resize_linear_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "resize_linear_grad SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "resize_linear_grad TearDown\n" << endl;
    }
};

TEST_F(resize_linear_grad_test, test_tensor_move_fp32)
{
    // Direct class invocation (bypass ASCENDC_TPL template macros)
    // This mirrors apt.cpp's schId=1 implementation: pure data copy
    
    size_t gradsByteSize = 1 * 2 * 10 * 10 * sizeof(float);
    size_t originalImageByteSize = 1 * 2 * 10 * 10 * sizeof(float);
    size_t yByteSize = gradsByteSize;
    size_t tilingDataSize = sizeof(ResizeLinearGradTilingData);
    
    uint8_t *grads = (uint8_t *)GmAlloc(gradsByteSize);
    uint8_t *originalImage = (uint8_t *)GmAlloc(originalImageByteSize);
    uint8_t *y = (uint8_t *)GmAlloc(yByteSize);
    
    uint8_t *tiling = (uint8_t *)GmAlloc(tilingDataSize);
    
    // Initialize input data to fixed value for reliability
    memset(grads, 0, gradsByteSize);
    
    // Manually construct TilingData
    ResizeLinearGradTilingData* tilingData = reinterpret_cast<ResizeLinearGradTilingData*>(tiling);
    tilingData->realCoreNum = 1;
    tilingData->initCoreNum = 0;
    tilingData->blkProcessNum = 200;  // N * C * H * W = 1 * 2 * 10 * 10
    tilingData->ubLoopSizeB = 1;
    tilingData->ubLoopSizeT = 1;
    tilingData->ubFactor = 200;
    tilingData->ubFactorTailB = 200;
    tilingData->ubFactorTailT = 200;
    tilingData->lenSrcLOrUb = 200 * sizeof(float);  // ubSize (for TensorMove)
    tilingData->lenDesL = 200;
    tilingData->scaleL = 1.0f;
    tilingData->inverseScaleL = 1.0f;
    
    // Directly instantiate ResizeLinearGradTensorMove class (bypass template function)
    TPipe pipe;
    ResizeLinearGradTensorMove<float> op(pipe);
    op.Init(grads, originalImage, y, tilingData);
    op.Process();
    
    GmFree(grads);
    GmFree(originalImage);
    GmFree(y);
    GmFree(tiling);
}