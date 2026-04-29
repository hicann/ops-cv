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
 * \file test_col2im.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kernel_tiling/kernel_tiling.h"
#include "image/col2im/op_kernel/col2im.cpp"
#include "../../../op_kernel/arch35/col2im_tiling_data.h"

using namespace std;

template <uint64_t dType>
__global__ __aicore__ void col2im(GM_ADDR gradOut, GM_ADDR outputSize, GM_ADDR gradIn, GM_ADDR workspace, GM_ADDR tiling);

class col2im_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "col2im SetUp\n"
             << endl;
    }
    static void TearDownTestCase()
    {
        cout << "col2im_test TearDown\n"
             << endl;
    }
};

TEST_F(col2im_test, test_col2im_950_fp16)
{
    size_t gradOutByteSize = 4 * 2 * sizeof(float);
    size_t outputSizeByteSize = 2 * sizeof(int32_t);
    size_t gradInByteSize = 1 * sizeof(float);
    size_t tilingDataSize = sizeof(Col2imRegBaseTilingData);

    uint8_t *gradOut = (uint8_t *)AscendC::GmAlloc(gradOutByteSize);
    uint8_t *outputSize = (uint8_t *)AscendC::GmAlloc(outputSizeByteSize);
    uint8_t *gradIn = (uint8_t *)AscendC::GmAlloc(gradInByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    Col2imRegBaseTilingData* tilingData = reinterpret_cast<Col2imRegBaseTilingData*>(tiling);
    tilingData->totalLength = 1;
    tilingData->outputSizeH = 1;
    tilingData->outputSizeW = 1;
    tilingData->kernelSizeH = 2;
    tilingData->kernelSizeW = 2;
    tilingData->dilationH = 1;
    tilingData->dilationW = 1;
    tilingData->paddingH = 1;
    tilingData->paddingW = 1;
    tilingData->strideH = 1;
    tilingData->strideW = 2;
    tilingData->colH = 2;
    tilingData->colW = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(col2im<0>, numBlocks, gradOut, outputSize, gradIn, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(outputSize);
    AscendC::GmFree(gradIn);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}