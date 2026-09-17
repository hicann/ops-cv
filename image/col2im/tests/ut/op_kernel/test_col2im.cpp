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
__global__ __aicore__ void col2im(GM_ADDR gradOut, GM_ADDR outputSize, GM_ADDR gradIn, GM_ADDR workspace,
                                  GM_ADDR tiling);

class col2im_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "col2im SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "col2im_test TearDown\n" << endl; }
};

TEST_F(col2im_test, test_col2im_950_fp16)
{
    size_t gradOutByteSize = 4 * 2 * sizeof(float);
    size_t outputSizeByteSize = 2 * sizeof(int32_t);
    size_t gradInByteSize = 1 * sizeof(float);
    size_t tilingDataSize = sizeof(Col2imRegBaseTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutByteSize);
    uint8_t* outputSize = (uint8_t*)AscendC::GmAlloc(outputSizeByteSize);
    uint8_t* gradIn = (uint8_t*)AscendC::GmAlloc(gradInByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
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

    // 已知值输入：gradOut [C*kH*kW=4, colL=2]（flat 索引见 kernel 的 gradOutIdx 公式），
    // 期望 gradIn[0,0] = gradOut[3] + gradOut[6] = 4.0 + 7.0 = 11.0
    // （hIm,wIm=(1,1)：贡献点 (hk=0,wk=1,hg=1,wg=0) 与 (hk=1,wk=1,hg=0,wg=0)，
    //  与 col2im_simt.h 的 wGradOutStart/End 与 dilation 整除判定逐行核对）
    {
        float* g = reinterpret_cast<float*>(gradOut);
        for (int i = 0; i < 8; i++) {
            g[i] = static_cast<float>(i + 1);
        }
    }
    memset(gradIn, 0xA5, gradInByteSize);

    // 已知值输入：gradOut [C*kH*kW=4, colL=2]（flat 索引见 kernel 的 gradOutIdx 公式），
    // 期望 gradIn[0,0] = gradOut[3] + gradOut[6] = 4.0 + 7.0 = 11.0
    // （hIm,wIm=(1,1)：贡献点 (hk=0,wk=1,hg=1,wg=0) 与 (hk=1,wk=1,hg=0,wg=0)，
    //  与 col2im_simt.h 的 wGradOutStart/End 与 dilation 整除判定逐行核对）
    {
        float* g = reinterpret_cast<float*>(gradOut);
        for (int i = 0; i < 8; i++) {
            g[i] = static_cast<float>(i + 1);
        }
    }
    memset(gradIn, 0xA5, gradInByteSize);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(col2im<0>, numBlocks, gradOut, outputSize, gradIn, workspace, tiling);

    // 校验：outputSize=[1,1]+pad=1+stride=(1,2)+kernel=2x2 下，col2im 为两贡献点累加
    {
        const float* out = reinterpret_cast<const float*>(gradIn);
        const float expected = 11.0f;
        ASSERT_EQ(0, memcmp(&out[0], &expected, sizeof(float)))
            << "col2im output: got " << out[0] << ", expected " << expected;
    }

    // 校验：outputSize=[1,1]+pad=1+stride=(1,2)+kernel=2x2 下，col2im 为两贡献点累加
    {
        const float* out = reinterpret_cast<const float*>(gradIn);
        const float expected = 11.0f;
        ASSERT_EQ(0, memcmp(&out[0], &expected, sizeof(float)))
            << "col2im output: got " << out[0] << ", expected " << expected;
    }

    AscendC::GmFree(gradOut);
    AscendC::GmFree(outputSize);
    AscendC::GmFree(gradIn);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
