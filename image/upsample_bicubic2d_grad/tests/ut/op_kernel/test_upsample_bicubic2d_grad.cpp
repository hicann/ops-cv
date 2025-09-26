/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

using namespace std;

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

extern "C" __global__ __aicore__ void upsample_bicubic2d_grad(
    GM_ADDR grad_output, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling_addr);

class upsample_bicubic2d_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "upsample_bicubic2d_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "upsample_bicubic2d_grad_test TearDown\n" << endl;
    }
};

TEST_F(upsample_bicubic2d_grad_test, test_case_1)
{
    int b = 2;
    int c = 16;
    int h_in = 64;
    int w_in = 64;
    int h_out = 32;
    int w_out = 32;

    int64_t inSize = b * c * h_in * w_in * sizeof(half);
    int64_t outSize = b * c * h_out * w_out * sizeof(half);

    uint8_t *inGM = (uint8_t *)AscendC::GmAlloc(inSize);
    uint8_t *outGM = (uint8_t *)AscendC::GmAlloc(outSize);

    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(UpsampleBicubic2dGradTilingData));
    UpsampleBicubic2dGradTilingData *tilingDatafromBin = reinterpret_cast<UpsampleBicubic2dGradTilingData *>(tiling);
    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->CoreNum = 4;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->scalesH = 0.492063;
    tilingDatafromBin->scalesW = 0.492063;
    tilingDatafromBin->baseNH = 16;
    tilingDatafromBin->baseNW = 16;
    tilingDatafromBin->batch = 32;
    tilingDatafromBin->inputH = 64;
    tilingDatafromBin->inputW = 64;
    tilingDatafromBin->outputH = 32;
    tilingDatafromBin->outputW = 32;
    tilingDatafromBin->tailH = 16;
    tilingDatafromBin->CoreNumH = 4;
    tilingDatafromBin->loopH = 1;
    tilingDatafromBin->loopTailCoreH = 0;
    tilingDatafromBin->innerCoreNumH = 1;
    tilingDatafromBin->innerBatchH = 32;
    tilingDatafromBin->innerBatchTailCoreH = 0;
    tilingDatafromBin->tailW = 16;
    tilingDatafromBin->CoreNumW = 4;
    tilingDatafromBin->loopW = 1;
    tilingDatafromBin->loopTailCoreW = 0;
    tilingDatafromBin->innerCoreNumW = 1;
    tilingDatafromBin->innerBatchW = 32;
    tilingDatafromBin->innerBatchTailCoreW = 0;
    tilingDatafromBin->clearBaseN = 65536;
    tilingDatafromBin->clearInterLoop = 0;
    tilingDatafromBin->clearInterTailN = 16384;
    tilingDatafromBin->clearInterTailCoreNum = 0;
    tilingDatafromBin->clearOutLoop = 0;
    tilingDatafromBin->clearOutTailN = 8192;
    tilingDatafromBin->clearOutTailCoreNum = 0;

    tilingDatafromBin->MMParamH.usedCoreNum = 1;
    tilingDatafromBin->MMParamH.M = 16;
    tilingDatafromBin->MMParamH.N = 32;
    tilingDatafromBin->MMParamH.Ka = 16;
    tilingDatafromBin->MMParamH.Kb = 64;
    tilingDatafromBin->MMParamH.singleCoreM = 16;
    tilingDatafromBin->MMParamH.singleCoreN = 32;
    tilingDatafromBin->MMParamH.singleCoreK = 16;
    tilingDatafromBin->MMParamH.baseM = 16;
    tilingDatafromBin->MMParamH.baseN = 32;
    tilingDatafromBin->MMParamH.baseK = 16;
    tilingDatafromBin->MMParamH.depthA1 = 1;
    tilingDatafromBin->MMParamH.depthB1 = 1;
    tilingDatafromBin->MMParamH.stepM = 1;
    tilingDatafromBin->MMParamH.stepN = 1;
    tilingDatafromBin->MMParamH.isBias = 0;
    tilingDatafromBin->MMParamH.transLength = 0;
    tilingDatafromBin->MMParamH.iterateOrder = 0;
    tilingDatafromBin->MMParamH.shareMode = 0;
    tilingDatafromBin->MMParamH.shareL1Size = 1536;
    tilingDatafromBin->MMParamH.shareL0CSize = 2048;
    tilingDatafromBin->MMParamH.shareUbSize = 0;
    tilingDatafromBin->MMParamH.batchM = 1;
    tilingDatafromBin->MMParamH.batchN = 1;
    tilingDatafromBin->MMParamH.singleBatchM = 1;
    tilingDatafromBin->MMParamH.singleBatchN = 1;
    tilingDatafromBin->MMParamH.stepKa = 1;
    tilingDatafromBin->MMParamH.stepKb = 1;
    tilingDatafromBin->MMParamH.dbL0A = 2;
    tilingDatafromBin->MMParamH.dbL0B = 2;
    tilingDatafromBin->MMParamH.dbL0C = 1;

    tilingDatafromBin->MMParamW.usedCoreNum = 1;
    tilingDatafromBin->MMParamW.M = 2048;
    tilingDatafromBin->MMParamW.N = 16;
    tilingDatafromBin->MMParamW.Ka = 64;
    tilingDatafromBin->MMParamW.Kb = 16;
    tilingDatafromBin->MMParamW.singleCoreM = 2048;
    tilingDatafromBin->MMParamW.singleCoreN = 16;
    tilingDatafromBin->MMParamW.singleCoreK = 16;
    tilingDatafromBin->MMParamW.baseM = 1024;
    tilingDatafromBin->MMParamW.baseN = 16;
    tilingDatafromBin->MMParamW.baseK = 16;
    tilingDatafromBin->MMParamW.depthA1 = 2;
    tilingDatafromBin->MMParamW.depthB1 = 1;
    tilingDatafromBin->MMParamW.stepM = 2;
    tilingDatafromBin->MMParamW.stepN = 1;
    tilingDatafromBin->MMParamH.isBias = 0;
    tilingDatafromBin->MMParamW.transLength = 0;
    tilingDatafromBin->MMParamW.iterateOrder = 0;
    tilingDatafromBin->MMParamW.shareMode = 0;
    tilingDatafromBin->MMParamW.shareL1Size = 66048;
    tilingDatafromBin->MMParamW.shareL0CSize = 65536;
    tilingDatafromBin->MMParamW.shareUbSize = 0;
    tilingDatafromBin->MMParamW.batchM = 1;
    tilingDatafromBin->MMParamW.batchN = 1;
    tilingDatafromBin->MMParamW.singleBatchM = 1;
    tilingDatafromBin->MMParamW.singleBatchN = 1;
    tilingDatafromBin->MMParamW.stepKa = 1;
    tilingDatafromBin->MMParamW.stepKb = 1;
    tilingDatafromBin->MMParamW.dbL0A = 2;
    tilingDatafromBin->MMParamW.dbL0B = 2;
    tilingDatafromBin->MMParamW.dbL0C = 1;

    int64_t workspaceSize = 16910336;
    uint8_t *workspaceGM = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(10000001);
    ICPU_RUN_KF(upsample_bicubic2d_grad, 2, inGM, outGM, workspaceGM, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(inGM);
    AscendC::GmFree(outGM);
    AscendC::GmFree(workspaceGM);
    AscendC::GmFree(tiling);
}

TEST_F(upsample_bicubic2d_grad_test, test_case_2)
{
    int b = 2;
    int c = 16;
    int h_in = 64;
    int w_in = 64;
    int h_out = 32;
    int w_out = 32;

    int64_t inSize = b * c * h_in * w_in * sizeof(float);
    int64_t outSize = b * c * h_out * w_out * sizeof(float);

    uint8_t *inGM = (uint8_t *)AscendC::GmAlloc(inSize);
    uint8_t *outGM = (uint8_t *)AscendC::GmAlloc(outSize);

    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(UpsampleBicubic2dGradTilingData));
    UpsampleBicubic2dGradTilingData *tilingDatafromBin = reinterpret_cast<UpsampleBicubic2dGradTilingData *>(tiling);
    tilingDatafromBin->dataType = 0;
    tilingDatafromBin->CoreNum = 4;
    tilingDatafromBin->alignCorners = 1;
    tilingDatafromBin->scalesH = 0.492063;
    tilingDatafromBin->scalesW = 0.492063;
    tilingDatafromBin->baseNH = 16;
    tilingDatafromBin->baseNW = 16;
    tilingDatafromBin->batch = 32;
    tilingDatafromBin->inputH = 64;
    tilingDatafromBin->inputW = 64;
    tilingDatafromBin->outputH = 32;
    tilingDatafromBin->outputW = 32;
    tilingDatafromBin->tailH = 16;
    tilingDatafromBin->CoreNumH = 4;
    tilingDatafromBin->loopH = 1;
    tilingDatafromBin->loopTailCoreH = 0;
    tilingDatafromBin->innerCoreNumH = 1;
    tilingDatafromBin->innerBatchH = 32;
    tilingDatafromBin->innerBatchTailCoreH = 0;
    tilingDatafromBin->tailW = 16;
    tilingDatafromBin->CoreNumW = 4;
    tilingDatafromBin->loopW = 1;
    tilingDatafromBin->loopTailCoreW = 0;
    tilingDatafromBin->innerCoreNumW = 1;
    tilingDatafromBin->innerBatchW = 32;
    tilingDatafromBin->innerBatchTailCoreW = 0;
    tilingDatafromBin->clearBaseN = 32768;
    tilingDatafromBin->clearInterLoop = 0;
    tilingDatafromBin->clearInterTailN = 16384;
    tilingDatafromBin->clearInterTailCoreNum = 0;
    tilingDatafromBin->clearOutLoop = 0;
    tilingDatafromBin->clearOutTailN = 8192;
    tilingDatafromBin->clearOutTailCoreNum = 0;

    tilingDatafromBin->MMParamH.usedCoreNum = 1;
    tilingDatafromBin->MMParamH.M = 16;
    tilingDatafromBin->MMParamH.N = 32;
    tilingDatafromBin->MMParamH.Ka = 16;
    tilingDatafromBin->MMParamH.Kb = 64;
    tilingDatafromBin->MMParamH.singleCoreM = 16;
    tilingDatafromBin->MMParamH.singleCoreN = 32;
    tilingDatafromBin->MMParamH.singleCoreK = 16;
    tilingDatafromBin->MMParamH.baseM = 16;
    tilingDatafromBin->MMParamH.baseN = 32;
    tilingDatafromBin->MMParamH.baseK = 16;
    tilingDatafromBin->MMParamH.depthA1 = 1;
    tilingDatafromBin->MMParamH.depthB1 = 1;
    tilingDatafromBin->MMParamH.stepM = 1;
    tilingDatafromBin->MMParamH.stepN = 1;
    tilingDatafromBin->MMParamH.isBias = 0;
    tilingDatafromBin->MMParamH.transLength = 0;
    tilingDatafromBin->MMParamH.iterateOrder = 0;
    tilingDatafromBin->MMParamH.shareMode = 0;
    tilingDatafromBin->MMParamH.shareL1Size = 3072;
    tilingDatafromBin->MMParamH.shareL0CSize = 2048;
    tilingDatafromBin->MMParamH.shareUbSize = 0;
    tilingDatafromBin->MMParamH.batchM = 1;
    tilingDatafromBin->MMParamH.batchN = 1;
    tilingDatafromBin->MMParamH.singleBatchM = 1;
    tilingDatafromBin->MMParamH.singleBatchN = 1;
    tilingDatafromBin->MMParamH.stepKa = 1;
    tilingDatafromBin->MMParamH.stepKb = 1;
    tilingDatafromBin->MMParamH.dbL0A = 2;
    tilingDatafromBin->MMParamH.dbL0B = 2;
    tilingDatafromBin->MMParamH.dbL0C = 1;

    tilingDatafromBin->MMParamW.usedCoreNum = 1;
    tilingDatafromBin->MMParamW.M = 2048;
    tilingDatafromBin->MMParamW.N = 16;
    tilingDatafromBin->MMParamW.Ka = 64;
    tilingDatafromBin->MMParamW.Kb = 16;
    tilingDatafromBin->MMParamW.singleCoreM = 2048;
    tilingDatafromBin->MMParamW.singleCoreN = 16;
    tilingDatafromBin->MMParamW.singleCoreK = 16;
    tilingDatafromBin->MMParamW.baseM = 1024;
    tilingDatafromBin->MMParamW.baseN = 16;
    tilingDatafromBin->MMParamW.baseK = 8;
    tilingDatafromBin->MMParamW.depthA1 = 4;
    tilingDatafromBin->MMParamW.depthB1 = 2;
    tilingDatafromBin->MMParamW.stepM = 2;
    tilingDatafromBin->MMParamW.stepN = 1;
    tilingDatafromBin->MMParamH.isBias = 0;
    tilingDatafromBin->MMParamW.transLength = 0;
    tilingDatafromBin->MMParamW.iterateOrder = 0;
    tilingDatafromBin->MMParamW.shareMode = 0;
    tilingDatafromBin->MMParamW.shareL1Size = 132096;
    tilingDatafromBin->MMParamW.shareL0CSize = 65536;
    tilingDatafromBin->MMParamW.shareUbSize = 0;
    tilingDatafromBin->MMParamW.batchM = 1;
    tilingDatafromBin->MMParamW.batchN = 1;
    tilingDatafromBin->MMParamW.singleBatchM = 1;
    tilingDatafromBin->MMParamW.singleBatchN = 1;
    tilingDatafromBin->MMParamW.stepKa = 2;
    tilingDatafromBin->MMParamW.stepKb = 2;
    tilingDatafromBin->MMParamW.dbL0A = 2;
    tilingDatafromBin->MMParamW.dbL0B = 2;
    tilingDatafromBin->MMParamW.dbL0C = 1;

    int64_t workspaceSize = 17043456;
    uint8_t *workspaceGM = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(10000001);
    ICPU_RUN_KF(upsample_bicubic2d_grad, 2, inGM, outGM, workspaceGM, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(inGM);
    AscendC::GmFree(outGM);
    AscendC::GmFree(workspaceGM);
    AscendC::GmFree(tiling);
}

TEST_F(upsample_bicubic2d_grad_test, test_case_3)
{
    int b = 2;
    int c = 16;
    int h_in = 32;
    int w_in = 32;
    int h_out = 64;
    int w_out = 64;

    int64_t inSize = b * c * h_in * w_in * sizeof(float);
    int64_t outSize = b * c * h_out * w_out * sizeof(float);

    uint8_t *inGM = (uint8_t *)AscendC::GmAlloc(inSize);
    uint8_t *outGM = (uint8_t *)AscendC::GmAlloc(outSize);

    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(UpsampleBicubic2dGradTilingData));
    UpsampleBicubic2dGradTilingData *tilingDatafromBin = reinterpret_cast<UpsampleBicubic2dGradTilingData *>(tiling);

    tilingDatafromBin->dataType = 0;
    tilingDatafromBin->CoreNum = 20;
    tilingDatafromBin->CoreNumW = 20;
    tilingDatafromBin->CoreNumH = 20;
    tilingDatafromBin->alignCorners = 0;
    tilingDatafromBin->scalesH = 2.000000;
    tilingDatafromBin->scalesW = 2.000000;
    tilingDatafromBin->singleCoreKW = 11;
    tilingDatafromBin->singleCoreKH = 11;
    tilingDatafromBin->needExpandW = 1;
    tilingDatafromBin->needExpandH = 1;
    tilingDatafromBin->batch = 32;
    tilingDatafromBin->inputN = 2;
    tilingDatafromBin->inputC = 16;
    tilingDatafromBin->inputH = 32;
    tilingDatafromBin->inputW = 32;
    tilingDatafromBin->outputH = 64;
    tilingDatafromBin->outputW = 64;
    tilingDatafromBin->tailStartW = 0;
    tilingDatafromBin->tailEndW = 64;
    tilingDatafromBin->tailStartH = 0;
    tilingDatafromBin->tailEndH = 64;
    tilingDatafromBin->slideSize = 16;
    tilingDatafromBin->radioMatrixSize = 176;
    tilingDatafromBin->intermediateMatrixSize = 65536;

    tilingDatafromBin->tailSlideStartListW[0] = 0;
    tilingDatafromBin->tailSlideStartListW[1] = 52;
    tilingDatafromBin->tailSlideStartListW[2] = 104;
    tilingDatafromBin->tailSlideStartListW[3] = 156;
    tilingDatafromBin->tailSlideStartListW[4] = 208;
    tilingDatafromBin->tailSlideStartListW[5] = 259;
    tilingDatafromBin->tailSlideStartListW[6] = 310;
    tilingDatafromBin->tailSlideStartListW[7] = 361;
    tilingDatafromBin->tailSlideStartListW[8] = 412;
    tilingDatafromBin->tailSlideStartListW[9] = 463;
    tilingDatafromBin->tailSlideStartListW[10] = 514;
    tilingDatafromBin->tailSlideStartListW[11] = 565;
    tilingDatafromBin->tailSlideStartListW[12] = 616;
    tilingDatafromBin->tailSlideStartListW[13] = 667;
    tilingDatafromBin->tailSlideStartListW[14] = 718;
    tilingDatafromBin->tailSlideStartListW[15] = 769;
    tilingDatafromBin->tailSlideStartListW[16] = 820;
    tilingDatafromBin->tailSlideStartListW[17] = 871;
    tilingDatafromBin->tailSlideStartListW[18] = 922;
    tilingDatafromBin->tailSlideStartListW[19] = 973;

    tilingDatafromBin->tailSlideEndListW[0] = 52;
    tilingDatafromBin->tailSlideEndListW[1] = 104;
    tilingDatafromBin->tailSlideEndListW[2] = 156;
    tilingDatafromBin->tailSlideEndListW[3] = 208;
    tilingDatafromBin->tailSlideEndListW[4] = 259;
    tilingDatafromBin->tailSlideEndListW[5] = 310;
    tilingDatafromBin->tailSlideEndListW[6] = 361;
    tilingDatafromBin->tailSlideEndListW[7] = 412;
    tilingDatafromBin->tailSlideEndListW[8] = 463;
    tilingDatafromBin->tailSlideEndListW[9] = 514;
    tilingDatafromBin->tailSlideEndListW[10] = 565;
    tilingDatafromBin->tailSlideEndListW[11] = 616;
    tilingDatafromBin->tailSlideEndListW[12] = 667;
    tilingDatafromBin->tailSlideEndListW[13] = 718;
    tilingDatafromBin->tailSlideEndListW[14] = 769;
    tilingDatafromBin->tailSlideEndListW[15] = 820;
    tilingDatafromBin->tailSlideEndListW[16] = 871;
    tilingDatafromBin->tailSlideEndListW[17] = 922;
    tilingDatafromBin->tailSlideEndListW[18] = 973;
    tilingDatafromBin->tailSlideEndListW[19] = 1024;

    tilingDatafromBin->tailSlideStartListH[0] = 0;
    tilingDatafromBin->tailSlideStartListH[1] = 4;
    tilingDatafromBin->tailSlideStartListH[2] = 8;
    tilingDatafromBin->tailSlideStartListH[3] = 12;
    tilingDatafromBin->tailSlideStartListH[4] = 16;
    tilingDatafromBin->tailSlideStartListH[5] = 19;
    tilingDatafromBin->tailSlideStartListH[6] = 22;
    tilingDatafromBin->tailSlideStartListH[7] = 25;
    tilingDatafromBin->tailSlideStartListH[8] = 28;
    tilingDatafromBin->tailSlideStartListH[9] = 31;
    tilingDatafromBin->tailSlideStartListH[10] = 34;
    tilingDatafromBin->tailSlideStartListH[11] = 37;
    tilingDatafromBin->tailSlideStartListH[12] = 40;
    tilingDatafromBin->tailSlideStartListH[13] = 43;
    tilingDatafromBin->tailSlideStartListH[14] = 46;
    tilingDatafromBin->tailSlideStartListH[15] = 49;
    tilingDatafromBin->tailSlideStartListH[16] = 52;
    tilingDatafromBin->tailSlideStartListH[17] = 55;
    tilingDatafromBin->tailSlideStartListH[18] = 58;
    tilingDatafromBin->tailSlideStartListH[19] = 61;

    tilingDatafromBin->tailSlideEndListH[0] = 4;
    tilingDatafromBin->tailSlideEndListH[1] = 8;
    tilingDatafromBin->tailSlideEndListH[2] = 12;
    tilingDatafromBin->tailSlideEndListH[3] = 16;
    tilingDatafromBin->tailSlideEndListH[4] = 19;
    tilingDatafromBin->tailSlideEndListH[5] = 22;
    tilingDatafromBin->tailSlideEndListH[6] = 25;
    tilingDatafromBin->tailSlideEndListH[7] = 28;
    tilingDatafromBin->tailSlideEndListH[8] = 31;
    tilingDatafromBin->tailSlideEndListH[9] = 34;
    tilingDatafromBin->tailSlideEndListH[10] = 37;
    tilingDatafromBin->tailSlideEndListH[11] = 40;
    tilingDatafromBin->tailSlideEndListH[12] = 43;
    tilingDatafromBin->tailSlideEndListH[13] = 46;
    tilingDatafromBin->tailSlideEndListH[14] = 49;
    tilingDatafromBin->tailSlideEndListH[15] = 52;
    tilingDatafromBin->tailSlideEndListH[16] = 55;
    tilingDatafromBin->tailSlideEndListH[17] = 58;
    tilingDatafromBin->tailSlideEndListH[18] = 61;
    tilingDatafromBin->tailSlideEndListH[19] = 64;

    tilingDatafromBin->MMParamH.usedCoreNum = 1;
    tilingDatafromBin->MMParamH.M = 64;
    tilingDatafromBin->MMParamH.N = 64;
    tilingDatafromBin->MMParamH.Ka = 32;
    tilingDatafromBin->MMParamH.Kb = 32;
    tilingDatafromBin->MMParamH.singleCoreM = 16;
    tilingDatafromBin->MMParamH.singleCoreN = 64;
    tilingDatafromBin->MMParamH.singleCoreK = 11;
    tilingDatafromBin->MMParamH.baseM = 16;
    tilingDatafromBin->MMParamH.baseN = 64;
    tilingDatafromBin->MMParamH.baseK = 16;
    tilingDatafromBin->MMParamH.depthA1 = 1;
    tilingDatafromBin->MMParamH.depthB1 = 1;
    tilingDatafromBin->MMParamH.stepM = 1;
    tilingDatafromBin->MMParamH.stepN = 1;
    tilingDatafromBin->MMParamH.isBias = 0;
    tilingDatafromBin->MMParamH.transLength = 0;
    tilingDatafromBin->MMParamH.iterateOrder = 0;
    tilingDatafromBin->MMParamH.shareMode = 0;
    tilingDatafromBin->MMParamH.shareL1Size = 5120;
    tilingDatafromBin->MMParamH.shareL0CSize = 4096;
    tilingDatafromBin->MMParamH.shareUbSize = 0;
    tilingDatafromBin->MMParamH.batchM = 1;
    tilingDatafromBin->MMParamH.batchN = 1;
    tilingDatafromBin->MMParamH.singleBatchM = 1;
    tilingDatafromBin->MMParamH.singleBatchN = 1;
    tilingDatafromBin->MMParamH.stepKa = 1;
    tilingDatafromBin->MMParamH.stepKb = 1;
    tilingDatafromBin->MMParamH.dbL0A = 2;
    tilingDatafromBin->MMParamH.dbL0B = 2;
    tilingDatafromBin->MMParamH.dbL0C = 1;

    tilingDatafromBin->MMParamW.usedCoreNum = 1;
    tilingDatafromBin->MMParamW.M = 1024;
    tilingDatafromBin->MMParamW.N = 64;
    tilingDatafromBin->MMParamW.Ka = 32;
    tilingDatafromBin->MMParamW.Kb = 32;
    tilingDatafromBin->MMParamW.singleCoreM = 1024;
    tilingDatafromBin->MMParamW.singleCoreN = 16;
    tilingDatafromBin->MMParamW.singleCoreK = 11;
    tilingDatafromBin->MMParamW.baseM = 1024;
    tilingDatafromBin->MMParamW.baseN = 16;
    tilingDatafromBin->MMParamW.baseK = 8;
    tilingDatafromBin->MMParamW.depthA1 = 2;
    tilingDatafromBin->MMParamW.depthB1 = 1;
    tilingDatafromBin->MMParamW.stepM = 1;
    tilingDatafromBin->MMParamW.stepN = 1;
    tilingDatafromBin->MMParamH.isBias = 0;
    tilingDatafromBin->MMParamW.transLength = 0;
    tilingDatafromBin->MMParamW.iterateOrder = 0;
    tilingDatafromBin->MMParamW.shareMode = 0;
    tilingDatafromBin->MMParamW.shareL1Size = 66048;
    tilingDatafromBin->MMParamW.shareL0CSize = 65536;
    tilingDatafromBin->MMParamW.shareUbSize = 0;
    tilingDatafromBin->MMParamW.batchM = 1;
    tilingDatafromBin->MMParamW.batchN = 1;
    tilingDatafromBin->MMParamW.singleBatchM = 1;
    tilingDatafromBin->MMParamW.singleBatchN = 1;
    tilingDatafromBin->MMParamW.stepKa = 2;
    tilingDatafromBin->MMParamW.stepKb = 1;
    tilingDatafromBin->MMParamW.dbL0A = 2;
    tilingDatafromBin->MMParamW.dbL0B = 2;
    tilingDatafromBin->MMParamW.dbL0C = 1;

    int64_t workspaceSize = 17053440;
    uint8_t *workspaceGM = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(10000002);
    ICPU_RUN_KF(upsample_bicubic2d_grad, 2, inGM, outGM, workspaceGM, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(inGM);
    AscendC::GmFree(outGM);
    AscendC::GmFree(workspaceGM);
    AscendC::GmFree(tiling);
}

TEST_F(upsample_bicubic2d_grad_test, test_case_4)
{
    int b = 2;
    int c = 16;
    int h_in = 32;
    int w_in = 32;
    int h_out = 64;
    int w_out = 64;

    int64_t inSize = b * c * h_in * w_in * sizeof(half);
    int64_t outSize = b * c * h_out * w_out * sizeof(half);

    uint8_t *inGM = (uint8_t *)AscendC::GmAlloc(inSize);
    uint8_t *outGM = (uint8_t *)AscendC::GmAlloc(outSize);

    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(UpsampleBicubic2dGradTilingData));
    UpsampleBicubic2dGradTilingData *tilingDatafromBin = reinterpret_cast<UpsampleBicubic2dGradTilingData *>(tiling);

    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->CoreNum = 20;
    tilingDatafromBin->CoreNumW = 20;
    tilingDatafromBin->CoreNumH = 20;
    tilingDatafromBin->alignCorners = 0;
    tilingDatafromBin->scalesH = 2.000000;
    tilingDatafromBin->scalesW = 2.000000;
    tilingDatafromBin->singleCoreKW = 11;
    tilingDatafromBin->singleCoreKH = 11;
    tilingDatafromBin->needExpandW = 1;
    tilingDatafromBin->needExpandH = 1;
    tilingDatafromBin->batch = 32;
    tilingDatafromBin->inputN = 2;
    tilingDatafromBin->inputC = 16;
    tilingDatafromBin->inputH = 32;
    tilingDatafromBin->inputW = 32;
    tilingDatafromBin->outputH = 64;
    tilingDatafromBin->outputW = 64;
    tilingDatafromBin->tailStartW = 0;
    tilingDatafromBin->tailEndW = 64;
    tilingDatafromBin->tailStartH = 0;
    tilingDatafromBin->tailEndH = 64;
    tilingDatafromBin->slideSize = 16;
    tilingDatafromBin->radioMatrixSize = 176;
    tilingDatafromBin->intermediateMatrixSize = 65536;

    tilingDatafromBin->tailSlideStartListW[0] = 0;
    tilingDatafromBin->tailSlideStartListW[1] = 52;
    tilingDatafromBin->tailSlideStartListW[2] = 104;
    tilingDatafromBin->tailSlideStartListW[3] = 156;
    tilingDatafromBin->tailSlideStartListW[4] = 208;
    tilingDatafromBin->tailSlideStartListW[5] = 259;
    tilingDatafromBin->tailSlideStartListW[6] = 310;
    tilingDatafromBin->tailSlideStartListW[7] = 361;
    tilingDatafromBin->tailSlideStartListW[8] = 412;
    tilingDatafromBin->tailSlideStartListW[9] = 463;
    tilingDatafromBin->tailSlideStartListW[10] = 514;
    tilingDatafromBin->tailSlideStartListW[11] = 565;
    tilingDatafromBin->tailSlideStartListW[12] = 616;
    tilingDatafromBin->tailSlideStartListW[13] = 667;
    tilingDatafromBin->tailSlideStartListW[14] = 718;
    tilingDatafromBin->tailSlideStartListW[15] = 769;
    tilingDatafromBin->tailSlideStartListW[16] = 820;
    tilingDatafromBin->tailSlideStartListW[17] = 871;
    tilingDatafromBin->tailSlideStartListW[18] = 922;
    tilingDatafromBin->tailSlideStartListW[19] = 973;

    tilingDatafromBin->tailSlideEndListW[0] = 52;
    tilingDatafromBin->tailSlideEndListW[1] = 104;
    tilingDatafromBin->tailSlideEndListW[2] = 156;
    tilingDatafromBin->tailSlideEndListW[3] = 208;
    tilingDatafromBin->tailSlideEndListW[4] = 259;
    tilingDatafromBin->tailSlideEndListW[5] = 310;
    tilingDatafromBin->tailSlideEndListW[6] = 361;
    tilingDatafromBin->tailSlideEndListW[7] = 412;
    tilingDatafromBin->tailSlideEndListW[8] = 463;
    tilingDatafromBin->tailSlideEndListW[9] = 514;
    tilingDatafromBin->tailSlideEndListW[10] = 565;
    tilingDatafromBin->tailSlideEndListW[11] = 616;
    tilingDatafromBin->tailSlideEndListW[12] = 667;
    tilingDatafromBin->tailSlideEndListW[13] = 718;
    tilingDatafromBin->tailSlideEndListW[14] = 769;
    tilingDatafromBin->tailSlideEndListW[15] = 820;
    tilingDatafromBin->tailSlideEndListW[16] = 871;
    tilingDatafromBin->tailSlideEndListW[17] = 922;
    tilingDatafromBin->tailSlideEndListW[18] = 973;
    tilingDatafromBin->tailSlideEndListW[19] = 1024;

    tilingDatafromBin->tailSlideStartListH[0] = 0;
    tilingDatafromBin->tailSlideStartListH[1] = 4;
    tilingDatafromBin->tailSlideStartListH[2] = 8;
    tilingDatafromBin->tailSlideStartListH[3] = 12;
    tilingDatafromBin->tailSlideStartListH[4] = 16;
    tilingDatafromBin->tailSlideStartListH[5] = 19;
    tilingDatafromBin->tailSlideStartListH[6] = 22;
    tilingDatafromBin->tailSlideStartListH[7] = 25;
    tilingDatafromBin->tailSlideStartListH[8] = 28;
    tilingDatafromBin->tailSlideStartListH[9] = 31;
    tilingDatafromBin->tailSlideStartListH[10] = 34;
    tilingDatafromBin->tailSlideStartListH[11] = 37;
    tilingDatafromBin->tailSlideStartListH[12] = 40;
    tilingDatafromBin->tailSlideStartListH[13] = 43;
    tilingDatafromBin->tailSlideStartListH[14] = 46;
    tilingDatafromBin->tailSlideStartListH[15] = 49;
    tilingDatafromBin->tailSlideStartListH[16] = 52;
    tilingDatafromBin->tailSlideStartListH[17] = 55;
    tilingDatafromBin->tailSlideStartListH[18] = 58;
    tilingDatafromBin->tailSlideStartListH[19] = 61;

    tilingDatafromBin->tailSlideEndListH[0] = 4;
    tilingDatafromBin->tailSlideEndListH[1] = 8;
    tilingDatafromBin->tailSlideEndListH[2] = 12;
    tilingDatafromBin->tailSlideEndListH[3] = 16;
    tilingDatafromBin->tailSlideEndListH[4] = 19;
    tilingDatafromBin->tailSlideEndListH[5] = 22;
    tilingDatafromBin->tailSlideEndListH[6] = 25;
    tilingDatafromBin->tailSlideEndListH[7] = 28;
    tilingDatafromBin->tailSlideEndListH[8] = 31;
    tilingDatafromBin->tailSlideEndListH[9] = 34;
    tilingDatafromBin->tailSlideEndListH[10] = 37;
    tilingDatafromBin->tailSlideEndListH[11] = 40;
    tilingDatafromBin->tailSlideEndListH[12] = 43;
    tilingDatafromBin->tailSlideEndListH[13] = 46;
    tilingDatafromBin->tailSlideEndListH[14] = 49;
    tilingDatafromBin->tailSlideEndListH[15] = 52;
    tilingDatafromBin->tailSlideEndListH[16] = 55;
    tilingDatafromBin->tailSlideEndListH[17] = 58;
    tilingDatafromBin->tailSlideEndListH[18] = 61;
    tilingDatafromBin->tailSlideEndListH[19] = 64;

    tilingDatafromBin->MMParamH.usedCoreNum = 1;
    tilingDatafromBin->MMParamH.M = 64;
    tilingDatafromBin->MMParamH.N = 64;
    tilingDatafromBin->MMParamH.Ka = 32;
    tilingDatafromBin->MMParamH.Kb = 32;
    tilingDatafromBin->MMParamH.singleCoreM = 16;
    tilingDatafromBin->MMParamH.singleCoreN = 64;
    tilingDatafromBin->MMParamH.singleCoreK = 11;
    tilingDatafromBin->MMParamH.baseM = 16;
    tilingDatafromBin->MMParamH.baseN = 64;
    tilingDatafromBin->MMParamH.baseK = 16;
    tilingDatafromBin->MMParamH.depthA1 = 1;
    tilingDatafromBin->MMParamH.depthB1 = 1;
    tilingDatafromBin->MMParamH.stepM = 1;
    tilingDatafromBin->MMParamH.stepN = 1;
    tilingDatafromBin->MMParamH.isBias = 0;
    tilingDatafromBin->MMParamH.transLength = 0;
    tilingDatafromBin->MMParamH.iterateOrder = 0;
    tilingDatafromBin->MMParamH.shareMode = 0;
    tilingDatafromBin->MMParamH.shareL1Size = 5120;
    tilingDatafromBin->MMParamH.shareL0CSize = 4096;
    tilingDatafromBin->MMParamH.shareUbSize = 0;
    tilingDatafromBin->MMParamH.batchM = 1;
    tilingDatafromBin->MMParamH.batchN = 1;
    tilingDatafromBin->MMParamH.singleBatchM = 1;
    tilingDatafromBin->MMParamH.singleBatchN = 1;
    tilingDatafromBin->MMParamH.stepKa = 1;
    tilingDatafromBin->MMParamH.stepKb = 1;
    tilingDatafromBin->MMParamH.dbL0A = 2;
    tilingDatafromBin->MMParamH.dbL0B = 2;
    tilingDatafromBin->MMParamH.dbL0C = 1;

    tilingDatafromBin->MMParamW.usedCoreNum = 1;
    tilingDatafromBin->MMParamW.M = 1024;
    tilingDatafromBin->MMParamW.N = 64;
    tilingDatafromBin->MMParamW.Ka = 32;
    tilingDatafromBin->MMParamW.Kb = 32;
    tilingDatafromBin->MMParamW.singleCoreM = 1024;
    tilingDatafromBin->MMParamW.singleCoreN = 16;
    tilingDatafromBin->MMParamW.singleCoreK = 11;
    tilingDatafromBin->MMParamW.baseM = 1024;
    tilingDatafromBin->MMParamW.baseN = 16;
    tilingDatafromBin->MMParamW.baseK = 8;
    tilingDatafromBin->MMParamW.depthA1 = 2;
    tilingDatafromBin->MMParamW.depthB1 = 1;
    tilingDatafromBin->MMParamW.stepM = 1;
    tilingDatafromBin->MMParamW.stepN = 1;
    tilingDatafromBin->MMParamH.isBias = 0;
    tilingDatafromBin->MMParamW.transLength = 0;
    tilingDatafromBin->MMParamW.iterateOrder = 0;
    tilingDatafromBin->MMParamW.shareMode = 0;
    tilingDatafromBin->MMParamW.shareL1Size = 66048;
    tilingDatafromBin->MMParamW.shareL0CSize = 65536;
    tilingDatafromBin->MMParamW.shareUbSize = 0;
    tilingDatafromBin->MMParamW.batchM = 1;
    tilingDatafromBin->MMParamW.batchN = 1;
    tilingDatafromBin->MMParamW.singleBatchM = 1;
    tilingDatafromBin->MMParamW.singleBatchN = 1;
    tilingDatafromBin->MMParamW.stepKa = 2;
    tilingDatafromBin->MMParamW.stepKb = 1;
    tilingDatafromBin->MMParamW.dbL0A = 2;
    tilingDatafromBin->MMParamW.dbL0B = 2;
    tilingDatafromBin->MMParamW.dbL0C = 1;

    int64_t workspaceSize = 17053440;
    uint8_t *workspaceGM = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(10000002);
    ICPU_RUN_KF(upsample_bicubic2d_grad, 2, inGM, outGM, workspaceGM, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(inGM);
    AscendC::GmFree(outGM);
    AscendC::GmFree(workspaceGM);
    AscendC::GmFree(tiling);
}
