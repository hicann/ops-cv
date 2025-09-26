/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_upsample_bilinear2d_aa_backward_backward.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../data_utils.h"

extern "C" __global__ __aicore__ void upsample_bilinear2d_aa_backward(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class upsample_bilinear2d_aa_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "upsample_bilinear2d_aa_backward_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "upsample_bilinear2d_aa_backward_test TearDown\n" << std::endl;
    }
};

TEST_F(upsample_bilinear2d_aa_backward_test, test_case_float_1)
{
    system("cp -rf "
           "../../../../../../../ops/image/upsample_bilinear2d_aa_backward/tests/ut/op_kernel/"
           "upsample_bilinear2d_aa_backward_data ./");
    system("chmod -R 755 ./upsample_bilinear2d_aa_backward_data/");
    system("cd ./upsample_bilinear2d_aa_backward_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)'  'float32'");

    size_t inputByteSize = 16 * 16 * sizeof(float);
    size_t outputByteSize = 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleBilinear2dAABackwardTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_bilinear2d_aa_backward_data/float32_input_upsample_bilinear2d_aa_backward.bin";
    ;
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleBilinear2dAABackwardTilingData *tilingDatafromBin =
        reinterpret_cast<UpsampleBilinear2dAABackwardTilingData *>(tiling);

    tilingDatafromBin->slideSize = 16;
    tilingDatafromBin->dataType = 2;
    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.25;
    tilingDatafromBin->invscaleW = 1.0;
    tilingDatafromBin->invscaleH = 1.0;
    tilingDatafromBin->supportW = 1;
    tilingDatafromBin->supportH = 1;
    tilingDatafromBin->maxInterpSizeW = 3;
    tilingDatafromBin->maxInterpSizeH = 3;
    tilingDatafromBin->radioMatrixSizeW = 256;
    tilingDatafromBin->radioMatrixSizeH = 256;
    tilingDatafromBin->needCoreNumW = 1;
    tilingDatafromBin->needCoreNumH = 1;
    tilingDatafromBin->needResizeH = true;
    tilingDatafromBin->needResizeW = true;
    tilingDatafromBin->intermediateMatrixSize = 256;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 1;
    tilingDatafromBin->inputShapes[2] = 16;
    tilingDatafromBin->inputShapes[3] = 16;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 1;
    tilingDatafromBin->outputShapes[2] = 4;
    tilingDatafromBin->outputShapes[3] = 4;
    tilingDatafromBin->tailSlideStartListW[0] = 0;
    tilingDatafromBin->tailSlideEndListW[0] = 4;
    tilingDatafromBin->tailRowStartListW[0] = 0;
    tilingDatafromBin->tailRowEndListW[0] = 16;
    tilingDatafromBin->tailSlideStartListH[0] = 0;
    tilingDatafromBin->tailSlideEndListH[0] = 4;
    tilingDatafromBin->tailRowStartListH[0] = 0;
    tilingDatafromBin->tailRowEndListH[0] = 4;

    tilingDatafromBin->matmulTilingW.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingW.M = 16;
    tilingDatafromBin->matmulTilingW.N = 16;
    tilingDatafromBin->matmulTilingW.Ka = 16;
    tilingDatafromBin->matmulTilingW.Kb = 16;
    tilingDatafromBin->matmulTilingW.singleCoreM = 16;
    tilingDatafromBin->matmulTilingW.singleCoreN = 16;
    tilingDatafromBin->matmulTilingW.singleCoreK = 16;
    tilingDatafromBin->matmulTilingW.baseM = 16;
    tilingDatafromBin->matmulTilingW.baseN = 16;
    tilingDatafromBin->matmulTilingW.baseK = 16;
    tilingDatafromBin->matmulTilingW.depthA1 = 1;
    tilingDatafromBin->matmulTilingW.depthB1 = 1;
    tilingDatafromBin->matmulTilingW.stepM = 1;
    tilingDatafromBin->matmulTilingW.stepN = 1;
    tilingDatafromBin->matmulTilingW.isBias = 0;
    tilingDatafromBin->matmulTilingW.transLength = 0;
    tilingDatafromBin->matmulTilingW.iterateOrder = 0;
    tilingDatafromBin->matmulTilingW.shareMode = 0;
    tilingDatafromBin->matmulTilingW.shareL1Size = 1024;
    tilingDatafromBin->matmulTilingW.shareL0CSize = 1024;
    tilingDatafromBin->matmulTilingW.shareUbSize = 0;
    tilingDatafromBin->matmulTilingW.batchM = 1;
    tilingDatafromBin->matmulTilingW.batchN = 1;
    tilingDatafromBin->matmulTilingW.singleBatchM = 1;
    tilingDatafromBin->matmulTilingW.singleBatchN = 1;

    tilingDatafromBin->matmulTilingH.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingH.M = 4;
    tilingDatafromBin->matmulTilingH.N = 4;
    tilingDatafromBin->matmulTilingH.Ka = 16;
    tilingDatafromBin->matmulTilingH.Kb = 16;
    tilingDatafromBin->matmulTilingH.singleCoreM = 16;
    tilingDatafromBin->matmulTilingH.singleCoreN = 4;
    tilingDatafromBin->matmulTilingH.singleCoreK = 16;
    tilingDatafromBin->matmulTilingH.baseM = 16;
    tilingDatafromBin->matmulTilingH.baseN = 16;
    tilingDatafromBin->matmulTilingH.baseK = 16;
    tilingDatafromBin->matmulTilingH.depthA1 = 1;
    tilingDatafromBin->matmulTilingH.depthB1 = 1;
    tilingDatafromBin->matmulTilingH.stepM = 1;
    tilingDatafromBin->matmulTilingH.stepN = 1;
    tilingDatafromBin->matmulTilingH.isBias = 0;
    tilingDatafromBin->matmulTilingH.transLength = 0;
    tilingDatafromBin->matmulTilingH.iterateOrder = 0;
    tilingDatafromBin->matmulTilingH.shareMode = 0;
    tilingDatafromBin->matmulTilingH.shareL1Size = 1024;
    tilingDatafromBin->matmulTilingH.shareL0CSize = 1024;
    tilingDatafromBin->matmulTilingH.shareUbSize = 0;
    tilingDatafromBin->matmulTilingH.batchM = 1;
    tilingDatafromBin->matmulTilingH.batchN = 1;
    tilingDatafromBin->matmulTilingH.singleBatchM = 1;
    tilingDatafromBin->matmulTilingH.singleBatchN = 1;

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(upsample_bilinear2d_aa_backward, blockDim, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./upsample_bilinear2d_aa_backward_data/float32_output_upsample_bilinear2d_aa_backward.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bilinear2d_aa_backward_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_bilinear2d_aa_backward_test, test_case_float16_2)
{
    system("cp -rf "
           "../../../../../../../ops/image/upsample_bilinear2d_aa_backward/tests/ut/op_kernel/"
           "upsample_bilinear2d_aa_backward_data ./");
    system("chmod -R 755 ./upsample_bilinear2d_aa_backward_data/");
    system("cd ./upsample_bilinear2d_aa_backward_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)'  'float16'");

    size_t inputByteSize = 16 * 16 * sizeof(half);
    size_t outputByteSize = 4 * 4 * sizeof(half);
    size_t tiling_data_size = sizeof(UpsampleBilinear2dAABackwardTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_bilinear2d_aa_backward_data/float16_input_upsample_bilinear2d_aa_backward.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleBilinear2dAABackwardTilingData *tilingDatafromBin =
        reinterpret_cast<UpsampleBilinear2dAABackwardTilingData *>(tiling);

    tilingDatafromBin->slideSize = 16;
    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.25;
    tilingDatafromBin->invscaleW = 1.0;
    tilingDatafromBin->invscaleH = 1.0;
    tilingDatafromBin->supportW = 1;
    tilingDatafromBin->supportH = 1;
    tilingDatafromBin->maxInterpSizeW = 3;
    tilingDatafromBin->maxInterpSizeH = 3;
    tilingDatafromBin->radioMatrixSizeW = 256;
    tilingDatafromBin->radioMatrixSizeH = 256;
    tilingDatafromBin->needCoreNumW = 1;
    tilingDatafromBin->needCoreNumH = 1;
    tilingDatafromBin->needResizeH = true;
    tilingDatafromBin->needResizeW = true;
    tilingDatafromBin->intermediateMatrixSize = 128;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 1;
    tilingDatafromBin->inputShapes[2] = 16;
    tilingDatafromBin->inputShapes[3] = 16;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 1;
    tilingDatafromBin->outputShapes[2] = 4;
    tilingDatafromBin->outputShapes[3] = 4;
    tilingDatafromBin->tailSlideStartListW[0] = 0;
    tilingDatafromBin->tailSlideEndListW[0] = 4;
    tilingDatafromBin->tailRowStartListW[0] = 0;
    tilingDatafromBin->tailRowEndListW[0] = 16;
    tilingDatafromBin->tailSlideStartListH[0] = 0;
    tilingDatafromBin->tailSlideEndListH[0] = 4;
    tilingDatafromBin->tailRowStartListH[0] = 0;
    tilingDatafromBin->tailRowEndListH[0] = 4;

    tilingDatafromBin->matmulTilingW.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingW.M = 16;
    tilingDatafromBin->matmulTilingW.N = 16;
    tilingDatafromBin->matmulTilingW.Ka = 16;
    tilingDatafromBin->matmulTilingW.Kb = 16;
    tilingDatafromBin->matmulTilingW.singleCoreM = 16;
    tilingDatafromBin->matmulTilingW.singleCoreN = 16;
    tilingDatafromBin->matmulTilingW.singleCoreK = 16;
    tilingDatafromBin->matmulTilingW.baseM = 16;
    tilingDatafromBin->matmulTilingW.baseN = 16;
    tilingDatafromBin->matmulTilingW.baseK = 16;
    tilingDatafromBin->matmulTilingW.depthA1 = 1;
    tilingDatafromBin->matmulTilingW.depthB1 = 1;
    tilingDatafromBin->matmulTilingW.stepM = 1;
    tilingDatafromBin->matmulTilingW.stepN = 1;
    tilingDatafromBin->matmulTilingW.isBias = 0;
    tilingDatafromBin->matmulTilingW.transLength = 0;
    tilingDatafromBin->matmulTilingW.iterateOrder = 0;
    tilingDatafromBin->matmulTilingW.shareMode = 0;
    tilingDatafromBin->matmulTilingW.shareL1Size = 1024;
    tilingDatafromBin->matmulTilingW.shareL0CSize = 1024;
    tilingDatafromBin->matmulTilingW.shareUbSize = 0;
    tilingDatafromBin->matmulTilingW.batchM = 1;
    tilingDatafromBin->matmulTilingW.batchN = 1;
    tilingDatafromBin->matmulTilingW.singleBatchM = 1;
    tilingDatafromBin->matmulTilingW.singleBatchN = 1;

    tilingDatafromBin->matmulTilingH.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingH.M = 4;
    tilingDatafromBin->matmulTilingH.N = 4;
    tilingDatafromBin->matmulTilingH.Ka = 16;
    tilingDatafromBin->matmulTilingH.Kb = 16;
    tilingDatafromBin->matmulTilingH.singleCoreM = 16;
    tilingDatafromBin->matmulTilingH.singleCoreN = 4;
    tilingDatafromBin->matmulTilingH.singleCoreK = 16;
    tilingDatafromBin->matmulTilingH.baseM = 16;
    tilingDatafromBin->matmulTilingH.baseN = 16;
    tilingDatafromBin->matmulTilingH.baseK = 16;
    tilingDatafromBin->matmulTilingH.depthA1 = 1;
    tilingDatafromBin->matmulTilingH.depthB1 = 1;
    tilingDatafromBin->matmulTilingH.stepM = 1;
    tilingDatafromBin->matmulTilingH.stepN = 1;
    tilingDatafromBin->matmulTilingH.isBias = 0;
    tilingDatafromBin->matmulTilingH.transLength = 0;
    tilingDatafromBin->matmulTilingH.iterateOrder = 0;
    tilingDatafromBin->matmulTilingH.shareMode = 0;
    tilingDatafromBin->matmulTilingH.shareL1Size = 1024;
    tilingDatafromBin->matmulTilingH.shareL0CSize = 1024;
    tilingDatafromBin->matmulTilingH.shareUbSize = 0;
    tilingDatafromBin->matmulTilingH.batchM = 1;
    tilingDatafromBin->matmulTilingH.batchN = 1;
    tilingDatafromBin->matmulTilingH.singleBatchM = 1;
    tilingDatafromBin->matmulTilingH.singleBatchN = 1;

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(upsample_bilinear2d_aa_backward, blockDim, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./upsample_bilinear2d_aa_backward_data/float16_output_upsample_bilinear2d_aa_backward.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bilinear2d_aa_backward_data/ && python3 compare_data.py 'float16'");
}
