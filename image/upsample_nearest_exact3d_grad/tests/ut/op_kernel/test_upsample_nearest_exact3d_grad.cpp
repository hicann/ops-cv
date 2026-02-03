/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_upsample_nearest_exact3d_grad.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

extern "C" __global__ __aicore__ void upsample_nearest_exact3d_grad(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class upsample_nearest_exact3d_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "upsample_nearest_exact3d_grad_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "upsample_nearest_exact3d_grad_test TearDown\n" << std::endl;
    }
};

TEST_F(upsample_nearest_exact3d_grad_test, test_case_float_1)
{
    system(
        "cp -rf "
        "../../../../image/upsample_nearest_exact3d_grad/tests/ut/op_kernel/"
        "upsample_nearest_exact3d_grad_data ./");
    system("chmod -R 755 ./upsample_nearest_exact3d_grad_data/");
    system(
        "cd ./upsample_nearest_exact3d_grad_data/ && python3 gen_data.py '(1, 1, 4, 4, 4)' '(16, 16, 16)' 'float32'");

    size_t gradInputByteSize = 4 * 4 * 4 * sizeof(float);
    size_t gradOutputByteSize = 16 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleNearest3dGradTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 4;

    uint8_t* gradInput = (uint8_t*)AscendC::GmAlloc(gradInputByteSize);
    uint8_t* gradOutput = (uint8_t*)AscendC::GmAlloc(gradOutputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_nearest_exact3d_grad_data/float32_input_upsample_nearest_exact3d_grad.bin";

    ReadFile(fileName, gradOutputByteSize, gradOutput, gradOutputByteSize);

    UpsampleNearest3dGradTilingData* tilingDatafromBin = reinterpret_cast<UpsampleNearest3dGradTilingData*>(tiling);

    tilingDatafromBin->dataType = 2;
    tilingDatafromBin->batches = 1;
    tilingDatafromBin->scaleW = 4;
    tilingDatafromBin->scaleH = 4;
    tilingDatafromBin->scaleD = 4;
    tilingDatafromBin->needResizeH = true;
    tilingDatafromBin->needResizeW = true;
    tilingDatafromBin->needResizeD = true;

    tilingDatafromBin->slideSize = 64;
    tilingDatafromBin->tensorSize = 128;
    tilingDatafromBin->tensorSizeMapping = 320;
    tilingDatafromBin->radioMatrixSize = 1024;
    tilingDatafromBin->intermediateMatrixSizeW = 1024;
    tilingDatafromBin->intermediateMatrixSizeH = 256;

    tilingDatafromBin->gradInputShapes[0] = 4;
    tilingDatafromBin->gradInputShapes[1] = 4;
    tilingDatafromBin->gradInputShapes[2] = 4;
    tilingDatafromBin->gradOutputShapes[0] = 16;
    tilingDatafromBin->gradOutputShapes[1] = 16;
    tilingDatafromBin->gradOutputShapes[2] = 16;

    tilingDatafromBin->eachCoreSlideNums[0] = 0;
    tilingDatafromBin->remainders[0] = 1;
    tilingDatafromBin->tailStartSlideNums[0] = 0;
    tilingDatafromBin->groupCoreNums[0] = 1;
    tilingDatafromBin->inputRows[0] = 16;
    tilingDatafromBin->tailAvergingRows[0] = 64;
    tilingDatafromBin->needCoreNums[0] = 1;

    tilingDatafromBin->eachCoreSlideNums[1] = 0;
    tilingDatafromBin->remainders[1] = 1;
    tilingDatafromBin->tailStartSlideNums[1] = 0;
    tilingDatafromBin->groupCoreNums[1] = 1;
    tilingDatafromBin->inputRows[1] = 16;
    tilingDatafromBin->tailAvergingRows[1] = 64;
    tilingDatafromBin->needCoreNums[1] = 1;

    tilingDatafromBin->eachCoreSlideNums[2] = 0;
    tilingDatafromBin->remainders[2] = 1;
    tilingDatafromBin->tailStartSlideNums[2] = 0;
    tilingDatafromBin->groupCoreNums[2] = 4;
    tilingDatafromBin->inputRows[2] = 256;
    tilingDatafromBin->tailAvergingRows[2] = 64;
    tilingDatafromBin->needCoreNums[2] = 4;

    tilingDatafromBin->matmulTilingW.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingW.M = 256;
    tilingDatafromBin->matmulTilingW.N = 4;
    tilingDatafromBin->matmulTilingW.Ka = 16;
    tilingDatafromBin->matmulTilingW.Kb = 16;
    tilingDatafromBin->matmulTilingW.singleCoreM = 256;
    tilingDatafromBin->matmulTilingW.singleCoreN = 64;
    tilingDatafromBin->matmulTilingW.singleCoreK = 16;
    tilingDatafromBin->matmulTilingW.baseM = 256;
    tilingDatafromBin->matmulTilingW.baseN = 64;
    tilingDatafromBin->matmulTilingW.baseK = 16;
    tilingDatafromBin->matmulTilingW.depthA1 = 1;
    tilingDatafromBin->matmulTilingW.depthB1 = 1;
    tilingDatafromBin->matmulTilingW.stepM = 1;
    tilingDatafromBin->matmulTilingW.stepN = 1;
    tilingDatafromBin->matmulTilingW.stepKa = 1;
    tilingDatafromBin->matmulTilingW.stepKb = 1;
    tilingDatafromBin->matmulTilingW.isBias = 0;
    tilingDatafromBin->matmulTilingW.transLength = 0;
    tilingDatafromBin->matmulTilingW.iterateOrder = 0;
    tilingDatafromBin->matmulTilingW.shareMode = 0;
    tilingDatafromBin->matmulTilingW.shareL1Size = 20480;
    tilingDatafromBin->matmulTilingW.shareL0CSize = 65536;
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
    tilingDatafromBin->matmulTilingH.singleCoreM = 64;
    tilingDatafromBin->matmulTilingH.singleCoreN = 4;
    tilingDatafromBin->matmulTilingH.singleCoreK = 16;
    tilingDatafromBin->matmulTilingH.baseM = 64;
    tilingDatafromBin->matmulTilingH.baseN = 16;
    tilingDatafromBin->matmulTilingH.baseK = 16;
    tilingDatafromBin->matmulTilingH.depthA1 = 1;
    tilingDatafromBin->matmulTilingH.depthB1 = 1;
    tilingDatafromBin->matmulTilingH.stepM = 1;
    tilingDatafromBin->matmulTilingH.stepN = 1;
    tilingDatafromBin->matmulTilingH.stepKa = 1;
    tilingDatafromBin->matmulTilingH.stepKb = 1;
    tilingDatafromBin->matmulTilingH.isBias = 0;
    tilingDatafromBin->matmulTilingH.transLength = 0;
    tilingDatafromBin->matmulTilingH.iterateOrder = 0;
    tilingDatafromBin->matmulTilingH.shareMode = 0;
    tilingDatafromBin->matmulTilingH.shareL1Size = 5120;
    tilingDatafromBin->matmulTilingH.shareL0CSize = 4096;
    tilingDatafromBin->matmulTilingH.shareUbSize = 0;
    tilingDatafromBin->matmulTilingH.batchM = 1;
    tilingDatafromBin->matmulTilingH.batchN = 1;
    tilingDatafromBin->matmulTilingH.singleBatchM = 1;
    tilingDatafromBin->matmulTilingH.singleBatchN = 1;

    tilingDatafromBin->matmulTilingD.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingD.M = 4;
    tilingDatafromBin->matmulTilingD.N = 16;
    tilingDatafromBin->matmulTilingD.Ka = 16;
    tilingDatafromBin->matmulTilingD.Kb = 16;
    tilingDatafromBin->matmulTilingD.singleCoreM = 64;
    tilingDatafromBin->matmulTilingD.singleCoreN = 16;
    tilingDatafromBin->matmulTilingD.singleCoreK = 16;
    tilingDatafromBin->matmulTilingD.baseM = 64;
    tilingDatafromBin->matmulTilingD.baseN = 16;
    tilingDatafromBin->matmulTilingD.baseK = 16;
    tilingDatafromBin->matmulTilingD.depthA1 = 1;
    tilingDatafromBin->matmulTilingD.depthB1 = 1;
    tilingDatafromBin->matmulTilingD.stepM = 1;
    tilingDatafromBin->matmulTilingD.stepN = 1;
    tilingDatafromBin->matmulTilingD.stepKa = 1;
    tilingDatafromBin->matmulTilingD.stepKb = 1;
    tilingDatafromBin->matmulTilingD.isBias = 0;
    tilingDatafromBin->matmulTilingD.transLength = 0;
    tilingDatafromBin->matmulTilingD.iterateOrder = 0;
    tilingDatafromBin->matmulTilingD.shareMode = 0;
    tilingDatafromBin->matmulTilingD.shareL1Size = 5120;
    tilingDatafromBin->matmulTilingD.shareL0CSize = 4096;
    tilingDatafromBin->matmulTilingD.shareUbSize = 0;
    tilingDatafromBin->matmulTilingD.batchM = 1;
    tilingDatafromBin->matmulTilingD.batchN = 1;
    tilingDatafromBin->matmulTilingD.singleBatchM = 1;
    tilingDatafromBin->matmulTilingD.singleBatchN = 1;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(upsample_nearest_exact3d_grad, numBlocks, gradOutput, gradInput, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./upsample_nearest_exact3d_grad_data/float32_output_upsample_nearest_exact3d_grad.bin";
    WriteFile(fileName, gradInput, gradInputByteSize);

    AscendC::GmFree((void*)(gradInput));
    AscendC::GmFree((void*)(gradOutput));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./upsample_nearest_exact3d_grad_data/ && python3 compare_data.py 'float32'");
}
