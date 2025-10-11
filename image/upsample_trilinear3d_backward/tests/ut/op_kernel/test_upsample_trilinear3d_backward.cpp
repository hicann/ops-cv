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
 * \file test_upsample_trilinear3d_backward.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "upsample_trilinear3d_backward_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void upsample_trilinear3d_backward(
    GM_ADDR grad_output, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling);

class UpsampleTrilinear3dBackwardTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "UpsampleTrilinear3dBackwardTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "UpsampleTrilinear3dBackwardTest TearDown\n" << endl;
    }
};

TEST_F(UpsampleTrilinear3dBackwardTest, test_case_float32)
{
    system(
        "cp -rf "
        "../../../../image/upsample_trilinear3d_backward/tests/ut/op_kernel/"
        "upsample_trilinear3d_backward_data ./");
    system("chmod -R 755 ./upsample_trilinear3d_backward_data/");
    system("cd ./upsample_trilinear3d_backward_data/ && python3 gen_data.py '(1, 2, 2, 4, 4)' '(8, 8, 16)' 'float32'");

    size_t grad_outputByteSize = 2 * 8 * 8 * 16 * sizeof(float);
    size_t grad_inputByteSize = 2 * 2 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleTrilinear3dBackwardTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(grad_outputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(grad_inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_trilinear3d_backward_data/float32_grad_output_trilinear3d_grad.bin";
    ReadFile(fileName, grad_outputByteSize, x, grad_outputByteSize);

    UpsampleTrilinear3dBackwardTilingData* tilingDatafromBin =
        reinterpret_cast<UpsampleTrilinear3dBackwardTilingData*>(tiling);

    tilingDatafromBin->dataType = 2;
    tilingDatafromBin->batches = 2;
    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.5;
    tilingDatafromBin->scaleD = 0.25;
    tilingDatafromBin->alignCorners = false;
    tilingDatafromBin->needResizeH = true;
    tilingDatafromBin->needResizeW = true;
    tilingDatafromBin->needResizeD = true;

    tilingDatafromBin->slideSize = 16;
    tilingDatafromBin->radioMatrixSize = 256;
    tilingDatafromBin->intermediateMatrixSizeW = 512;
    tilingDatafromBin->intermediateMatrixSizeH = 256;

    tilingDatafromBin->inputShapes[0] = 2;
    tilingDatafromBin->inputShapes[1] = 4;
    tilingDatafromBin->inputShapes[2] = 4;
    tilingDatafromBin->outputShapes[0] = 8;
    tilingDatafromBin->outputShapes[1] = 8;
    tilingDatafromBin->outputShapes[2] = 16;

    tilingDatafromBin->eachCoreSlideNums[0] = 0;
    tilingDatafromBin->remainders[0] = 1;
    tilingDatafromBin->tailStartSlideNums[0] = 0;
    tilingDatafromBin->groupCoreNums[0] = 1;
    tilingDatafromBin->inputRows[0] = 2;
    tilingDatafromBin->tailAvergingRows[0] = 16;
    tilingDatafromBin->needCoreNums[0] = 1;

    tilingDatafromBin->eachCoreSlideNums[1] = 0;
    tilingDatafromBin->remainders[1] = 1;
    tilingDatafromBin->tailStartSlideNums[1] = 0;
    tilingDatafromBin->groupCoreNums[1] = 1;
    tilingDatafromBin->inputRows[1] = 16;
    tilingDatafromBin->tailAvergingRows[1] = 16;
    tilingDatafromBin->needCoreNums[1] = 1;

    tilingDatafromBin->eachCoreSlideNums[2] = 0;
    tilingDatafromBin->remainders[2] = 1;
    tilingDatafromBin->tailStartSlideNums[2] = 0;
    tilingDatafromBin->groupCoreNums[2] = 8;
    tilingDatafromBin->inputRows[2] = 128;
    tilingDatafromBin->tailAvergingRows[2] = 16;
    tilingDatafromBin->needCoreNums[2] = 8;

    tilingDatafromBin->matmulTilingW.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingW.M = 128;
    tilingDatafromBin->matmulTilingW.N = 4;
    tilingDatafromBin->matmulTilingW.Ka = 16;
    tilingDatafromBin->matmulTilingW.Kb = 16;
    tilingDatafromBin->matmulTilingW.singleCoreM = 128;
    tilingDatafromBin->matmulTilingW.singleCoreN = 16;
    tilingDatafromBin->matmulTilingW.singleCoreK = 16;
    tilingDatafromBin->matmulTilingW.baseM = 128;
    tilingDatafromBin->matmulTilingW.baseN = 16;
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
    tilingDatafromBin->matmulTilingW.shareL1Size = 9216;
    tilingDatafromBin->matmulTilingW.shareL0CSize = 8192;
    tilingDatafromBin->matmulTilingW.shareUbSize = 0;
    tilingDatafromBin->matmulTilingW.batchM = 1;
    tilingDatafromBin->matmulTilingW.batchN = 1;
    tilingDatafromBin->matmulTilingW.singleBatchM = 1;
    tilingDatafromBin->matmulTilingW.singleBatchN = 1;

    tilingDatafromBin->matmulTilingH.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingH.M = 4;
    tilingDatafromBin->matmulTilingH.N = 4;
    tilingDatafromBin->matmulTilingH.Ka = 8;
    tilingDatafromBin->matmulTilingH.Kb = 8;
    tilingDatafromBin->matmulTilingH.singleCoreM = 16;
    tilingDatafromBin->matmulTilingH.singleCoreN = 4;
    tilingDatafromBin->matmulTilingH.singleCoreK = 8;
    tilingDatafromBin->matmulTilingH.baseM = 16;
    tilingDatafromBin->matmulTilingH.baseN = 16;
    tilingDatafromBin->matmulTilingH.baseK = 8;
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
    tilingDatafromBin->matmulTilingH.shareL1Size = 1024;
    tilingDatafromBin->matmulTilingH.shareL0CSize = 1024;
    tilingDatafromBin->matmulTilingH.shareUbSize = 0;
    tilingDatafromBin->matmulTilingH.batchM = 1;
    tilingDatafromBin->matmulTilingH.batchN = 1;
    tilingDatafromBin->matmulTilingH.singleBatchM = 1;
    tilingDatafromBin->matmulTilingH.singleBatchN = 1;

    tilingDatafromBin->matmulTilingD.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingD.M = 2;
    tilingDatafromBin->matmulTilingD.N = 16;
    tilingDatafromBin->matmulTilingD.Ka = 8;
    tilingDatafromBin->matmulTilingD.Kb = 8;
    tilingDatafromBin->matmulTilingD.singleCoreM = 16;
    tilingDatafromBin->matmulTilingD.singleCoreN = 16;
    tilingDatafromBin->matmulTilingD.singleCoreK = 8;
    tilingDatafromBin->matmulTilingD.baseM = 16;
    tilingDatafromBin->matmulTilingD.baseN = 16;
    tilingDatafromBin->matmulTilingD.baseK = 8;
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
    tilingDatafromBin->matmulTilingD.shareL1Size = 1024;
    tilingDatafromBin->matmulTilingD.shareL0CSize = 1024;
    tilingDatafromBin->matmulTilingD.shareUbSize = 0;
    tilingDatafromBin->matmulTilingD.batchM = 1;
    tilingDatafromBin->matmulTilingD.batchN = 1;
    tilingDatafromBin->matmulTilingD.singleBatchM = 1;
    tilingDatafromBin->matmulTilingD.singleBatchN = 1;

    ICPU_SET_TILING_KEY(1);
	AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(upsample_trilinear3d_backward, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    fileName = "./upsample_trilinear3d_backward_data/float32_output_trilinear3d_grad.bin";
    WriteFile(fileName, y, grad_inputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./upsample_trilinear3d_backward_data/ && python3 compare_data.py 'float32'");
}
