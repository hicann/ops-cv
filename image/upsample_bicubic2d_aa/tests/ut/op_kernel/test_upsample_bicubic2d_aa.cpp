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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "upsample_bicubic2d_aa_tiling.h"
#include "../data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void upsample_bicubic2d_aa(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class upsample_bicubic2d_aa_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "upsample_bicubic2d_aa_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "upsample_bicubic2d_aa_test TearDown\n" << endl;
    }
};

TEST_F(upsample_bicubic2d_aa_test, test_case_float32)
{
    system("cp -rf "
           "../../../../../../../ops/image/upsample_bicubic2d_aa/tests/ut/op_kernel/upsample_bicubic2d_aa_data ./");
    system("chmod -R 755 ./upsample_bicubic2d_aa_data/");
    system("cd ./upsample_bicubic2d_aa_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)' 'float32'");

    size_t inputByteSize = 4 * 4 * sizeof(float);
    size_t outputByteSize = 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleBicubic2dAATilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_bicubic2d_aa_data/float32_input_bicubic2d_aa.bin";
    ;
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleBicubic2dAATilingData *tilingDatafromBin = reinterpret_cast<UpsampleBicubic2dAATilingData *>(tiling);

    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.25;
    tilingDatafromBin->invscaleW = 1.0;
    tilingDatafromBin->invscaleH = 1.0;
    tilingDatafromBin->supportW = 2;
    tilingDatafromBin->supportH = 2;
    tilingDatafromBin->maxInterpSizeW = 5;
    tilingDatafromBin->maxInterpSizeH = 5;
    tilingDatafromBin->radioMatrixWSize = 512;
    tilingDatafromBin->radioMatrixHSize = 512;
    tilingDatafromBin->needCoreNumW = 1;
    tilingDatafromBin->needCoreNumH = 1;
    tilingDatafromBin->sliceSize = 16;
    tilingDatafromBin->intermediateMatrixSize = 1024;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 1;
    tilingDatafromBin->inputShapes[2] = 4;
    tilingDatafromBin->inputShapes[3] = 4;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 1;
    tilingDatafromBin->outputShapes[2] = 16;
    tilingDatafromBin->outputShapes[3] = 16;
    tilingDatafromBin->tailSliceEndListW[0] = 16;
    tilingDatafromBin->tailRowEndListW[0] = 4;
    tilingDatafromBin->tailSliceEndListH[0] = 16;
    tilingDatafromBin->tailBatchEndListH[0] = 1;

    tilingDatafromBin->matmulTilingW.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingW.M = 4;
    tilingDatafromBin->matmulTilingW.N = 8;
    tilingDatafromBin->matmulTilingW.Ka = 4;
    tilingDatafromBin->matmulTilingW.Kb = 4;
    tilingDatafromBin->matmulTilingW.singleCoreM = 4;
    tilingDatafromBin->matmulTilingW.singleCoreN = 16;
    tilingDatafromBin->matmulTilingW.singleCoreK = 8;
    tilingDatafromBin->matmulTilingW.baseM = 16;
    tilingDatafromBin->matmulTilingW.baseN = 16;
    tilingDatafromBin->matmulTilingW.baseK = 8;
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
    tilingDatafromBin->matmulTilingW.shareL1Size = 1024;
    tilingDatafromBin->matmulTilingW.shareL0CSize = 1024;
    tilingDatafromBin->matmulTilingW.shareUbSize = 0;
    tilingDatafromBin->matmulTilingW.batchM = 1;
    tilingDatafromBin->matmulTilingW.batchN = 1;
    tilingDatafromBin->matmulTilingW.singleBatchM = 1;
    tilingDatafromBin->matmulTilingW.singleBatchN = 1;

    tilingDatafromBin->matmulTilingH.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingH.M = 16;
    tilingDatafromBin->matmulTilingH.N = 16;
    tilingDatafromBin->matmulTilingH.Ka = 4;
    tilingDatafromBin->matmulTilingH.Kb = 4;
    tilingDatafromBin->matmulTilingH.singleCoreM = 16;
    tilingDatafromBin->matmulTilingH.singleCoreN = 16;
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

    ICPU_SET_TILING_KEY(2);

    ICPU_RUN_KF(upsample_bicubic2d_aa, blockDim, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./upsample_bicubic2d_aa_data/float32_output_bicubic2d_aa.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bicubic2d_aa_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_bicubic2d_aa_test, test_case_float16)
{
    system("cp -rf "
           "../../../../../../../ops/image/upsample_bicubic2d_aa/tests/ut/op_kernel/upsample_bicubic2d_aa_data ./");
    system("chmod -R 755 ./upsample_bicubic2d_aa_data/");
    system("cd ./upsample_bicubic2d_aa_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)' 'float16'");

    size_t inputByteSize = 4 * 4 * sizeof(half);
    size_t outputByteSize = 16 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(UpsampleBicubic2dAATilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_bicubic2d_aa_data/float16_input_bicubic2d_aa.bin";
    ;
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleBicubic2dAATilingData *tilingDatafromBin = reinterpret_cast<UpsampleBicubic2dAATilingData *>(tiling);

    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.25;
    tilingDatafromBin->invscaleW = 1.0;
    tilingDatafromBin->invscaleH = 1.0;
    tilingDatafromBin->supportW = 2;
    tilingDatafromBin->supportH = 2;
    tilingDatafromBin->maxInterpSizeW = 5;
    tilingDatafromBin->maxInterpSizeH = 5;
    tilingDatafromBin->radioMatrixWSize = 512;
    tilingDatafromBin->radioMatrixHSize = 512;
    tilingDatafromBin->needCoreNumW = 1;
    tilingDatafromBin->needCoreNumH = 1;
    tilingDatafromBin->sliceSize = 16;
    tilingDatafromBin->intermediateMatrixSize = 1024;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 1;
    tilingDatafromBin->inputShapes[2] = 4;
    tilingDatafromBin->inputShapes[3] = 4;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 1;
    tilingDatafromBin->outputShapes[2] = 16;
    tilingDatafromBin->outputShapes[3] = 16;
    tilingDatafromBin->tailSliceEndListW[0] = 16;
    tilingDatafromBin->tailRowEndListW[0] = 4;
    tilingDatafromBin->tailSliceEndListH[0] = 16;
    tilingDatafromBin->tailBatchEndListH[0] = 1;

    tilingDatafromBin->matmulTilingW.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingW.M = 4;
    tilingDatafromBin->matmulTilingW.N = 8;
    tilingDatafromBin->matmulTilingW.Ka = 4;
    tilingDatafromBin->matmulTilingW.Kb = 4;
    tilingDatafromBin->matmulTilingW.singleCoreM = 4;
    tilingDatafromBin->matmulTilingW.singleCoreN = 16;
    tilingDatafromBin->matmulTilingW.singleCoreK = 8;
    tilingDatafromBin->matmulTilingW.baseM = 16;
    tilingDatafromBin->matmulTilingW.baseN = 16;
    tilingDatafromBin->matmulTilingW.baseK = 8;
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
    tilingDatafromBin->matmulTilingW.shareL1Size = 1024;
    tilingDatafromBin->matmulTilingW.shareL0CSize = 1024;
    tilingDatafromBin->matmulTilingW.shareUbSize = 0;
    tilingDatafromBin->matmulTilingW.batchM = 1;
    tilingDatafromBin->matmulTilingW.batchN = 1;
    tilingDatafromBin->matmulTilingW.singleBatchM = 1;
    tilingDatafromBin->matmulTilingW.singleBatchN = 1;

    tilingDatafromBin->matmulTilingH.usedCoreNum = 1;
    tilingDatafromBin->matmulTilingH.M = 16;
    tilingDatafromBin->matmulTilingH.N = 16;
    tilingDatafromBin->matmulTilingH.Ka = 4;
    tilingDatafromBin->matmulTilingH.Kb = 4;
    tilingDatafromBin->matmulTilingH.singleCoreM = 16;
    tilingDatafromBin->matmulTilingH.singleCoreN = 16;
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

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(upsample_bicubic2d_aa, blockDim, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./upsample_bicubic2d_aa_data/float16_output_bicubic2d_aa.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bicubic2d_aa_data/ && python3 compare_data.py 'float16'");
}
