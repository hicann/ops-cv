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
 * \file test_upsample_nearest_exact3d.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/upsample_nearest_exact3d.cpp"
#include "data_utils.h"

extern "C" __global__ __aicore__ void upsample_nearest_exact3d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class upsample_nearest_exact3d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "upsample_nearest_exact3d_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "upsample_nearest_exact3d_test TearDown\n" << std::endl;
    }
};

TEST_F(upsample_nearest_exact3d_test, test_case_float_1)
{
    system("cp -rf "
           "../../../../image/upsample_nearest_exact3d/tests/ut/op_kernel/upsample_nearest_exact3d_data "
           "./");
    system("chmod -R 755 ./upsample_nearest_exact3d_data/");
    system("cd ./upsample_nearest_exact3d_data/ && python3 gen_data.py '(1, 1, 4, 4, 4)' '(16, 16, 16)' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t inputByteSize = 4 * 4 * 4 * sizeof(float);
    size_t outputByteSize = 16 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleNearest3dTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 16;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_nearest_exact3d_data/float32_input_upsample_nearest_exact3d.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleNearest3dTilingData *tilingDatafromBin = reinterpret_cast<UpsampleNearest3dTilingData *>(tiling);

    tilingDatafromBin->dataType = 2;
    tilingDatafromBin->batches = 1;
    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.25;
    tilingDatafromBin->scaleD = 0.25;

    tilingDatafromBin->slideSizeW = 4096;
    tilingDatafromBin->tensorSizeW = 1028;
    tilingDatafromBin->tensorSizeH = 8;
    tilingDatafromBin->tensorSizeD = 8;
    tilingDatafromBin->slideNumH = 4;
    tilingDatafromBin->slideNumD = 4;

    tilingDatafromBin->eachCoreSlideNum = 0;
    tilingDatafromBin->remainder = 16;
    tilingDatafromBin->tailStartSlideNum = 0;
    tilingDatafromBin->groupCoreNum = 1;
    tilingDatafromBin->inputRow = 1;
    tilingDatafromBin->tailAvergingRow = 1;
    tilingDatafromBin->needCoreNum = 16;

    tilingDatafromBin->inputShapes[0] = 4;
    tilingDatafromBin->inputShapes[1] = 4;
    tilingDatafromBin->inputShapes[2] = 4;
    tilingDatafromBin->outputShapes[0] = 16;
    tilingDatafromBin->outputShapes[1] = 16;
    tilingDatafromBin->outputShapes[2] = 16;

    ICPU_SET_TILING_KEY(5140);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = upsample_nearest_exact3d<UPSAMPLE_NEAREST3D_TPL_FP32, UPSAMPLE_NEAREST3D_TPL_FP32>; 
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./upsample_nearest_exact3d_data/float32_output_upsample_nearest_exact3d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_nearest_exact3d_data/ && python3 compare_data.py 'float32'");
}
