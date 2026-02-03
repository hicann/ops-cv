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
 * \file test_upsample_nearest.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/upsample_nearest_tiling.h"
#include "data_utils.h"

extern "C" __global__ __aicore__ void upsample_nearest(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class upsample_nearest_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "upsample_nearest_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "upsample_nearest_test TearDown\n" << std::endl;
    }
};

TEST_F(upsample_nearest_test, test_case_float_1)
{
    system(
        "cp -rf "
        "../../../../image/upsample_nearest/tests/ut/op_kernel/upsample_nearest_data ./");
    system("chmod -R 755 ./upsample_nearest_data/");
    system("cd ./upsample_nearest_data/ && python3 gen_data.py '(1, 4, 4, 1)' '(16, 16)' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t inputByteSize = 4 * 4 * sizeof(float);
    size_t outputByteSize = 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleNearestTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_nearest_data/float32_input_upsample_nearest.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleNearestTilingData* tilingDatafromBin = reinterpret_cast<UpsampleNearestTilingData*>(tiling);

    tilingDatafromBin->dataType = 4;
    tilingDatafromBin->exactMode = true;
    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.25;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 4;
    tilingDatafromBin->inputShapes[2] = 4;
    tilingDatafromBin->inputShapes[3] = 1;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 16;
    tilingDatafromBin->outputShapes[2] = 16;
    tilingDatafromBin->outputShapes[3] = 1;

    tilingDatafromBin->tailColStartList[0] = 0;
    tilingDatafromBin->tailColEndList[0] = 16;
    tilingDatafromBin->tailRowStartList[0] = 0;
    tilingDatafromBin->tailRowEndList[0] = 16;

    ICPU_SET_TILING_KEY(1001);

    ICPU_RUN_KF(upsample_nearest, numBlocks, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./upsample_nearest_data/float32_output_upsample_nearest.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./upsample_nearest_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_nearest_test, test_case_float_2)
{
    system(
        "cp -rf "
        "../../../../image/upsample_nearest/tests/ut/op_kernel/upsample_nearest_data ./");
    system("chmod -R 755 ./upsample_nearest_data/");
    system("cd ./upsample_nearest_data/ && python3 gen_data.py '(1, 4, 4, 1)' '(16, 16)' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t inputByteSize = 4 * 4 * sizeof(float);
    size_t outputByteSize = 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleNearestTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_nearest_data/float32_input_upsample_nearest.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleNearestTilingData* tilingDatafromBin = reinterpret_cast<UpsampleNearestTilingData*>(tiling);

    tilingDatafromBin->dataType = 4;
    tilingDatafromBin->exactMode = true;
    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.25;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 4;
    tilingDatafromBin->inputShapes[2] = 4;
    tilingDatafromBin->inputShapes[3] = 1;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 16;
    tilingDatafromBin->outputShapes[2] = 16;
    tilingDatafromBin->outputShapes[3] = 1;

    tilingDatafromBin->tailColStartList[0] = 0;
    tilingDatafromBin->tailColEndList[0] = 16;
    tilingDatafromBin->tailRowStartList[0] = 0;
    tilingDatafromBin->tailRowEndList[0] = 16;

    ICPU_SET_TILING_KEY(1002);

    ICPU_RUN_KF(upsample_nearest, numBlocks, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./upsample_nearest_data/float32_output_upsample_nearest.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./upsample_nearest_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_nearest_test, test_case_float_3)
{
    system(
        "cp -rf "
        "../../../../image/upsample_nearest/tests/ut/op_kernel/upsample_nearest_data ./");
    system("chmod -R 755 ./upsample_nearest_data/");
    system("cd ./upsample_nearest_data/ && python3 gen_data.py '(1, 1, 128, 1)' '(1, 64)' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t inputByteSize = 1 * 128 * sizeof(float);
    size_t outputByteSize = 1 * 64 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleNearestTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_nearest_data/float32_input_upsample_nearest.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleNearestTilingData* tilingDatafromBin = reinterpret_cast<UpsampleNearestTilingData*>(tiling);

    tilingDatafromBin->dataType = 4;
    tilingDatafromBin->exactMode = true;
    tilingDatafromBin->scaleW = 2;
    tilingDatafromBin->scaleH = 1;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 1;
    tilingDatafromBin->inputShapes[2] = 128;
    tilingDatafromBin->inputShapes[3] = 1;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 1;
    tilingDatafromBin->outputShapes[2] = 64;
    tilingDatafromBin->outputShapes[3] = 1;

    tilingDatafromBin->tailColStartList[0] = 0;
    tilingDatafromBin->tailColEndList[0] = 64;
    tilingDatafromBin->tailRowStartList[0] = 0;
    tilingDatafromBin->tailRowEndList[0] = 1;

    ICPU_SET_TILING_KEY(1000);

    ICPU_RUN_KF(upsample_nearest, numBlocks, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./upsample_nearest_data/float32_output_upsample_nearest.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./upsample_nearest_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_nearest_test, test_case_float16_1)
{
    system(
        "cp -rf "
        "../../../../image/upsample_nearest/tests/ut/op_kernel/upsample_nearest_data ./");
    system("chmod -R 755 ./upsample_nearest_data/");
    system("cd ./upsample_nearest_data/ && python3 gen_data.py '(1, 4, 4, 1)' '(16, 16)' 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t inputByteSize = 4 * 4 * sizeof(half);
    size_t outputByteSize = 16 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(UpsampleNearestTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_nearest_data/float16_input_upsample_nearest.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleNearestTilingData* tilingDatafromBin = reinterpret_cast<UpsampleNearestTilingData*>(tiling);

    tilingDatafromBin->dataType = 2;
    tilingDatafromBin->exactMode = true;
    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.25;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 4;
    tilingDatafromBin->inputShapes[2] = 4;
    tilingDatafromBin->inputShapes[3] = 1;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 16;
    tilingDatafromBin->outputShapes[2] = 16;
    tilingDatafromBin->outputShapes[3] = 1;

    tilingDatafromBin->tailColStartList[0] = 0;
    tilingDatafromBin->tailColEndList[0] = 16;
    tilingDatafromBin->tailRowStartList[0] = 0;
    tilingDatafromBin->tailRowEndList[0] = 16;

    ICPU_SET_TILING_KEY(1002);

    ICPU_RUN_KF(upsample_nearest, numBlocks, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./upsample_nearest_data/float16_output_upsample_nearest.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./upsample_nearest_data/ && python3 compare_data.py 'float16'");
}

TEST_F(upsample_nearest_test, test_case_float16_2)
{
    system(
        "cp -rf "
        "../../../../image/upsample_nearest/tests/ut/op_kernel/upsample_nearest_data ./");
    system("chmod -R 755 ./upsample_nearest_data/");
    system("cd ./upsample_nearest_data/ && python3 gen_data.py '(1, 4, 4, 1)' '(16, 16)' 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t inputByteSize = 4 * 4 * sizeof(half);
    size_t outputByteSize = 16 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(UpsampleNearestTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_nearest_data/float16_input_upsample_nearest.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleNearestTilingData* tilingDatafromBin = reinterpret_cast<UpsampleNearestTilingData*>(tiling);

    tilingDatafromBin->dataType = 2;
    tilingDatafromBin->exactMode = true;
    tilingDatafromBin->scaleW = 0.25;
    tilingDatafromBin->scaleH = 0.25;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 4;
    tilingDatafromBin->inputShapes[2] = 4;
    tilingDatafromBin->inputShapes[3] = 1;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 16;
    tilingDatafromBin->outputShapes[2] = 16;
    tilingDatafromBin->outputShapes[3] = 1;

    tilingDatafromBin->tailColStartList[0] = 0;
    tilingDatafromBin->tailColEndList[0] = 16;
    tilingDatafromBin->tailRowStartList[0] = 0;
    tilingDatafromBin->tailRowEndList[0] = 16;

    ICPU_SET_TILING_KEY(1001);

    ICPU_RUN_KF(upsample_nearest, numBlocks, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./upsample_nearest_data/float16_output_upsample_nearest.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./upsample_nearest_data/ && python3 compare_data.py 'float16'");
}
