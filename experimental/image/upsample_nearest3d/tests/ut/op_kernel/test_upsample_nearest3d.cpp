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
 * \file test_upsample_nearest3d.cpp
 * \brief
 */

#include <algorithm>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/upsample_nearest3d.cpp"
#include "data_utils.h"

extern "C" __global__ __aicore__ void upsample_nearest3d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class upsample_nearest3d_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "upsample_nearest3d_test SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "upsample_nearest3d_test TearDown\n" << std::endl; }
};

TEST_F(upsample_nearest3d_test, test_case_float_1)
{
    system("cp -rf "
           "../../../../experimental/image/upsample_nearest3d/tests/ut/op_kernel/upsample_nearest3d_data ./");
    system("chmod -R 755 ./upsample_nearest3d_data/");
    system("cd ./upsample_nearest3d_data/ && python3 gen_data.py '(1, 1, 4, 4, 4)' '(16, 16, 16)' 'float32'");

    size_t inputByteSize = 4 * 4 * 4 * sizeof(float);
    size_t outputByteSize = 16 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleNearest3dTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 16;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_nearest3d_data/float32_input_upsample_nearest3d.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleNearest3dTilingData* tilingDatafromBin = reinterpret_cast<UpsampleNearest3dTilingData*>(tiling);

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
    auto func = upsample_nearest3d<UPSAMPLE_NEAREST3D_TPL_FP32, UPSAMPLE_NEAREST3D_TPL_FP32>;
    ICPU_RUN_KF(func, numBlocks, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./upsample_nearest3d_data/float32_output_upsample_nearest3d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./upsample_nearest3d_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_nearest3d_test, test_case_float_2)
{
    system("cp -rf "
           "../../../../experimental/image/upsample_nearest3d/tests/ut/op_kernel/upsample_nearest3d_data ./");
    system("chmod -R 755 ./upsample_nearest3d_data/");
    system("cd ./upsample_nearest3d_data/ && python3 gen_data.py '(1, 64, 1, 1, 1)' '(1, 1, 4)' 'float32'");

    size_t inputByteSize = 64 * sizeof(float);
    size_t outputByteSize = 64 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleNearest3dTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 32;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_nearest3d_data/float32_input_upsample_nearest3d.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleNearest3dTilingData* tilingDatafromBin = reinterpret_cast<UpsampleNearest3dTilingData*>(tiling);

    tilingDatafromBin->dataType = 0;
    tilingDatafromBin->batches = 64;
    tilingDatafromBin->scaleW = 1;
    tilingDatafromBin->scaleH = 1;
    tilingDatafromBin->scaleD = 0.25;

    tilingDatafromBin->slideSizeW = 2048;
    tilingDatafromBin->tensorSizeW = 516;
    tilingDatafromBin->tensorSizeH = 1;
    tilingDatafromBin->tensorSizeD = 1;
    tilingDatafromBin->slideNumH = 1;
    tilingDatafromBin->slideNumD = 1;

    tilingDatafromBin->eachCoreSlideNum = 0;
    tilingDatafromBin->remainder = 1;
    tilingDatafromBin->tailStartSlideNum = 0;
    tilingDatafromBin->groupCoreNum = 32;
    tilingDatafromBin->inputRow = 64;
    tilingDatafromBin->tailAvergingRow = 2;
    tilingDatafromBin->needCoreNum = 32;

    tilingDatafromBin->inputShapes[0] = 1;
    tilingDatafromBin->inputShapes[1] = 1;
    tilingDatafromBin->inputShapes[2] = 1;
    tilingDatafromBin->outputShapes[0] = 1;
    tilingDatafromBin->outputShapes[1] = 1;
    tilingDatafromBin->outputShapes[2] = 4;

    tilingDatafromBin->isView1DAndSmallW = true;

    ICPU_SET_TILING_KEY(5140);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = upsample_nearest3d<UPSAMPLE_NEAREST3D_TPL_FP32, UPSAMPLE_NEAREST3D_TPL_FP32>;
    ICPU_RUN_KF(func, numBlocks, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./upsample_nearest3d_data/float32_output_upsample_nearest3d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./upsample_nearest3d_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_nearest3d_test, test_case_uint8_common_batched_bridge_and_non_aligned_w_tail)
{
    constexpr int64_t batches = 3;
    constexpr int64_t inputW = 1025;
    constexpr int64_t outputW = 2051;
    constexpr int64_t slideSizeW = 2048;
    constexpr size_t inputByteSize = batches * inputW;
    constexpr size_t outputByteSize = batches * outputW;

    uint8_t* x = static_cast<uint8_t*>(AscendC::GmAlloc(inputByteSize));
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(outputByteSize));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(1));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(UpsampleNearest3dTilingData)));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(tiling, nullptr);
    std::memset(y, 0, outputByteSize);
    std::memset(tiling, 0, sizeof(UpsampleNearest3dTilingData));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t w = 0; w < inputW; ++w) {
            x[batch * inputW + w] = static_cast<uint8_t>((batch * 37 + w) % 251);
        }
    }

    auto* data = reinterpret_cast<UpsampleNearest3dTilingData*>(tiling);
    data->dataType = 4;
    data->batches = batches;
    data->inputShapes[0] = 1;
    data->inputShapes[1] = 1;
    data->inputShapes[2] = inputW;
    data->outputShapes[0] = 1;
    data->outputShapes[1] = 1;
    data->outputShapes[2] = outputW;
    data->scaleD = 1.0f;
    data->scaleH = 1.0f;
    data->scaleW = static_cast<float>(inputW) / static_cast<float>(outputW);
    data->slideSizeW = slideSizeW;
    data->tensorSizeW = slideSizeW;
    data->tensorSizeH = 1;
    data->tensorSizeD = 1;
    data->slideNumH = 1;
    data->slideNumD = 1;
    data->eachCoreSlideNum = 1;
    data->remainder = 0;
    data->tailStartSlideNum = 2;
    data->groupCoreNum = 1;
    data->inputRow = batches;
    data->tailAvergingRow = batches;
    data->needCoreNum = 2;
    data->isView1DAndSmallW = false;

    ICPU_SET_TILING_KEY(10280);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = upsample_nearest3d<UPSAMPLE_NEAREST3D_TPL_UINT8, UPSAMPLE_NEAREST3D_TPL_UINT8>;
    ICPU_RUN_KF(func, 2, x, y, workspace, tiling);

    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t w = 0; w < outputW; ++w) {
            const int64_t mappedW = std::min(inputW - 1, static_cast<int64_t>(static_cast<float>(w) * data->scaleW));
            EXPECT_EQ(y[batch * outputW + w], x[batch * inputW + mappedW]);
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(upsample_nearest3d_test, test_case_uint8_view1d_small_w_batched_bridge_and_tail)
{
    constexpr int64_t batches = 39;
    constexpr int64_t inputW = 4;
    constexpr int64_t outputW = 7;
    constexpr size_t inputByteSize = batches * inputW;
    constexpr size_t outputByteSize = batches * outputW;

    uint8_t* x = static_cast<uint8_t*>(AscendC::GmAlloc(inputByteSize));
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(outputByteSize));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(1));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(UpsampleNearest3dTilingData)));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(tiling, nullptr);
    std::memset(y, 0, outputByteSize);
    std::memset(tiling, 0, sizeof(UpsampleNearest3dTilingData));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t w = 0; w < inputW; ++w) {
            x[batch * inputW + w] = static_cast<uint8_t>((batch * 7 + w + 1) % 251);
        }
    }

    auto* data = reinterpret_cast<UpsampleNearest3dTilingData*>(tiling);
    data->dataType = 4;
    data->batches = batches;
    data->inputShapes[0] = 1;
    data->inputShapes[1] = 1;
    data->inputShapes[2] = inputW;
    data->outputShapes[0] = 1;
    data->outputShapes[1] = 1;
    data->outputShapes[2] = outputW;
    data->scaleD = 1.0f;
    data->scaleH = 1.0f;
    data->scaleW = static_cast<float>(inputW) / static_cast<float>(outputW);
    data->slideSizeW = 2048;
    data->tensorSizeW = 2048;
    data->tensorSizeH = 1;
    data->tensorSizeD = 1;
    data->slideNumH = 1;
    data->slideNumD = 1;
    data->eachCoreSlideNum = 0;
    data->remainder = 1;
    data->tailStartSlideNum = 0;
    data->groupCoreNum = 1;
    data->inputRow = batches;
    data->tailAvergingRow = batches;
    data->needCoreNum = 1;
    data->isView1DAndSmallW = true;

    ICPU_SET_TILING_KEY(10280);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = upsample_nearest3d<UPSAMPLE_NEAREST3D_TPL_UINT8, UPSAMPLE_NEAREST3D_TPL_UINT8>;
    ICPU_RUN_KF(func, 1, x, y, workspace, tiling);

    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t w = 0; w < outputW; ++w) {
            const int64_t mappedW = std::min(inputW - 1, static_cast<int64_t>(static_cast<float>(w) * data->scaleW));
            EXPECT_EQ(y[batch * outputW + w], x[batch * inputW + mappedW]);
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
