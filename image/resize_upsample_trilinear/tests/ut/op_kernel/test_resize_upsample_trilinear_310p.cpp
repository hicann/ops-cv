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
#include "data_utils.h"
#include "resize_upsample_trilinear_310p_tiling.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void resize_upsample_trilinear(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class resize_upsample_trilinear_310p_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "resize_upsample_trilinear_310p_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "resize_upsample_trilinear_310p_test TearDown\n" << endl;
    }
};

TEST_F(resize_upsample_trilinear_310p_test, test1_case_float32)
{
    system(
        "cp -rf "
        "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
        "resize_upsample_trilinear_310p_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_310p_data/");
    system(
        "cd ./resize_upsample_trilinear_310p_data/ && python3 gen_data.py '(1, 2, 32, 1, 16)' '(1, 4, 64)' 'float32'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t inputByteSize = 1 * 2 * 32 * 16 * sizeof(float);
    size_t outputByteSize = 1 * 4 * 64 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleTrilinearTilingDataTest);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 4;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./resize_upsample_trilinear_310p_data/float32_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleTrilinearTilingDataTest* tilingDatafromBin = reinterpret_cast<UpsampleTrilinearTilingDataTest*>(tiling);

    tilingDatafromBin->scale_w = 0.5;
    tilingDatafromBin->scale_h = 0.5;
    tilingDatafromBin->scale_d = 1;
    tilingDatafromBin->total_core_num = 8;
    tilingDatafromBin->real_core_num = 4;

    tilingDatafromBin->output_w = 64;
    tilingDatafromBin->output_h = 4;
    tilingDatafromBin->output_d = 1;
    tilingDatafromBin->input_w = 32;
    tilingDatafromBin->input_h = 2;
    tilingDatafromBin->input_d = 1;
    tilingDatafromBin->batches = 16;
    tilingDatafromBin->align_corners = 1;

    tilingDatafromBin->each_core_slide_num = 0;
    tilingDatafromBin->remainder = 4;
    tilingDatafromBin->tail_start_slide_num = 0;
    tilingDatafromBin->slide_size = 254;
    tilingDatafromBin->batch_size = 16;
    tilingDatafromBin->tensor_size = 131;

    ICPU_SET_TILING_KEY(3000);

    ICPU_RUN_KF(resize_upsample_trilinear, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "././resize_upsample_trilinear_310p_data/float32_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./resize_upsample_trilinear_310p_data/ && python3 compare_data.py 'float32'");
}

TEST_F(resize_upsample_trilinear_310p_test, test_case_float32)
{
    system(
        "cp -rf "
        "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
        "resize_upsample_trilinear_310p_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_310p_data/");
    system("cd ./resize_upsample_trilinear_310p_data/ && python3 gen_data.py '(1, 2, 2, 1, 16)' '(1, 2, 4)' 'float32'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t inputByteSize = 1 * 2 * 2 * 16 * sizeof(float);
    size_t outputByteSize = 1 * 2 * 4 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleTrilinearTilingDataTest);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./resize_upsample_trilinear_310p_data/float32_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleTrilinearTilingDataTest* tilingDatafromBin = reinterpret_cast<UpsampleTrilinearTilingDataTest*>(tiling);

    tilingDatafromBin->scale_w = 0.5;
    tilingDatafromBin->scale_h = 1;
    tilingDatafromBin->scale_d = 1;
    tilingDatafromBin->total_core_num = 8;
    tilingDatafromBin->real_core_num = 2;

    tilingDatafromBin->output_w = 4;
    tilingDatafromBin->output_h = 2;
    tilingDatafromBin->output_d = 1;
    tilingDatafromBin->input_w = 2;
    tilingDatafromBin->input_h = 2;
    tilingDatafromBin->input_d = 1;
    tilingDatafromBin->batches = 16;
    tilingDatafromBin->align_corners = 0;

    tilingDatafromBin->each_core_slide_num = 0;
    tilingDatafromBin->remainder = 2;
    tilingDatafromBin->tail_start_slide_num = 0;
    tilingDatafromBin->slide_size = 254;
    tilingDatafromBin->batch_size = 16;
    tilingDatafromBin->tensor_size = 131;

    ICPU_SET_TILING_KEY(3000);

    ICPU_RUN_KF(resize_upsample_trilinear, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "././resize_upsample_trilinear_310p_data/float32_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./resize_upsample_trilinear_310p_data/ && python3 compare_data.py 'float32'");
}

TEST_F(resize_upsample_trilinear_310p_test, test_case_float16)
{
    system(
        "cp -rf "
        "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
        "resize_upsample_trilinear_310p_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_310p_data/");
    system("cd ./resize_upsample_trilinear_310p_data/ && python3 gen_data.py '(1, 2, 2, 1, 16)' '(1, 2, 4)' 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t inputByteSize = 1 * 2 * 2 * 16 * sizeof(half);
    size_t outputByteSize = 1 * 2 * 4 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(UpsampleTrilinearTilingDataTest);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./resize_upsample_trilinear_310p_data/float16_input_trilinear.bin";
    ;
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleTrilinearTilingDataTest* tilingDatafromBin = reinterpret_cast<UpsampleTrilinearTilingDataTest*>(tiling);

    tilingDatafromBin->scale_w = 0.5;
    tilingDatafromBin->scale_h = 1;
    tilingDatafromBin->scale_d = 1;
    tilingDatafromBin->total_core_num = 8;
    tilingDatafromBin->real_core_num = 2;

    tilingDatafromBin->output_w = 4;
    tilingDatafromBin->output_h = 2;
    tilingDatafromBin->output_d = 1;
    tilingDatafromBin->input_w = 2;
    tilingDatafromBin->input_h = 2;
    tilingDatafromBin->input_d = 1;
    tilingDatafromBin->batches = 16;
    tilingDatafromBin->align_corners = 0;

    tilingDatafromBin->each_core_slide_num = 0;
    tilingDatafromBin->remainder = 2;
    tilingDatafromBin->tail_start_slide_num = 0;
    tilingDatafromBin->slide_size = 254;
    tilingDatafromBin->batch_size = 16;
    tilingDatafromBin->tensor_size = 131;

    ICPU_SET_TILING_KEY(1000);

    ICPU_RUN_KF(resize_upsample_trilinear, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./resize_upsample_trilinear_310p_data/float16_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./resize_upsample_trilinear_310p_data/ && python3 compare_data.py 'float16'");
}
