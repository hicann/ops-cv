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
#include "resize_upsample_trilinear_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void resize_upsample_trilinear(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class ResizeUpsampleTrilinearTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ResizeUpsampleTrilinearTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ResizeUpsampleTrilinearTest TearDown\n" << endl;
    }
};

TEST_F(ResizeUpsampleTrilinearTest, test_case_float32)
{
    system(
        "cp -rf "
        "../../../../../../../ops/image/resize_upsample_trilinear/tests/ut/op_kernel/"
        "resize_upsample_trilinear_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_data/");
    system("cd ./resize_upsample_trilinear_data/ && python3 gen_data.py '(1, 2, 2, 4, 4)' '(8, 8, 16)' 'float32'");

    size_t inputByteSize = 2 * 2 * 4 * 4 * sizeof(float);
    size_t outputByteSize = 2 * 8 * 8 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleTrilinearTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 4;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./resize_upsample_trilinear_data/float32_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleTrilinearTilingData* tilingDatafromBin = reinterpret_cast<UpsampleTrilinearTilingData*>(tiling);

    tilingDatafromBin->scale_w = 0.25;
    tilingDatafromBin->scale_h = 0.5;
    tilingDatafromBin->scale_d = 0.25;
    tilingDatafromBin->total_core_num = 20;
    tilingDatafromBin->ratio_metrix_size = 160;
    tilingDatafromBin->output_w = 16;
    tilingDatafromBin->output_h = 8;
    tilingDatafromBin->output_d = 8;
    tilingDatafromBin->input_w = 4;
    tilingDatafromBin->input_h = 4;
    tilingDatafromBin->input_d = 2;
    tilingDatafromBin->batches = 2;
    tilingDatafromBin->align_corners = 0;

    tilingDatafromBin->tail_group_start_inx_w_list[0] = 0;
    tilingDatafromBin->tail_group_end_inx_w_list[0] = 15;
    tilingDatafromBin->tail_group_start_inx_w_list[1] = 0;
    tilingDatafromBin->tail_group_end_inx_w_list[1] = 0;
    tilingDatafromBin->tail_group_start_inx_w_list[2] = 0;
    tilingDatafromBin->tail_group_end_inx_w_list[3] = 0;
    tilingDatafromBin->tail_group_start_inx_w_list[3] = 0;
    tilingDatafromBin->tail_group_end_inx_w_list[3] = 0;
    tilingDatafromBin->tail_group_slide_start_inx_w_list[0] = 0;
    tilingDatafromBin->tail_group_slide_end_inx_w_list[0] = 15;
    tilingDatafromBin->tail_group_slide_start_inx_w_list[1] = 0;
    tilingDatafromBin->tail_group_slide_end_inx_w_list[1] = 0;
    tilingDatafromBin->tail_group_slide_start_inx_w_list[2] = 0;
    tilingDatafromBin->tail_group_slide_end_inx_w_list[2] = 0;
    tilingDatafromBin->tail_group_slide_start_inx_w_list[3] = 0;
    tilingDatafromBin->tail_group_slide_end_inx_w_list[4] = 0;

    tilingDatafromBin->tail_group_start_inx_h_list[0] = 0;
    tilingDatafromBin->tail_group_end_inx_h_list[0] = 7;
    tilingDatafromBin->tail_group_start_inx_h_list[1] = 0;
    tilingDatafromBin->tail_group_end_inx_h_list[1] = 7;
    tilingDatafromBin->tail_group_start_inx_h_list[2] = 0;
    tilingDatafromBin->tail_group_end_inx_h_list[2] = 7;
    tilingDatafromBin->tail_group_start_inx_h_list[3] = 0;
    tilingDatafromBin->tail_group_end_inx_h_list[3] = 7;
    tilingDatafromBin->tail_group_batch_start_inx_h_list[0] = 0;
    tilingDatafromBin->tail_group_batch_end_inx_h_list[0] = 1;
    tilingDatafromBin->tail_group_batch_start_inx_h_list[1] = 1;
    tilingDatafromBin->tail_group_batch_end_inx_h_list[1] = 2;
    tilingDatafromBin->tail_group_batch_start_inx_h_list[2] = 2;
    tilingDatafromBin->tail_group_batch_end_inx_h_list[2] = 3;
    tilingDatafromBin->tail_group_batch_start_inx_h_list[3] = 3;
    tilingDatafromBin->tail_group_batch_end_inx_h_list[3] = 4;

    tilingDatafromBin->tail_group_start_inx_d_list[0] = 0;
    tilingDatafromBin->tail_group_end_inx_d_list[0] = 7;
    tilingDatafromBin->tail_group_start_inx_d_list[1] = 0;
    tilingDatafromBin->tail_group_end_inx_d_list[1] = 7;
    tilingDatafromBin->tail_group_start_inx_d_list[2] = 0;
    tilingDatafromBin->tail_group_end_inx_d_list[2] = 0;
    tilingDatafromBin->tail_group_start_inx_d_list[3] = 0;
    tilingDatafromBin->tail_group_end_inx_d_list[3] = 0;
    tilingDatafromBin->tail_group_batch_start_inx_d_list[0] = 0;
    tilingDatafromBin->tail_group_batch_end_inx_d_list[0] = 1;
    tilingDatafromBin->tail_group_batch_start_inx_d_list[1] = 1;
    tilingDatafromBin->tail_group_batch_end_inx_d_list[1] = 2;
    tilingDatafromBin->tail_group_batch_start_inx_d_list[2] = 0;
    tilingDatafromBin->tail_group_batch_end_inx_d_list[2] = 0;
    tilingDatafromBin->tail_group_batch_start_inx_d_list[3] = 0;
    tilingDatafromBin->tail_group_batch_end_inx_d_list[3] = 0;

    tilingDatafromBin->matmul_tiling_w.usedCoreNum = 1;
    tilingDatafromBin->matmul_tiling_w.M = 16;
    tilingDatafromBin->matmul_tiling_w.N = 16;
    tilingDatafromBin->matmul_tiling_w.Ka = 4;
    tilingDatafromBin->matmul_tiling_w.Kb = 4;
    tilingDatafromBin->matmul_tiling_w.singleCoreM = 16;
    tilingDatafromBin->matmul_tiling_w.singleCoreN = 16;
    tilingDatafromBin->matmul_tiling_w.singleCoreK = 6;
    tilingDatafromBin->matmul_tiling_w.baseM = 16;
    tilingDatafromBin->matmul_tiling_w.baseN = 16;
    tilingDatafromBin->matmul_tiling_w.baseK = 8;
    tilingDatafromBin->matmul_tiling_w.depthA1 = 1;
    tilingDatafromBin->matmul_tiling_w.depthB1 = 1;
    tilingDatafromBin->matmul_tiling_w.stepM = 1;
    tilingDatafromBin->matmul_tiling_w.stepN = 1;
    tilingDatafromBin->matmul_tiling_w.stepKa = 1;
    tilingDatafromBin->matmul_tiling_w.stepKb = 1;
    tilingDatafromBin->matmul_tiling_w.isBias = 0;
    tilingDatafromBin->matmul_tiling_w.transLength = 0;
    tilingDatafromBin->matmul_tiling_w.iterateOrder = 0;
    tilingDatafromBin->matmul_tiling_w.shareMode = 0;
    tilingDatafromBin->matmul_tiling_w.shareL1Size = 1024;
    tilingDatafromBin->matmul_tiling_w.shareL0CSize = 1024;
    tilingDatafromBin->matmul_tiling_w.shareUbSize = 0;
    tilingDatafromBin->matmul_tiling_w.batchM = 1;
    tilingDatafromBin->matmul_tiling_w.batchN = 1;
    tilingDatafromBin->matmul_tiling_w.singleBatchM = 1;
    tilingDatafromBin->matmul_tiling_w.singleBatchN = 1;

    tilingDatafromBin->matmul_tiling_h.usedCoreNum = 1;
    tilingDatafromBin->matmul_tiling_h.M = 8;
    tilingDatafromBin->matmul_tiling_h.N = 16;
    tilingDatafromBin->matmul_tiling_h.Ka = 16;
    tilingDatafromBin->matmul_tiling_h.Kb = 16;
    tilingDatafromBin->matmul_tiling_h.singleCoreM = 8;
    tilingDatafromBin->matmul_tiling_h.singleCoreN = 16;
    tilingDatafromBin->matmul_tiling_h.singleCoreK = 10;
    tilingDatafromBin->matmul_tiling_h.baseM = 16;
    tilingDatafromBin->matmul_tiling_h.baseN = 16;
    tilingDatafromBin->matmul_tiling_h.baseK = 16;
    tilingDatafromBin->matmul_tiling_h.depthA1 = 1;
    tilingDatafromBin->matmul_tiling_h.depthB1 = 1;
    tilingDatafromBin->matmul_tiling_h.stepM = 1;
    tilingDatafromBin->matmul_tiling_h.stepN = 1;
    tilingDatafromBin->matmul_tiling_h.stepKa = 1;
    tilingDatafromBin->matmul_tiling_h.stepKb = 1;
    tilingDatafromBin->matmul_tiling_h.isBias = 0;
    tilingDatafromBin->matmul_tiling_h.transLength = 0;
    tilingDatafromBin->matmul_tiling_h.iterateOrder = 0;
    tilingDatafromBin->matmul_tiling_h.shareMode = 0;
    tilingDatafromBin->matmul_tiling_h.shareL1Size = 2048;
    tilingDatafromBin->matmul_tiling_h.shareL0CSize = 1024;
    tilingDatafromBin->matmul_tiling_h.shareUbSize = 0;
    tilingDatafromBin->matmul_tiling_h.batchM = 1;
    tilingDatafromBin->matmul_tiling_h.batchN = 1;
    tilingDatafromBin->matmul_tiling_h.singleBatchM = 1;
    tilingDatafromBin->matmul_tiling_h.singleBatchN = 1;

    tilingDatafromBin->matmul_tiling_d.usedCoreNum = 1;
    tilingDatafromBin->matmul_tiling_d.M = 8;
    tilingDatafromBin->matmul_tiling_d.N = 128;
    tilingDatafromBin->matmul_tiling_d.Ka = 4;
    tilingDatafromBin->matmul_tiling_d.Kb = 4;
    tilingDatafromBin->matmul_tiling_d.singleCoreM = 8;
    tilingDatafromBin->matmul_tiling_d.singleCoreN = 128;
    tilingDatafromBin->matmul_tiling_d.singleCoreK = 6;
    tilingDatafromBin->matmul_tiling_d.baseM = 16;
    tilingDatafromBin->matmul_tiling_d.baseN = 128;
    tilingDatafromBin->matmul_tiling_d.baseK = 8;
    tilingDatafromBin->matmul_tiling_d.depthA1 = 1;
    tilingDatafromBin->matmul_tiling_d.depthB1 = 1;
    tilingDatafromBin->matmul_tiling_d.stepM = 1;
    tilingDatafromBin->matmul_tiling_d.stepN = 1;
    tilingDatafromBin->matmul_tiling_d.stepKa = 1;
    tilingDatafromBin->matmul_tiling_d.stepKb = 1;
    tilingDatafromBin->matmul_tiling_d.isBias = 0;
    tilingDatafromBin->matmul_tiling_d.transLength = 0;
    tilingDatafromBin->matmul_tiling_d.iterateOrder = 0;
    tilingDatafromBin->matmul_tiling_d.shareMode = 0;
    tilingDatafromBin->matmul_tiling_d.shareL1Size = 4608;
    tilingDatafromBin->matmul_tiling_d.shareL0CSize = 8192;
    tilingDatafromBin->matmul_tiling_d.shareUbSize = 0;
    tilingDatafromBin->matmul_tiling_d.batchM = 1;
    tilingDatafromBin->matmul_tiling_d.batchN = 1;
    tilingDatafromBin->matmul_tiling_d.singleBatchM = 1;
    tilingDatafromBin->matmul_tiling_d.singleBatchN = 1;

    ICPU_SET_TILING_KEY(3000);

    ICPU_RUN_KF(resize_upsample_trilinear, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./resize_upsample_trilinear_data/float32_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./resize_upsample_trilinear_data/ && python3 compare_data.py 'float32'");
}

TEST_F(ResizeUpsampleTrilinearTest, test_case_float16)
{
    system(
        "cp -rf "
        "../../../../../../../ops/image/resize_upsample_trilinear/tests/ut/op_kernel/"
        "resize_upsample_trilinear_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_data/");
    system("cd ./resize_upsample_trilinear_data/ && python3 gen_data.py '(1, 2, 2, 4, 4)' '(8, 8, 16)' 'float16'");

    size_t inputByteSize = 2 * 2 * 4 * 4 * sizeof(half);
    size_t outputByteSize = 2 * 8 * 8 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(UpsampleTrilinearTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 4;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./resize_upsample_trilinear_data/float16_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleTrilinearTilingData* tilingDatafromBin = reinterpret_cast<UpsampleTrilinearTilingData*>(tiling);

    tilingDatafromBin->scale_w = 0.25;
    tilingDatafromBin->scale_h = 0.5;
    tilingDatafromBin->scale_d = 0.25;
    tilingDatafromBin->total_core_num = 20;
    tilingDatafromBin->ratio_metrix_size = 160;
    tilingDatafromBin->output_w = 16;
    tilingDatafromBin->output_h = 8;
    tilingDatafromBin->output_d = 8;
    tilingDatafromBin->input_w = 4;
    tilingDatafromBin->input_h = 4;
    tilingDatafromBin->input_d = 2;
    tilingDatafromBin->batches = 2;
    tilingDatafromBin->align_corners = 0;

    tilingDatafromBin->tail_group_start_inx_w_list[0] = 0;
    tilingDatafromBin->tail_group_end_inx_w_list[0] = 15;
    tilingDatafromBin->tail_group_start_inx_w_list[1] = 0;
    tilingDatafromBin->tail_group_end_inx_w_list[1] = 0;
    tilingDatafromBin->tail_group_start_inx_w_list[2] = 0;
    tilingDatafromBin->tail_group_end_inx_w_list[3] = 0;
    tilingDatafromBin->tail_group_start_inx_w_list[3] = 0;
    tilingDatafromBin->tail_group_end_inx_w_list[3] = 0;
    tilingDatafromBin->tail_group_slide_start_inx_w_list[0] = 0;
    tilingDatafromBin->tail_group_slide_end_inx_w_list[0] = 15;
    tilingDatafromBin->tail_group_slide_start_inx_w_list[1] = 0;
    tilingDatafromBin->tail_group_slide_end_inx_w_list[1] = 0;
    tilingDatafromBin->tail_group_slide_start_inx_w_list[2] = 0;
    tilingDatafromBin->tail_group_slide_end_inx_w_list[2] = 0;
    tilingDatafromBin->tail_group_slide_start_inx_w_list[3] = 0;
    tilingDatafromBin->tail_group_slide_end_inx_w_list[4] = 0;

    tilingDatafromBin->tail_group_start_inx_h_list[0] = 0;
    tilingDatafromBin->tail_group_end_inx_h_list[0] = 7;
    tilingDatafromBin->tail_group_start_inx_h_list[1] = 0;
    tilingDatafromBin->tail_group_end_inx_h_list[1] = 7;
    tilingDatafromBin->tail_group_start_inx_h_list[2] = 0;
    tilingDatafromBin->tail_group_end_inx_h_list[2] = 7;
    tilingDatafromBin->tail_group_start_inx_h_list[3] = 0;
    tilingDatafromBin->tail_group_end_inx_h_list[3] = 7;
    tilingDatafromBin->tail_group_batch_start_inx_h_list[0] = 0;
    tilingDatafromBin->tail_group_batch_end_inx_h_list[0] = 1;
    tilingDatafromBin->tail_group_batch_start_inx_h_list[1] = 1;
    tilingDatafromBin->tail_group_batch_end_inx_h_list[1] = 2;
    tilingDatafromBin->tail_group_batch_start_inx_h_list[2] = 2;
    tilingDatafromBin->tail_group_batch_end_inx_h_list[2] = 3;
    tilingDatafromBin->tail_group_batch_start_inx_h_list[3] = 3;
    tilingDatafromBin->tail_group_batch_end_inx_h_list[3] = 4;

    tilingDatafromBin->tail_group_start_inx_d_list[0] = 0;
    tilingDatafromBin->tail_group_end_inx_d_list[0] = 7;
    tilingDatafromBin->tail_group_start_inx_d_list[1] = 0;
    tilingDatafromBin->tail_group_end_inx_d_list[1] = 7;
    tilingDatafromBin->tail_group_start_inx_d_list[2] = 0;
    tilingDatafromBin->tail_group_end_inx_d_list[2] = 0;
    tilingDatafromBin->tail_group_start_inx_d_list[3] = 0;
    tilingDatafromBin->tail_group_end_inx_d_list[3] = 0;
    tilingDatafromBin->tail_group_batch_start_inx_d_list[0] = 0;
    tilingDatafromBin->tail_group_batch_end_inx_d_list[0] = 1;
    tilingDatafromBin->tail_group_batch_start_inx_d_list[1] = 1;
    tilingDatafromBin->tail_group_batch_end_inx_d_list[1] = 2;
    tilingDatafromBin->tail_group_batch_start_inx_d_list[2] = 0;
    tilingDatafromBin->tail_group_batch_end_inx_d_list[2] = 0;
    tilingDatafromBin->tail_group_batch_start_inx_d_list[3] = 0;
    tilingDatafromBin->tail_group_batch_end_inx_d_list[3] = 0;

    tilingDatafromBin->matmul_tiling_w.usedCoreNum = 1;
    tilingDatafromBin->matmul_tiling_w.M = 16;
    tilingDatafromBin->matmul_tiling_w.N = 16;
    tilingDatafromBin->matmul_tiling_w.Ka = 4;
    tilingDatafromBin->matmul_tiling_w.Kb = 4;
    tilingDatafromBin->matmul_tiling_w.singleCoreM = 16;
    tilingDatafromBin->matmul_tiling_w.singleCoreN = 16;
    tilingDatafromBin->matmul_tiling_w.singleCoreK = 6;
    tilingDatafromBin->matmul_tiling_w.baseM = 16;
    tilingDatafromBin->matmul_tiling_w.baseN = 16;
    tilingDatafromBin->matmul_tiling_w.baseK = 8;
    tilingDatafromBin->matmul_tiling_w.depthA1 = 1;
    tilingDatafromBin->matmul_tiling_w.depthB1 = 1;
    tilingDatafromBin->matmul_tiling_w.stepM = 1;
    tilingDatafromBin->matmul_tiling_w.stepN = 1;
    tilingDatafromBin->matmul_tiling_w.stepKa = 1;
    tilingDatafromBin->matmul_tiling_w.stepKb = 1;
    tilingDatafromBin->matmul_tiling_w.isBias = 0;
    tilingDatafromBin->matmul_tiling_w.transLength = 0;
    tilingDatafromBin->matmul_tiling_w.iterateOrder = 0;
    tilingDatafromBin->matmul_tiling_w.shareMode = 0;
    tilingDatafromBin->matmul_tiling_w.shareL1Size = 1024;
    tilingDatafromBin->matmul_tiling_w.shareL0CSize = 1024;
    tilingDatafromBin->matmul_tiling_w.shareUbSize = 0;
    tilingDatafromBin->matmul_tiling_w.batchM = 1;
    tilingDatafromBin->matmul_tiling_w.batchN = 1;
    tilingDatafromBin->matmul_tiling_w.singleBatchM = 1;
    tilingDatafromBin->matmul_tiling_w.singleBatchN = 1;

    tilingDatafromBin->matmul_tiling_h.usedCoreNum = 1;
    tilingDatafromBin->matmul_tiling_h.M = 8;
    tilingDatafromBin->matmul_tiling_h.N = 16;
    tilingDatafromBin->matmul_tiling_h.Ka = 16;
    tilingDatafromBin->matmul_tiling_h.Kb = 16;
    tilingDatafromBin->matmul_tiling_h.singleCoreM = 8;
    tilingDatafromBin->matmul_tiling_h.singleCoreN = 16;
    tilingDatafromBin->matmul_tiling_h.singleCoreK = 10;
    tilingDatafromBin->matmul_tiling_h.baseM = 16;
    tilingDatafromBin->matmul_tiling_h.baseN = 16;
    tilingDatafromBin->matmul_tiling_h.baseK = 16;
    tilingDatafromBin->matmul_tiling_h.depthA1 = 1;
    tilingDatafromBin->matmul_tiling_h.depthB1 = 1;
    tilingDatafromBin->matmul_tiling_h.stepM = 1;
    tilingDatafromBin->matmul_tiling_h.stepN = 1;
    tilingDatafromBin->matmul_tiling_h.stepKa = 1;
    tilingDatafromBin->matmul_tiling_h.stepKb = 1;
    tilingDatafromBin->matmul_tiling_h.isBias = 0;
    tilingDatafromBin->matmul_tiling_h.transLength = 0;
    tilingDatafromBin->matmul_tiling_h.iterateOrder = 0;
    tilingDatafromBin->matmul_tiling_h.shareMode = 0;
    tilingDatafromBin->matmul_tiling_h.shareL1Size = 2048;
    tilingDatafromBin->matmul_tiling_h.shareL0CSize = 1024;
    tilingDatafromBin->matmul_tiling_h.shareUbSize = 0;
    tilingDatafromBin->matmul_tiling_h.batchM = 1;
    tilingDatafromBin->matmul_tiling_h.batchN = 1;
    tilingDatafromBin->matmul_tiling_h.singleBatchM = 1;
    tilingDatafromBin->matmul_tiling_h.singleBatchN = 1;

    tilingDatafromBin->matmul_tiling_d.usedCoreNum = 1;
    tilingDatafromBin->matmul_tiling_d.M = 8;
    tilingDatafromBin->matmul_tiling_d.N = 128;
    tilingDatafromBin->matmul_tiling_d.Ka = 4;
    tilingDatafromBin->matmul_tiling_d.Kb = 4;
    tilingDatafromBin->matmul_tiling_d.singleCoreM = 8;
    tilingDatafromBin->matmul_tiling_d.singleCoreN = 128;
    tilingDatafromBin->matmul_tiling_d.singleCoreK = 6;
    tilingDatafromBin->matmul_tiling_d.baseM = 16;
    tilingDatafromBin->matmul_tiling_d.baseN = 128;
    tilingDatafromBin->matmul_tiling_d.baseK = 8;
    tilingDatafromBin->matmul_tiling_d.depthA1 = 1;
    tilingDatafromBin->matmul_tiling_d.depthB1 = 1;
    tilingDatafromBin->matmul_tiling_d.stepM = 1;
    tilingDatafromBin->matmul_tiling_d.stepN = 1;
    tilingDatafromBin->matmul_tiling_d.stepKa = 1;
    tilingDatafromBin->matmul_tiling_d.stepKb = 1;
    tilingDatafromBin->matmul_tiling_d.isBias = 0;
    tilingDatafromBin->matmul_tiling_d.transLength = 0;
    tilingDatafromBin->matmul_tiling_d.iterateOrder = 0;
    tilingDatafromBin->matmul_tiling_d.shareMode = 0;
    tilingDatafromBin->matmul_tiling_d.shareL1Size = 4608;
    tilingDatafromBin->matmul_tiling_d.shareL0CSize = 8192;
    tilingDatafromBin->matmul_tiling_d.shareUbSize = 0;
    tilingDatafromBin->matmul_tiling_d.batchM = 1;
    tilingDatafromBin->matmul_tiling_d.batchN = 1;
    tilingDatafromBin->matmul_tiling_d.singleBatchM = 1;
    tilingDatafromBin->matmul_tiling_d.singleBatchN = 1;

    ICPU_SET_TILING_KEY(2000);

    ICPU_RUN_KF(resize_upsample_trilinear, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    fileName = "./resize_upsample_trilinear_data/float16_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./resize_upsample_trilinear_data/ && python3 compare_data.py 'float16'");
}
