/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "upsample_bicubic2d_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void upsample_bicubic2d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class upsample_bicubic2d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "upsample_bicubic2d_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "upsample_bicubic2d_test TearDown\n" << endl;
    }
};

TEST_F(upsample_bicubic2d_test, test_case_float32_1)
{
    system("cp -rf "
           "../../../../image/upsample_bicubic2d/tests/ut/op_kernel/upsample_bicubic2d_data ./");
    system("chmod -R 755 ./upsample_bicubic2d_data/");
    system("cd ./upsample_bicubic2d_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)' 'float32'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t inputByteSize = 4 * 4 * sizeof(float);
    size_t outputByteSize = 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleBicubic2dTilingData);
    size_t workspaceSize = 64 * 1024 * 1024;
    uint32_t numBlocks = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_bicubic2d_data/float32_input_bicubic2d.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleBicubic2dTilingData *tilingDatafromBin = reinterpret_cast<UpsampleBicubic2dTilingData *>(tiling);

    tilingDatafromBin->slide_size = 64;
    tilingDatafromBin->dataType = 2;
    tilingDatafromBin->scale_w = 0.25;
    tilingDatafromBin->scale_h = 0.25;
    tilingDatafromBin->align_corners = false;
    tilingDatafromBin->max_interp_size_w = 4;
    tilingDatafromBin->max_interp_size_h = 4;
    tilingDatafromBin->ratio_matrix_size_w = 1280;
    tilingDatafromBin->ratio_matrix_size_h = 1280;
    tilingDatafromBin->need_core_num_w = 1;
    tilingDatafromBin->need_core_num_h = 1;
    tilingDatafromBin->intermediate_matrix_size = 512;

    tilingDatafromBin->input_shapes[0] = 1;
    tilingDatafromBin->input_shapes[1] = 1;
    tilingDatafromBin->input_shapes[2] = 4;
    tilingDatafromBin->input_shapes[3] = 4;
    tilingDatafromBin->output_shapes[0] = 1;
    tilingDatafromBin->output_shapes[1] = 1;
    tilingDatafromBin->output_shapes[2] = 16;
    tilingDatafromBin->output_shapes[3] = 16;
    tilingDatafromBin->tailSlideStartList_w[0] = 0;
    tilingDatafromBin->tailSlideEndList_w[0] = 16;
    tilingDatafromBin->tailRowStartList_w[0] = 0;
    tilingDatafromBin->tailRowEndList_w[0] = 4;
    tilingDatafromBin->tailSlideStartList_h[0] = 0;
    tilingDatafromBin->tailSlideEndList_h[0] = 16;
    tilingDatafromBin->tailRowStartList_h[0] = 0;
    tilingDatafromBin->tailRowEndList_h[0] = 1;

    tilingDatafromBin->matmulTiling_w.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_w.M = 4;
    tilingDatafromBin->matmulTiling_w.N = 16;
    tilingDatafromBin->matmulTiling_w.Ka = 4;
    tilingDatafromBin->matmulTiling_w.Kb = 4;
    tilingDatafromBin->matmulTiling_w.singleCoreM = 4;
    tilingDatafromBin->matmulTiling_w.singleCoreN = 64;
    tilingDatafromBin->matmulTiling_w.singleCoreK = 20;
    tilingDatafromBin->matmulTiling_w.baseM = 16;
    tilingDatafromBin->matmulTiling_w.baseN = 64;
    tilingDatafromBin->matmulTiling_w.baseK = 24;
    tilingDatafromBin->matmulTiling_w.depthA1 = 1;
    tilingDatafromBin->matmulTiling_w.depthB1 = 1;
    tilingDatafromBin->matmulTiling_w.stepM = 1;
    tilingDatafromBin->matmulTiling_w.stepN = 1;
    tilingDatafromBin->matmulTiling_w.stepKa = 1;
    tilingDatafromBin->matmulTiling_w.stepKb = 1;
    tilingDatafromBin->matmulTiling_w.isBias = 0;
    tilingDatafromBin->matmulTiling_w.transLength = 0;
    tilingDatafromBin->matmulTiling_w.iterateOrder = 0;
    tilingDatafromBin->matmulTiling_w.shareMode = 0;
    tilingDatafromBin->matmulTiling_w.shareL1Size = 7680;
    tilingDatafromBin->matmulTiling_w.shareL0CSize = 4096;
    tilingDatafromBin->matmulTiling_w.shareUbSize = 0;
    tilingDatafromBin->matmulTiling_w.batchM = 1;
    tilingDatafromBin->matmulTiling_w.batchN = 1;
    tilingDatafromBin->matmulTiling_w.singleBatchM = 1;
    tilingDatafromBin->matmulTiling_w.singleBatchN = 1;

    tilingDatafromBin->matmulTiling_h.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_h.M = 16;
    tilingDatafromBin->matmulTiling_h.N = 16;
    tilingDatafromBin->matmulTiling_h.Ka = 4;
    tilingDatafromBin->matmulTiling_h.Kb = 4;
    tilingDatafromBin->matmulTiling_h.singleCoreM = 64;
    tilingDatafromBin->matmulTiling_h.singleCoreN = 16;
    tilingDatafromBin->matmulTiling_h.singleCoreK = 20;
    tilingDatafromBin->matmulTiling_h.baseM = 64;
    tilingDatafromBin->matmulTiling_h.baseN = 16;
    tilingDatafromBin->matmulTiling_h.baseK = 24;
    tilingDatafromBin->matmulTiling_h.depthA1 = 1;
    tilingDatafromBin->matmulTiling_h.depthB1 = 1;
    tilingDatafromBin->matmulTiling_h.stepM = 1;
    tilingDatafromBin->matmulTiling_h.stepN = 1;
    tilingDatafromBin->matmulTiling_h.stepKa = 1;
    tilingDatafromBin->matmulTiling_h.stepKb = 1;
    tilingDatafromBin->matmulTiling_h.isBias = 0;
    tilingDatafromBin->matmulTiling_h.transLength = 0;
    tilingDatafromBin->matmulTiling_h.iterateOrder = 0;
    tilingDatafromBin->matmulTiling_h.shareMode = 0;
    tilingDatafromBin->matmulTiling_h.shareL1Size = 7680;
    tilingDatafromBin->matmulTiling_h.shareL0CSize = 4096;
    tilingDatafromBin->matmulTiling_h.shareUbSize = 0;
    tilingDatafromBin->matmulTiling_h.batchM = 1;
    tilingDatafromBin->matmulTiling_h.batchN = 1;
    tilingDatafromBin->matmulTiling_h.singleBatchM = 1;
    tilingDatafromBin->matmulTiling_h.singleBatchN = 1;

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(upsample_bicubic2d, numBlocks, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./upsample_bicubic2d_data/float32_output_bicubic2d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bicubic2d_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_bicubic2d_test, test_case_float32_2)
{
    system("cp -rf "
           "../../../../image/upsample_bicubic2d/tests/ut/op_kernel/upsample_bicubic2d_data ./");
    system("chmod -R 755 ./upsample_bicubic2d_data/");
    system("cd ./upsample_bicubic2d_data/ && python3 gen_data.py '(1, 1, 21, 21)' '(3, 3)' 'float32'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t inputByteSize = 21 * 21 * sizeof(float);
    size_t outputByteSize = 3 * 3 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleBicubic2dTilingData);
    size_t workspaceSize = 64 * 1024 * 1024;
    uint32_t numBlocks = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_bicubic2d_data/float32_input_bicubic2d.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleBicubic2dTilingData *tilingDatafromBin = reinterpret_cast<UpsampleBicubic2dTilingData *>(tiling);

    tilingDatafromBin->slide_size = 48;
    tilingDatafromBin->dataType = 2;
    tilingDatafromBin->scale_w = 7;
    tilingDatafromBin->scale_h = 7;
    tilingDatafromBin->align_corners = false;
    tilingDatafromBin->max_interp_size_w = 4;
    tilingDatafromBin->max_interp_size_h = 4;
    tilingDatafromBin->ratio_matrix_size_w = 16320;
    tilingDatafromBin->ratio_matrix_size_h = 16320;
    tilingDatafromBin->need_core_num_w = 1;
    tilingDatafromBin->need_core_num_h = 1;
    tilingDatafromBin->intermediate_matrix_size = 512;

    tilingDatafromBin->input_shapes[0] = 1;
    tilingDatafromBin->input_shapes[1] = 1;
    tilingDatafromBin->input_shapes[2] = 21;
    tilingDatafromBin->input_shapes[3] = 21;
    tilingDatafromBin->output_shapes[0] = 1;
    tilingDatafromBin->output_shapes[1] = 1;
    tilingDatafromBin->output_shapes[2] = 3;
    tilingDatafromBin->output_shapes[3] = 3;
    tilingDatafromBin->tailSlideStartList_w[0] = 0;
    tilingDatafromBin->tailSlideEndList_w[0] = 3;
    tilingDatafromBin->tailRowStartList_w[0] = 0;
    tilingDatafromBin->tailRowEndList_w[0] = 21;
    tilingDatafromBin->tailSlideStartList_h[0] = 0;
    tilingDatafromBin->tailSlideEndList_h[0] = 3;
    tilingDatafromBin->tailRowStartList_h[0] = 0;
    tilingDatafromBin->tailRowEndList_h[0] = 1;

    tilingDatafromBin->matmulTiling_w.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_w.M = 21;
    tilingDatafromBin->matmulTiling_w.N = 3;
    tilingDatafromBin->matmulTiling_w.Ka = 21;
    tilingDatafromBin->matmulTiling_w.Kb = 21;
    tilingDatafromBin->matmulTiling_w.singleCoreM = 21;
    tilingDatafromBin->matmulTiling_w.singleCoreN = 48;
    tilingDatafromBin->matmulTiling_w.singleCoreK = 340;
    tilingDatafromBin->matmulTiling_w.baseM = 32;
    tilingDatafromBin->matmulTiling_w.baseN = 48;
    tilingDatafromBin->matmulTiling_w.baseK = 128;
    tilingDatafromBin->matmulTiling_w.depthA1 = 3;
    tilingDatafromBin->matmulTiling_w.depthB1 = 3;
    tilingDatafromBin->matmulTiling_w.stepM = 1;
    tilingDatafromBin->matmulTiling_w.stepN = 1;
    tilingDatafromBin->matmulTiling_w.stepKa = 3;
    tilingDatafromBin->matmulTiling_w.stepKb = 3;
    tilingDatafromBin->matmulTiling_w.isBias = 0;
    tilingDatafromBin->matmulTiling_w.transLength = 0;
    tilingDatafromBin->matmulTiling_w.iterateOrder = 0;
    tilingDatafromBin->matmulTiling_w.shareMode = 0;
    tilingDatafromBin->matmulTiling_w.shareL1Size = 122880;
    tilingDatafromBin->matmulTiling_w.shareL0CSize = 6144;
    tilingDatafromBin->matmulTiling_w.shareUbSize = 0;
    tilingDatafromBin->matmulTiling_w.batchM = 1;
    tilingDatafromBin->matmulTiling_w.batchN = 1;
    tilingDatafromBin->matmulTiling_w.singleBatchM = 1;
    tilingDatafromBin->matmulTiling_w.singleBatchN = 1;

    tilingDatafromBin->matmulTiling_h.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_h.M = 3;
    tilingDatafromBin->matmulTiling_h.N = 3;
    tilingDatafromBin->matmulTiling_h.Ka = 21;
    tilingDatafromBin->matmulTiling_h.Kb = 21;
    tilingDatafromBin->matmulTiling_h.singleCoreM = 48;
    tilingDatafromBin->matmulTiling_h.singleCoreN = 3;
    tilingDatafromBin->matmulTiling_h.singleCoreK = 340;
    tilingDatafromBin->matmulTiling_h.baseM = 48;
    tilingDatafromBin->matmulTiling_h.baseN = 16;
    tilingDatafromBin->matmulTiling_h.baseK = 128;
    tilingDatafromBin->matmulTiling_h.depthA1 = 3;
    tilingDatafromBin->matmulTiling_h.depthB1 = 3;
    tilingDatafromBin->matmulTiling_h.stepM = 1;
    tilingDatafromBin->matmulTiling_h.stepN = 1;
    tilingDatafromBin->matmulTiling_h.stepKa = 3;
    tilingDatafromBin->matmulTiling_h.stepKb = 3;
    tilingDatafromBin->matmulTiling_h.isBias = 0;
    tilingDatafromBin->matmulTiling_h.transLength = 0;
    tilingDatafromBin->matmulTiling_h.iterateOrder = 0;
    tilingDatafromBin->matmulTiling_h.shareMode = 0;
    tilingDatafromBin->matmulTiling_h.shareL1Size = 98304;
    tilingDatafromBin->matmulTiling_h.shareL0CSize = 3072;
    tilingDatafromBin->matmulTiling_h.shareUbSize = 0;
    tilingDatafromBin->matmulTiling_h.batchM = 1;
    tilingDatafromBin->matmulTiling_h.batchN = 1;
    tilingDatafromBin->matmulTiling_h.singleBatchM = 1;
    tilingDatafromBin->matmulTiling_h.singleBatchN = 1;

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(upsample_bicubic2d, numBlocks, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./upsample_bicubic2d_data/float32_output_bicubic2d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bicubic2d_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_bicubic2d_test, test_case_float16)
{
    system("cp -rf "
           "../../../../image/upsample_bicubic2d/tests/ut/op_kernel/upsample_bicubic2d_data ./");
    system("chmod -R 755 ./upsample_bicubic2d_data/");
    system("cd ./upsample_bicubic2d_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)' 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t inputByteSize = 4 * 4 * sizeof(half);
    size_t outputByteSize = 16 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(UpsampleBicubic2dTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_bicubic2d_data/float16_input_bicubic2d.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleBicubic2dTilingData *tilingDatafromBin = reinterpret_cast<UpsampleBicubic2dTilingData *>(tiling);

    tilingDatafromBin->slide_size = 64;
    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->scale_w = 0.25;
    tilingDatafromBin->scale_h = 0.25;
    tilingDatafromBin->align_corners = false;
    tilingDatafromBin->max_interp_size_w = 4;
    tilingDatafromBin->max_interp_size_h = 4;
    tilingDatafromBin->ratio_matrix_size_w = 1280;
    tilingDatafromBin->ratio_matrix_size_h = 1280;
    tilingDatafromBin->need_core_num_w = 1;
    tilingDatafromBin->need_core_num_h = 1;
    tilingDatafromBin->intermediate_matrix_size = 512;

    tilingDatafromBin->input_shapes[0] = 1;
    tilingDatafromBin->input_shapes[1] = 1;
    tilingDatafromBin->input_shapes[2] = 4;
    tilingDatafromBin->input_shapes[3] = 4;
    tilingDatafromBin->output_shapes[0] = 1;
    tilingDatafromBin->output_shapes[1] = 1;
    tilingDatafromBin->output_shapes[2] = 16;
    tilingDatafromBin->output_shapes[3] = 16;
    tilingDatafromBin->tailSlideStartList_w[0] = 0;
    tilingDatafromBin->tailSlideEndList_w[0] = 16;
    tilingDatafromBin->tailRowStartList_w[0] = 0;
    tilingDatafromBin->tailRowEndList_w[0] = 4;
    tilingDatafromBin->tailSlideStartList_h[0] = 0;
    tilingDatafromBin->tailSlideEndList_h[0] = 16;
    tilingDatafromBin->tailRowStartList_h[0] = 0;
    tilingDatafromBin->tailRowEndList_h[0] = 1;

    tilingDatafromBin->matmulTiling_w.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_w.M = 4;
    tilingDatafromBin->matmulTiling_w.N = 16;
    tilingDatafromBin->matmulTiling_w.Ka = 4;
    tilingDatafromBin->matmulTiling_w.Kb = 4;
    tilingDatafromBin->matmulTiling_w.singleCoreM = 4;
    tilingDatafromBin->matmulTiling_w.singleCoreN = 64;
    tilingDatafromBin->matmulTiling_w.singleCoreK = 20;
    tilingDatafromBin->matmulTiling_w.baseM = 16;
    tilingDatafromBin->matmulTiling_w.baseN = 64;
    tilingDatafromBin->matmulTiling_w.baseK = 24;
    tilingDatafromBin->matmulTiling_w.depthA1 = 1;
    tilingDatafromBin->matmulTiling_w.depthB1 = 1;
    tilingDatafromBin->matmulTiling_w.stepM = 1;
    tilingDatafromBin->matmulTiling_w.stepN = 1;
    tilingDatafromBin->matmulTiling_w.stepKa = 1;
    tilingDatafromBin->matmulTiling_w.stepKb = 1;
    tilingDatafromBin->matmulTiling_w.isBias = 0;
    tilingDatafromBin->matmulTiling_w.transLength = 0;
    tilingDatafromBin->matmulTiling_w.iterateOrder = 0;
    tilingDatafromBin->matmulTiling_w.shareMode = 0;
    tilingDatafromBin->matmulTiling_w.shareL1Size = 7680;
    tilingDatafromBin->matmulTiling_w.shareL0CSize = 4096;
    tilingDatafromBin->matmulTiling_w.shareUbSize = 0;
    tilingDatafromBin->matmulTiling_w.batchM = 1;
    tilingDatafromBin->matmulTiling_w.batchN = 1;
    tilingDatafromBin->matmulTiling_w.singleBatchM = 1;
    tilingDatafromBin->matmulTiling_w.singleBatchN = 1;

    tilingDatafromBin->matmulTiling_h.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_h.M = 16;
    tilingDatafromBin->matmulTiling_h.N = 16;
    tilingDatafromBin->matmulTiling_h.Ka = 4;
    tilingDatafromBin->matmulTiling_h.Kb = 4;
    tilingDatafromBin->matmulTiling_h.singleCoreM = 64;
    tilingDatafromBin->matmulTiling_h.singleCoreN = 16;
    tilingDatafromBin->matmulTiling_h.singleCoreK = 20;
    tilingDatafromBin->matmulTiling_h.baseM = 64;
    tilingDatafromBin->matmulTiling_h.baseN = 16;
    tilingDatafromBin->matmulTiling_h.baseK = 24;
    tilingDatafromBin->matmulTiling_h.depthA1 = 1;
    tilingDatafromBin->matmulTiling_h.depthB1 = 1;
    tilingDatafromBin->matmulTiling_h.stepM = 1;
    tilingDatafromBin->matmulTiling_h.stepN = 1;
    tilingDatafromBin->matmulTiling_h.stepKa = 1;
    tilingDatafromBin->matmulTiling_h.stepKb = 1;
    tilingDatafromBin->matmulTiling_h.isBias = 0;
    tilingDatafromBin->matmulTiling_h.transLength = 0;
    tilingDatafromBin->matmulTiling_h.iterateOrder = 0;
    tilingDatafromBin->matmulTiling_h.shareMode = 0;
    tilingDatafromBin->matmulTiling_h.shareL1Size = 7680;
    tilingDatafromBin->matmulTiling_h.shareL0CSize = 4096;
    tilingDatafromBin->matmulTiling_h.shareUbSize = 0;
    tilingDatafromBin->matmulTiling_h.batchM = 1;
    tilingDatafromBin->matmulTiling_h.batchN = 1;
    tilingDatafromBin->matmulTiling_h.singleBatchM = 1;
    tilingDatafromBin->matmulTiling_h.singleBatchN = 1;

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(upsample_bicubic2d, numBlocks, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./upsample_bicubic2d_data/float16_output_bicubic2d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bicubic2d_data/ && python3 compare_data.py 'float16'");
}
