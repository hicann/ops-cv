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
 * \file test_upsample_bilinear2d_aa.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tensor_list_operate.h"

extern "C" __global__ __aicore__ void upsample_bilinear2d_aa(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class upsample_bilinear2d_aa_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "upsample_bilinear2d_aa_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "upsample_bilinear2d_aa_test TearDown\n" << std::endl;
    }
};

TEST_F(upsample_bilinear2d_aa_test, test_case_float_1)
{
    system("cp -rf "
           "../../../../../../../ops/image/upsample_bilinear2d_aa/tests/ut/op_kernel/cos_data ./");
    system("chmod -R 755 ./cos_data/");
    system("cd ./cos_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)'  'float32'");

    size_t inputByteSize = 4 * 4 * sizeof(float);
    size_t outputByteSize = 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleBilinearAATilingData);
    size_t workspaceSize = 64 * 1024 * 1024;
    uint32_t blockDim = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./cos_data/float32_input_upsample_bilinear2d_aa.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleBilinearAATilingData *tilingDatafromBin = reinterpret_cast<UpsampleBilinearAATilingData *>(tiling);

    tilingDatafromBin->slide_size = 64;
    tilingDatafromBin->dataType = 2;
    tilingDatafromBin->scale_w = 0.25;
    tilingDatafromBin->scale_h = 0.25;
    tilingDatafromBin->invscale_w = 1.0;
    tilingDatafromBin->invscale_h = 1.0;
    tilingDatafromBin->support_w = 1;
    tilingDatafromBin->support_h = 1;
    tilingDatafromBin->max_interp_size_w = 3;
    tilingDatafromBin->max_interp_size_h = 3;
    tilingDatafromBin->radio_matrix_size_w = 1216;
    tilingDatafromBin->radio_matrix_size_h = 1216;
    tilingDatafromBin->need_core_num_w = 1;
    tilingDatafromBin->need_core_num_h = 1;
    tilingDatafromBin->intermediate_matrix_size = 512;
    tilingDatafromBin->eachCoreSlideNumW = 0;
    tilingDatafromBin->tailStartSlideNumW = 0;
    tilingDatafromBin->slideNumW = 1;
    tilingDatafromBin->groupCoreNumW = 1;
    tilingDatafromBin->tailAvergingRowsW = 64;
    tilingDatafromBin->remainderW = 1;
    tilingDatafromBin->eachCoreSlideNumH = 0;
    tilingDatafromBin->tailStartSlideNumH = 0;
    tilingDatafromBin->slideNumH = 1;
    tilingDatafromBin->groupCoreNumH = 1;
    tilingDatafromBin->tailAvergingRowsH = 1;
    tilingDatafromBin->remainderH = 1;

    tilingDatafromBin->input_shapes[0] = 1;
    tilingDatafromBin->input_shapes[1] = 1;
    tilingDatafromBin->input_shapes[2] = 4;
    tilingDatafromBin->input_shapes[3] = 4;
    tilingDatafromBin->output_shapes[0] = 1;
    tilingDatafromBin->output_shapes[1] = 1;
    tilingDatafromBin->output_shapes[2] = 16;
    tilingDatafromBin->output_shapes[3] = 16;

    tilingDatafromBin->matmulTiling_w.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_w.M = 4;
    tilingDatafromBin->matmulTiling_w.N = 16;
    tilingDatafromBin->matmulTiling_w.Ka = 4;
    tilingDatafromBin->matmulTiling_w.Kb = 4;
    tilingDatafromBin->matmulTiling_w.singleCoreM = 4;
    tilingDatafromBin->matmulTiling_w.singleCoreN = 64;
    tilingDatafromBin->matmulTiling_w.singleCoreK = 19;
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
    tilingDatafromBin->matmulTiling_h.singleCoreK = 19;
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

    ICPU_RUN_KF(upsample_bilinear2d_aa, blockDim, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./cos_data/float32_output_upsample_bilinear2d_aa.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./cos_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_bilinear2d_aa_test, test_case_float16_2)
{
    system("cp -rf "
           "../../../../../../../ops/image/upsample_bilinear2d_aa/tests/ut/op_kernel/cos_data ./");
    system("chmod -R 755 ./cos_data/");
    system("cd ./cos_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)'  'float16'");

    size_t inputByteSize = 4 * 4 * sizeof(half);
    size_t outputByteSize = 16 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(UpsampleBilinearAATilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./cos_data/float16_input_upsample_bilinear2d_aa.bin";

    ReadFile(fileName, inputByteSize, x, inputByteSize);

    UpsampleBilinearAATilingData *tilingDatafromBin = reinterpret_cast<UpsampleBilinearAATilingData *>(tiling);

    tilingDatafromBin->slide_size = 64;
    tilingDatafromBin->dataType = 1;
    tilingDatafromBin->scale_w = 0.25;
    tilingDatafromBin->scale_h = 0.25;
    tilingDatafromBin->invscale_w = 1.0;
    tilingDatafromBin->invscale_h = 1.0;
    tilingDatafromBin->support_w = 1;
    tilingDatafromBin->support_h = 1;
    tilingDatafromBin->max_interp_size_w = 3;
    tilingDatafromBin->max_interp_size_h = 3;
    tilingDatafromBin->radio_matrix_size_w = 1216;
    tilingDatafromBin->radio_matrix_size_h = 1216;
    tilingDatafromBin->need_core_num_w = 1;
    tilingDatafromBin->need_core_num_h = 1;
    tilingDatafromBin->intermediate_matrix_size = 512;
    tilingDatafromBin->eachCoreSlideNumW = 0;
    tilingDatafromBin->tailStartSlideNumW = 0;
    tilingDatafromBin->slideNumW = 1;
    tilingDatafromBin->groupCoreNumW = 1;
    tilingDatafromBin->tailAvergingRowsW = 64;
    tilingDatafromBin->remainderW = 1;
    tilingDatafromBin->eachCoreSlideNumH = 0;
    tilingDatafromBin->tailStartSlideNumH = 0;
    tilingDatafromBin->slideNumH = 1;
    tilingDatafromBin->groupCoreNumH = 1;
    tilingDatafromBin->tailAvergingRowsH = 1;
    tilingDatafromBin->remainderH = 1;

    tilingDatafromBin->input_shapes[0] = 1;
    tilingDatafromBin->input_shapes[1] = 1;
    tilingDatafromBin->input_shapes[2] = 4;
    tilingDatafromBin->input_shapes[3] = 4;
    tilingDatafromBin->output_shapes[0] = 1;
    tilingDatafromBin->output_shapes[1] = 1;
    tilingDatafromBin->output_shapes[2] = 16;
    tilingDatafromBin->output_shapes[3] = 16;

    tilingDatafromBin->matmulTiling_w.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_w.M = 4;
    tilingDatafromBin->matmulTiling_w.N = 16;
    tilingDatafromBin->matmulTiling_w.Ka = 4;
    tilingDatafromBin->matmulTiling_w.Kb = 4;
    tilingDatafromBin->matmulTiling_w.singleCoreM = 4;
    tilingDatafromBin->matmulTiling_w.singleCoreN = 64;
    tilingDatafromBin->matmulTiling_w.singleCoreK = 19;
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
    tilingDatafromBin->matmulTiling_h.singleCoreK = 19;
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

    ICPU_RUN_KF(upsample_bilinear2d_aa, blockDim, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./cos_data/float16_output_upsample_bilinear2d_aa.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./cos_data/ && python3 compare_data.py 'float16'");
}
