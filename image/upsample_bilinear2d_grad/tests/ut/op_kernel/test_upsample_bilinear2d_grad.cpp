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
#include "upsample_bilinear2d_grad_tiling.h"
#include "../data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void upsample_bilinear2d_grad(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class upsample_bilinear2d_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "upsample_bilinear2d_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "upsample_bilinear2d_grad_test TearDown\n" << endl;
    }
};

TEST_F(upsample_bilinear2d_grad_test, test_case_float32)
{
    system(
        "cp -rf "
        "../../../../../../../ops/image/upsample_bilinear2d_grad/tests/ut/op_kernel/upsample_bilinear2d_grad_data ./");
    system("chmod -R 755 ./upsample_bilinear2d_grad_data/");
    system("cd ./upsample_bilinear2d_grad_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)' 'float32'");

    size_t inputByteSize = 4 * 4 * sizeof(float);
    size_t outputByteSize = 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleBilinear2dGradTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(inputByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    std::string fileName = "./upsample_bilinear2d_grad_data/float32_input_upsample_bilinear2d_grad.bin";
    ;
    ReadFile(fileName, outputByteSize, x, outputByteSize);

    UpsampleBilinear2dGradTilingData *tilingDatafromBin = reinterpret_cast<UpsampleBilinear2dGradTilingData *>(tiling);

    tilingDatafromBin->scale_w = 0.25;
    tilingDatafromBin->scale_h = 0.25;
    tilingDatafromBin->slide_size = 16;
    tilingDatafromBin->max_interp_size_w = 5;
    tilingDatafromBin->max_interp_size_h = 5;
    tilingDatafromBin->radio_matrix_size = 2208;    //
    tilingDatafromBin->radio_matrix_size_h = 2208;  //
    tilingDatafromBin->need_core_num_w = 1;
    tilingDatafromBin->need_core_num_h = 1;
    tilingDatafromBin->align_corners = 1;
    tilingDatafromBin->intermediate_matrix_size = 256;  //

    tilingDatafromBin->input_shapes[0] = 1;
    tilingDatafromBin->input_shapes[1] = 1;
    tilingDatafromBin->input_shapes[2] = 16;
    tilingDatafromBin->input_shapes[3] = 16;
    tilingDatafromBin->output_shapes[0] = 1;
    tilingDatafromBin->output_shapes[1] = 1;
    tilingDatafromBin->output_shapes[2] = 4;
    tilingDatafromBin->output_shapes[3] = 4;
    tilingDatafromBin->tailSlideStartList_w[0] = 0;
    tilingDatafromBin->tailSlideEndList_w[0] = 4;
    tilingDatafromBin->tailRowStartList_w[0] = 0;
    // tilingDatafromBin->tailRowEndList_w[0] = 4;
    tilingDatafromBin->tailRowEndList_w[0] = 16;
    tilingDatafromBin->tailSlideStartList_h[0] = 0;
    tilingDatafromBin->tailSlideEndList_h[0] = 4;
    tilingDatafromBin->tailRowStartList_h[0] = 0;
    tilingDatafromBin->tailRowEndList_h[0] = 0;

    tilingDatafromBin->matmulTiling_w.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_w.M = 16;
    tilingDatafromBin->matmulTiling_w.N = 4;
    tilingDatafromBin->matmulTiling_w.Ka = 16;
    tilingDatafromBin->matmulTiling_w.Kb = 16;
    tilingDatafromBin->matmulTiling_w.singleCoreM = 16;
    tilingDatafromBin->matmulTiling_w.singleCoreN = 16;
    tilingDatafromBin->matmulTiling_w.singleCoreK = 170;
    tilingDatafromBin->matmulTiling_w.baseM = 16;
    tilingDatafromBin->matmulTiling_w.baseN = 16;
    tilingDatafromBin->matmulTiling_w.baseK = 176;
    tilingDatafromBin->matmulTiling_w.depthA1 = 1;
    tilingDatafromBin->matmulTiling_w.depthB1 = 1;
    tilingDatafromBin->matmulTiling_w.stepM = 1;
    tilingDatafromBin->matmulTiling_w.stepN = 1;
    tilingDatafromBin->matmulTiling_w.isBias = 0;
    tilingDatafromBin->matmulTiling_w.transLength = 0;
    tilingDatafromBin->matmulTiling_w.iterateOrder = 0;
    tilingDatafromBin->matmulTiling_w.shareMode = 0;
    tilingDatafromBin->matmulTiling_w.shareL1Size = 22528;
    tilingDatafromBin->matmulTiling_w.shareL0CSize = 1024;
    tilingDatafromBin->matmulTiling_w.shareUbSize = 0;
    tilingDatafromBin->matmulTiling_w.batchM = 1;
    tilingDatafromBin->matmulTiling_w.batchN = 1;
    tilingDatafromBin->matmulTiling_w.singleBatchM = 1;
    tilingDatafromBin->matmulTiling_w.singleBatchN = 1;
    tilingDatafromBin->matmulTiling_w.stepKa = 1;
    tilingDatafromBin->matmulTiling_w.stepKb = 1;
    tilingDatafromBin->matmulTiling_h.usedCoreNum = 1;
    tilingDatafromBin->matmulTiling_h.M = 4;
    tilingDatafromBin->matmulTiling_h.N = 4;
    tilingDatafromBin->matmulTiling_h.Ka = 16;
    tilingDatafromBin->matmulTiling_h.Kb = 16;
    tilingDatafromBin->matmulTiling_h.singleCoreM = 16;
    tilingDatafromBin->matmulTiling_h.singleCoreN = 4;
    tilingDatafromBin->matmulTiling_h.singleCoreK = 170;
    tilingDatafromBin->matmulTiling_h.baseM = 16;
    tilingDatafromBin->matmulTiling_h.baseN = 16;
    tilingDatafromBin->matmulTiling_h.baseK = 176;
    tilingDatafromBin->matmulTiling_h.depthA1 = 1;
    tilingDatafromBin->matmulTiling_h.depthB1 = 1;
    tilingDatafromBin->matmulTiling_h.stepM = 1;
    tilingDatafromBin->matmulTiling_h.stepN = 1;
    tilingDatafromBin->matmulTiling_h.isBias = 0;
    tilingDatafromBin->matmulTiling_h.transLength = 0;
    tilingDatafromBin->matmulTiling_h.iterateOrder = 0;
    tilingDatafromBin->matmulTiling_h.shareMode = 0;
    tilingDatafromBin->matmulTiling_h.shareL1Size = 22528;
    tilingDatafromBin->matmulTiling_h.shareL0CSize = 1024;
    tilingDatafromBin->matmulTiling_h.shareUbSize = 0;
    tilingDatafromBin->matmulTiling_h.batchM = 1;
    tilingDatafromBin->matmulTiling_h.batchN = 1;
    tilingDatafromBin->matmulTiling_h.singleBatchM = 1;
    tilingDatafromBin->matmulTiling_h.singleBatchN = 1;
    tilingDatafromBin->matmulTiling_h.stepKa = 1;
    tilingDatafromBin->matmulTiling_h.stepKb = 1;

    ICPU_SET_TILING_KEY(2);

    ICPU_RUN_KF(upsample_bilinear2d_grad, blockDim, x, y, workspace, (uint8_t *)(tilingDatafromBin));
    fileName = "./upsample_bilinear2d_grad_data/float32_output_bilinear2d_grad.bin";
    WriteFile(fileName, y, inputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bilinear2d_grad_data/ && python3 compare_data.py 'float32'");
}
