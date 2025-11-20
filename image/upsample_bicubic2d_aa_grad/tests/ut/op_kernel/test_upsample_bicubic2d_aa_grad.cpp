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
#include "data_utils.h"
#include "tiling_case_executor.h"
#include "../../../op_host/upsample_bicubic2d_aa_grad_tiling.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void upsample_bicubic2d_aa_grad(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class upsample_bicubic2d_aa_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "upsample_bicubic2d_aa_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "upsample_bicubic2d_aa_grad_test TearDown\n" << endl;
    }
};

TEST_F(upsample_bicubic2d_aa_grad_test, test_case_float32)
{
    system("cp -rf ../../../../image/upsample_bicubic2d_aa_grad/tests/ut/op_kernel/upsample_bicubic2d_aa_grad_data ./");
    system("chmod -R 755 ./upsample_bicubic2d_aa_grad_data/");
    system("cd ./upsample_bicubic2d_aa_grad_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)' 'float32'");

    struct UpsampleBicubic2dAAGradCompileInfo {
        uint32_t coreNum = 24;
    } compile_info;

    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dAAGrad",
                                                {{{{1, 1, 16, 16}, {1, 1, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({16, 16})),
                                                gert::TilingContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({1, 1, 4, 4})),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compile_info, 48, 192*1024, 8192);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputByteSize = 4 * 4 * sizeof(float);
    size_t outputByteSize = 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(UpsampleBicubicAAGradTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t blockDim = 1;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    std::string fileName = "./upsample_bicubic2d_aa_grad_data/float32_input_bicubic2d_aa_grad.bin";
    ReadFile(fileName, outputByteSize, x, outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(upsample_bicubic2d_aa_grad, blockDim, x, y, workspace, tiling);
    fileName = "./upsample_bicubic2d_aa_grad_data/float32_output_bicubic2d_aa_grad.bin";
    WriteFile(fileName, y, inputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bicubic2d_aa_grad_data/ && python3 compare_data.py 'float32'");
}
