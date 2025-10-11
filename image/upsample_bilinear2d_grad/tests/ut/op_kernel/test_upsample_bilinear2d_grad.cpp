/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
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
#include "../../../op_host/upsample_bilinear2d_grad_tiling.h"

using namespace std;
using namespace optiling;

extern "C" __global__ __aicore__ void upsample_bilinear2d_grad(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class upsample_bilinear2d_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "upsample_bilinear2d_grad_test SetUp\n" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./upsample_bilinear2d_grad_data/");
    }
    static void TearDownTestCase()
    {
        cout << "upsample_bilinear2d_grad_test TearDown\n" << endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string upsample_bilinear2d_grad_test::rootPath = "../../../../";
const std::string upsample_bilinear2d_grad_test::dataPath =
    rootPath + "image/upsample_bilinear2d_grad/tests/ut/op_kernel/upsample_bilinear2d_grad_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(upsample_bilinear2d_grad_test, test_case_float_1)
{
    optiling::UpsampleBilinear2dGradCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {4, 4};
    std::vector<int64_t> input_size = {1, 1, 16, 16};
    gert::TilingContextPara tilingContextPara("UpsampleBilinear2dGrad",
        {
            {{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 16, 16}, {1, 1, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
            {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)},
            {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
            {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo,
        48,
        196608,
        8192);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./upsample_bilinear2d_grad_data/ && python3 gen_data.py '(1, 1, 16, 16)' '(4, 4)'  'float32'");
    size_t inputByteSize = 1 * 1 * 4 * 4 * sizeof(float);
    std::string fileName = "./upsample_bilinear2d_grad_data/float32_input_upsample_bilinear2d_grad.bin";
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = 1 * 1 * 16 * 16 * sizeof(float);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(upsample_bilinear2d_grad, tilingInfo.blockNum, x, y, workspace, tiling);

    fileName = "./upsample_bilinear2d_grad_data/float32_output_bilinear2d_grad.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bilinear2d_grad_data/ && python3 compare_data.py 'float32'");
}