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
#include "data_utils.h"
#include "tiling_case_executor.h"
#include "../../../op_host/upsample_bilinear2d_aa_tiling.h"

using namespace optiling;

extern "C" __global__ __aicore__ void upsample_bilinear2d_aa(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class upsample_bilinear2d_aa_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "upsample_bilinear2d_aa_test SetUp\n" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./cos_data/");
    }
    static void TearDownTestCase()
    {
        std::cout << "upsample_bilinear2d_aa_test TearDown\n" << std::endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string upsample_bilinear2d_aa_test::rootPath = "../../../../";
const std::string upsample_bilinear2d_aa_test::dataPath = rootPath + "image/upsample_bilinear2d_aa/tests/ut/op_kernel/cos_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b) {
    return (a + b - 1) / b * b;
}

TEST_F(upsample_bilinear2d_aa_test, test_case_float_1) {
    optiling::UpsampleBilinear2dAACompileInfo compileInfo = {24};
    std::vector<int64_t> output_size = {16, 16};
    gert::TilingContextPara tilingContextPara("UpsampleBilinear2dAA",
                                              {{{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{1, 1, 16, 16}, {1, 1, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
                                                {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
                                                {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
                                                {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
                                              &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./cos_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)'  'float32'");
    uint32_t dataCount = 1 * 1 * 4 * 4;
    size_t inputByteSize = dataCount * sizeof(float);
    std::string fileName = "./cos_data/float32_input_upsample_bilinear2d_aa.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = 1 * 1 * 16 * 16 * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(upsample_bilinear2d_aa, tilingInfo.blockNum, x, y, workspace, tiling);

    fileName = "./cos_data/float32_output_upsample_bilinear2d_aa.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./cos_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_bilinear2d_aa_test, test_case_float16_2) {
    optiling::UpsampleBilinear2dAACompileInfo compileInfo = {24};
    std::vector<int64_t> output_size = {16, 16};
    gert::TilingContextPara tilingContextPara("UpsampleBilinear2dAA",
                                              {{{{1, 1, 4, 4}, {1, 1, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {{{{1, 1, 16, 16}, {1, 1, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
                                                {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
                                                {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
                                                {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
                                              &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./cos_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)'  'float16'");
    uint32_t dataCount = 1 * 1 * 4 * 4;
    size_t inputByteSize = dataCount * sizeof(half);
    std::string fileName = "./cos_data/float16_input_upsample_bilinear2d_aa.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = 1 * 1 * 16 * 16 * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(upsample_bilinear2d_aa, tilingInfo.blockNum, x, y, workspace, tiling);

    fileName = "./cos_data/float16_output_upsample_bilinear2d_aa.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./cos_data/ && python3 compare_data.py 'float16'");
}