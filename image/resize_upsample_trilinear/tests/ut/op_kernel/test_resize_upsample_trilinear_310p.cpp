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
#include "../../../op_host/resize_upsample_trilinear_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

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

TEST_F(resize_upsample_trilinear_310p_test, test_case_float32)
{
    system(
        "cp -rf "
        "../../../../image/resize_upsample_trilinear/tests/ut/op_kernel/"
        "resize_upsample_trilinear_310p_data ./");
    system("chmod -R 755 ./resize_upsample_trilinear_310p_data/");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    struct ResizeUpsampleTrilinearCompileInfo {
        uint32_t totalCoreNum = 48;
    } compileInfo;
    string socVersion = "Ascend310p";
    gert::TilingContextPara tilingContextPara("ResizeUpsampleTrilinear",
                                                {{{{1, 2, 2, 1, 16}, {1, 2, 2, 1, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{1, 2, 1, 2, 4}, {1, 2, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 4})),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo, socVersion, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./resize_upsample_trilinear_310p_data/ && python3 gen_data.py '(1, 2, 2, 1, 16)' '(1, 2, 4)' 'float32'");

    size_t inputByteSize = 1 * 2 * 2 * 16 * sizeof(float);
    size_t outputByteSize = 1 * 2 * 4 * 16 * sizeof(float);
    uint32_t blockDim = tilingInfo.blockNum;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    std::string fileName = "./resize_upsample_trilinear_310p_data/float32_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    ICPU_RUN_KF(resize_upsample_trilinear, blockDim, x, y, workspace, tiling);
    fileName = "./resize_upsample_trilinear_310p_data/float32_output_trilinear.bin";
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
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    struct ResizeUpsampleTrilinearCompileInfo {
        uint32_t totalCoreNum = 48;
    } compileInfo;
    string socVersion = "Ascend310p";
    gert::TilingContextPara tilingContextPara("ResizeUpsampleTrilinear",
                                                {{{{1, 2, 2, 1, 16}, {1, 2, 2, 1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{1, 2, 1, 2, 4}, {1, 2, 1, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 4})),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo, socVersion, 48, 192 * 1024, 16384);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./resize_upsample_trilinear_310p_data/ && python3 gen_data.py '(1, 2, 2, 1, 16)' '(1, 2, 4)' 'float16'");

    size_t inputByteSize = 1 * 2 * 2 * 16 * sizeof(float);
    size_t outputByteSize = 1 * 2 * 4 * 16 * sizeof(float);
    uint32_t blockDim = tilingInfo.blockNum;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    std::string fileName = "./resize_upsample_trilinear_310p_data/float16_input_trilinear.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);


    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    ICPU_RUN_KF(resize_upsample_trilinear, blockDim, x, y, workspace, tiling);
    fileName = "./resize_upsample_trilinear_310p_data/float16_output_trilinear.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./resize_upsample_trilinear_310p_data/ && python3 compare_data.py 'float16'");
}
