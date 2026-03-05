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
#include "../../../op_host/upsample_bicubic2d_tiling.h"

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
struct UpsampleBicubic2dCompileInfo {
    uint16_t totalCoreNum = 0;
    uint16_t socVersionType = 220;
};

TEST_F(upsample_bicubic2d_test, test_case_float32_1)
{
    system("cp -rf "
           "../../../../image/upsample_bicubic2d/tests/ut/op_kernel/upsample_bicubic2d_data ./");
    system("chmod -R 755 ./upsample_bicubic2d_data/");
    system("cd ./upsample_bicubic2d_data/ && python3 gen_data.py '(1, 1, 16, 16)' '(4, 4)' 'float32'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    gert::StorageShape input_shape = {{1, 1, 16, 16}, {1, 1, 16, 16}};
    gert::StorageShape out_shape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    UpsampleBicubic2dCompileInfo compileInfo = {24, 220};
    std::vector<int64_t> output_size = {4, 4};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2d",
                                                {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo, socVersion, 48, 192*1024, 8192);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputByteSize = 16 * 16 * sizeof(float);
    size_t outputByteSize = 4 * 4 * sizeof(float);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    std::string fileName = "./upsample_bicubic2d_data/float32_input_bicubic2d.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(upsample_bicubic2d, tilingInfo.blockNum, x, y, workspace, tiling);
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

    gert::StorageShape input_shape = {{1, 1, 21, 21}, {1, 1, 21, 21}};
    gert::StorageShape out_shape = {{1, 1, 3, 3}, {1, 1, 3, 3}};
    UpsampleBicubic2dCompileInfo compileInfo = {24, 220};
    std::vector<int64_t> output_size = {3, 3};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2d",
                                                {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo, socVersion, 48, 192*1024, 8192);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputByteSize = 21 * 21 * sizeof(float);
    size_t outputByteSize = 3 * 3 * sizeof(float);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    std::string fileName = "./upsample_bicubic2d_data/float32_input_bicubic2d.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(upsample_bicubic2d, tilingInfo.blockNum, x, y, workspace, tiling);
    fileName = "./upsample_bicubic2d_data/float32_output_bicubic2d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bicubic2d_data/ && python3 compare_data.py 'float32'");
}

TEST_F(upsample_bicubic2d_test, test_case_float16_1)
{
    system("cp -rf "
           "../../../../image/upsample_bicubic2d/tests/ut/op_kernel/upsample_bicubic2d_data ./");
    system("chmod -R 755 ./upsample_bicubic2d_data/");
    system("cd ./upsample_bicubic2d_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)' 'float16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    gert::StorageShape input_shape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape out_shape = {{1, 1, 16, 16}, {1, 1, 16, 16}};
    UpsampleBicubic2dCompileInfo compileInfo = {24, 220};
    std::vector<int64_t> output_size = {16, 16};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2d",
                                                {{input_shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{out_shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo, socVersion, 48, 192*1024, 8192);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputByteSize = 4 * 4 * sizeof(half);
    size_t outputByteSize = 16 * 16 * sizeof(half);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    std::string fileName = "./upsample_bicubic2d_data/float16_input_bicubic2d.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(upsample_bicubic2d, tilingInfo.blockNum, x, y, workspace, tiling);
    fileName = "./upsample_bicubic2d_data/float16_output_bicubic2d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bicubic2d_data/ && python3 compare_data.py 'float16'");
}

TEST_F(upsample_bicubic2d_test, test_case_bfloat16_1)
{
    system("cp -rf "
           "../../../../image/upsample_bicubic2d/tests/ut/op_kernel/upsample_bicubic2d_data ./");
    system("chmod -R 755 ./upsample_bicubic2d_data/");
    system("cd ./upsample_bicubic2d_data/ && python3 gen_data.py '(1, 1, 4, 4)' '(16, 16)' 'bfloat16'");
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    gert::StorageShape input_shape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape out_shape = {{1, 1, 16, 16}, {1, 1, 16, 16}};
    UpsampleBicubic2dCompileInfo compileInfo = {24, 220};
    std::vector<int64_t> output_size = {16, 16};
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2d",
                                                {{input_shape, ge::DT_BF16, ge::FORMAT_ND}},
                                                {{out_shape, ge::DT_BF16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo, socVersion, 48, 192*1024, 8192);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputByteSize = 4 * 4 * sizeof(half);
    size_t outputByteSize = 16 * 16 * sizeof(half);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    std::string fileName = "./upsample_bicubic2d_data/bfloat16_input_bicubic2d.bin";
    ReadFile(fileName, inputByteSize, x, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(upsample_bicubic2d, tilingInfo.blockNum, x, y, workspace, tiling);
    fileName = "./upsample_bicubic2d_data/bfloat16_output_bicubic2d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void *)(x));
    AscendC::GmFree((void *)(y));
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    system("cd ./upsample_bicubic2d_data/ && python3 compare_data.py 'bfloat16'");
}