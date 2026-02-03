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
#include "../../../op_host/upsample_bicubic2d_grad_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void upsample_bicubic2d_grad(
    GM_ADDR grad_output, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling_addr);

class upsample_bicubic2d_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "upsample_bicubic2d_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "upsample_bicubic2d_grad_test TearDown\n" << endl;
    }
};

TEST_F(upsample_bicubic2d_grad_test, test_case_1)
{
    struct UpsampleBicubic2dGradCompileInfo {
        uint32_t aicNum = 20;
        uint32_t aivNum = 40;
        uint64_t ubSize = 196608;
        uint64_t l1Size = 524288;
        uint64_t l2Size = 201326592;
        uint64_t l0CSize = 131072;
        uint64_t l0ASize = 65536;
        uint64_t l0BSize = 65536;
        platform_ascendc::SocVersion socVersion;
        std::string socVersionStr = "";
    } compile_info;
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dGrad",
                                                {{{{2, 16, 64, 64}, {2, 16, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{2, 16, 16, 16}, {2, 16, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.5))},
                                                &compile_info);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputByteSize = 2 * 16 * 64 * 64 * sizeof(half);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    size_t outputByteSize = 2 * 16 * 16 * 16 * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    uint32_t numBlocks = 20;
    ICPU_RUN_KF(upsample_bicubic2d_grad, numBlocks, x, y, workspace, tiling);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(upsample_bicubic2d_grad_test, test_case_2)
{
    struct UpsampleBicubic2dGradCompileInfo {
        uint32_t aicNum = 20;
        uint32_t aivNum = 40;
        uint64_t ubSize = 196608;
        uint64_t l1Size = 524288;
        uint64_t l2Size = 201326592;
        uint64_t l0CSize = 131072;
        uint64_t l0ASize = 65536;
        uint64_t l0BSize = 65536;
        platform_ascendc::SocVersion socVersion;
        std::string socVersionStr = "";
    } compile_info;
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dGrad",
                                                {{{{2, 16, 64, 64}, {2, 16, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 16, 16, 16}, {2, 16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.5))},
                                                &compile_info);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputByteSize = 2 * 16 * 64 * 64 * sizeof(float);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    size_t outputByteSize = 2 * 16 * 16 * 16 * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    uint32_t numBlocks = 20;
    ICPU_RUN_KF(upsample_bicubic2d_grad, numBlocks, x, y, workspace, tiling);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(upsample_bicubic2d_grad_test, test_case_3)
{
    struct UpsampleBicubic2dGradCompileInfo {
        uint32_t aicNum = 20;
        uint32_t aivNum = 40;
        uint64_t ubSize = 196608;
        uint64_t l1Size = 524288;
        uint64_t l2Size = 201326592;
        uint64_t l0CSize = 131072;
        uint64_t l0ASize = 65536;
        uint64_t l0BSize = 65536;
        platform_ascendc::SocVersion socVersion;
        std::string socVersionStr = "";
    } compile_info;
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dGrad",
                                                {{{{2, 16, 64, 64}, {2, 16, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{{{2, 16, 16, 16}, {2, 16, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.5))},
                                                &compile_info);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputByteSize = 2 * 16 * 64 * 64 * sizeof(half);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    size_t outputByteSize = 2 * 16 * 16 * 16 * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(10000002);
    uint32_t numBlocks = 20;
    ICPU_RUN_KF(upsample_bicubic2d_grad, numBlocks, x, y, workspace, tiling);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(upsample_bicubic2d_grad_test, test_case_4)
{
    struct UpsampleBicubic2dGradCompileInfo {
        uint32_t aicNum = 20;
        uint32_t aivNum = 40;
        uint64_t ubSize = 196608;
        uint64_t l1Size = 524288;
        uint64_t l2Size = 201326592;
        uint64_t l0CSize = 131072;
        uint64_t l0ASize = 65536;
        uint64_t l0BSize = 65536;
        platform_ascendc::SocVersion socVersion;
        std::string socVersionStr = "";
    } compile_info;
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dGrad",
                                                {{{{2, 16, 64, 64}, {2, 16, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{2, 16, 16, 16}, {2, 16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.5))},
                                                &compile_info);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    size_t inputByteSize = 2 * 16 * 64 * 64 * sizeof(float);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    size_t outputByteSize = 2 * 16 * 16 * 16 * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(10000002);
    uint32_t numBlocks = 20;
    ICPU_RUN_KF(upsample_bicubic2d_grad, numBlocks, x, y, workspace, tiling);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
