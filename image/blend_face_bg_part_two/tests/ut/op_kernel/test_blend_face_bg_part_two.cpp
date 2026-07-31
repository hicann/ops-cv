/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "data_utils.h"
#include "../../../op_kernel/arch35/blend_face_bg_part_two_tiling_data.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace optiling;

extern "C" __global__ __aicore__ void blend_face_bg_part_two(GM_ADDR acc_face, GM_ADDR acc_mask, GM_ADDR max_mask,
                                                             GM_ADDR bg_img, GM_ADDR fused_img, GM_ADDR workspace,
                                                             GM_ADDR tiling);

class blend_face_bg_part_two_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "blend_face_bg_part_two SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "blend_face_bg_part_two TearDown\n" << std::endl; }
};

TEST_F(blend_face_bg_part_two_test, test_float_case_4x4x3)
{
    uint32_t H = 4;
    uint32_t W = 4;
    uint32_t C = 3;
    float epsilon = 1e-12f;

    size_t inputByteSize = H * W * C * sizeof(float);
    Tiling4BlendFaceBgPartTwoCompileInfo compileInfo = {48, 196608, true}; // coreNum=48, ubSize=196608, regBase=true

    gert::TilingContextPara tilingContextPara(
        "BlendFaceBgPartTwo",
        {{{{H, W, C}, {H, W, C}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{H, W, C}, {H, W, C}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{H, W, C}, {H, W, C}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{H, W, C}, {H, W, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{H, W, C}, {H, W, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("epsilon", Ops::Cv::AnyValue::CreateFrom<float>(epsilon))}, &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t* acc_face = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* acc_mask = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* max_mask = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* bg_img = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* fused_img = (uint8_t*)AscendC::GmAlloc(inputByteSize);

    uint32_t numBlocks = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(blend_face_bg_part_two, numBlocks, acc_face, acc_mask, max_mask, bg_img, fused_img, workspace, tiling);

    AscendC::GmFree(acc_face);
    AscendC::GmFree(acc_mask);
    AscendC::GmFree(max_mask);
    AscendC::GmFree(bg_img);
    AscendC::GmFree(fused_img);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(blend_face_bg_part_two_test, test_uint8_bg_case_8x8x3)
{
    uint32_t H = 8;
    uint32_t W = 8;
    uint32_t C = 3;
    float epsilon = 1e-12f;

    size_t floatByteSize = H * W * C * sizeof(float);
    size_t uint8ByteSize = H * W * C * sizeof(uint8_t);
    Tiling4BlendFaceBgPartTwoCompileInfo compileInfo = {48, 196608, true};

    gert::TilingContextPara tilingContextPara(
        "BlendFaceBgPartTwo",
        {{{{H, W, C}, {H, W, C}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{H, W, C}, {H, W, C}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{H, W, C}, {H, W, C}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{H, W, C}, {H, W, C}}, ge::DT_UINT8, ge::FORMAT_ND}}, // bg_img 为 uint8
        {{{{H, W, C}, {H, W, C}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("epsilon", Ops::Cv::AnyValue::CreateFrom<float>(epsilon))}, &compileInfo);

    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t* acc_face = (uint8_t*)AscendC::GmAlloc(floatByteSize);
    uint8_t* acc_mask = (uint8_t*)AscendC::GmAlloc(floatByteSize);
    uint8_t* max_mask = (uint8_t*)AscendC::GmAlloc(floatByteSize);
    uint8_t* bg_img = (uint8_t*)AscendC::GmAlloc(uint8ByteSize);
    uint8_t* fused_img = (uint8_t*)AscendC::GmAlloc(floatByteSize);

    uint32_t numBlocks = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(blend_face_bg_part_two, numBlocks, acc_face, acc_mask, max_mask, bg_img, fused_img, workspace, tiling);

    AscendC::GmFree(acc_face);
    AscendC::GmFree(acc_mask);
    AscendC::GmFree(max_mask);
    AscendC::GmFree(bg_img);
    AscendC::GmFree(fused_img);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
