/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../../op_host/arch35/resize_bicubic_v2_grad_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;
using namespace optiling;

class ResizeBicubicV2GradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResizeBicubicV2GradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResizeBicubicV2GradTilingTest TearDown" << std::endl; }
};

TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_01)
{
    gert::StorageShape inputGradsShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    gert::StorageShape inputOriImageShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    gert::StorageShape outputShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 30000;
    string expectTilingData = "64 48 0 32736 ";
    // 非 split-K 路径 workspace 即系统预留区 GetLibApiWorkSpaceSize()。该值由测试 faker 的
    // 平台描述符决定, 本仓 ascend950 UT 环境下为 UINT32_MAX(4294967295), 与同仓 col2im 用例一致。
    std::vector<size_t> expectWorkspaces = {4294967295};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_02)
{
    gert::StorageShape inputGradsShape = {{1, 3, 225, 32}, {1, 3, 225, 32}};
    gert::StorageShape inputOriImageShape = {{1, 3, 113, 32}, {1, 3, 113, 32}};
    gert::StorageShape outputShape = {{1, 3, 113, 32}, {1, 3, 113, 32}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({2.0f, 2.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 20000;
    // 尾部 3 个字段 (0 1 1) 为新增的 splitK/coresPerOutput/segsPerOutput, 非 split-K 场景取默认值。
    string expectTilingData = "3 113 32 225 32 0 1 64 169 32 4575657222465388544 4575657222482165760 0 1 1 ";
    std::vector<size_t> expectWorkspaces = {4294967295};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_03)
{
    gert::StorageShape inputGradsShape = {{32, 2048, 4096, 32}, {32, 2048, 4096, 32}};
    gert::StorageShape inputOriImageShape = {{32, 2048, 4096, 32}, {32, 2048, 4096, 32}};
    gert::StorageShape outputShape = {{32, 2048, 4096, 32}, {32, 2048, 4096, 32}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({2.0f, 2.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 30000;
    string expectTilingData = "64 134217728 0 32736 ";
    std::vector<size_t> expectWorkspaces = {4294967295};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 04: NCHW downsampling with scales
TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_04)
{
    gert::StorageShape inputGradsShape = {{1, 3, 16, 16}, {1, 3, 16, 16}};
    gert::StorageShape inputOriImageShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};
    gert::StorageShape outputShape = {{1, 3, 32, 32}, {1, 3, 32, 32}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.5f, 0.5f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 05: NHWC with half_pixel_centers (simulated) upsampling
TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_05)
{
    gert::StorageShape inputGradsShape = {{2, 4, 6, 64}, {2, 4, 6, 64}};
    gert::StorageShape inputOriImageShape = {{2, 2, 3, 64}, {2, 2, 3, 64}};
    gert::StorageShape outputShape = {{2, 2, 3, 64}, {2, 2, 3, 64}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({2.0f, 2.0f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 06: Large NCHW all-copy
TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_06)
{
    gert::StorageShape inputGradsShape = {{4, 8, 64, 64}, {4, 8, 64, 64}};
    gert::StorageShape inputOriImageShape = {{4, 8, 64, 64}, {4, 8, 64, 64}};
    gert::StorageShape outputShape = {{4, 8, 64, 64}, {4, 8, 64, 64}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Test 07 (issue-17 回归): tiling 必须使用运行时(storage)format, 而非 origin format。
// bicubic 的 CheckFormatValid 允许 ND 且把 ND 等同 NCHW 处理, 因此需用 NHWC 场景才能区分:
//   A: origin=ND,   storage=NHWC  (模拟图优化后 ori 被改写为 ND)
//   B: origin=NHWC, storage=NHWC  (基准)
// 修复后 tiling 只读 storage format(NHWC), A 与 B 的 tilingKey/blockNum 必然一致;
// 若回退为 GetOriginFormat, A 会按 origin=ND(等价NCHW) 误解 NHWC 形状的 H/W/C 轴,
// 得到与 B 不同的 tiling 结果, 本用例(EXPECT_EQ)即失效。
TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_origin_ne_storage_nhwc)
{
    gert::StorageShape gradsShape = {{2, 4, 6, 64}, {2, 4, 6, 64}};
    gert::StorageShape oriImageShape = {{2, 2, 3, 64}, {2, 2, 3, 64}};
    gert::StorageShape yShape = {{2, 2, 3, 64}, {2, 2, 3, 64}};
    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    // A: origin=ND, storage=NHWC
    gert::TilingContextPara paraOriNd(
        "ResizeBicubicV2Grad",
        {gert::TilingContextPara::TensorDescription(gradsShape, ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr,
                                                    ge::FORMAT_NHWC),
         gert::TilingContextPara::TensorDescription(oriImageShape, ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr,
                                                    ge::FORMAT_NHWC)},
        {gert::TilingContextPara::TensorDescription(yShape, ge::DT_FLOAT, ge::FORMAT_ND, false, nullptr,
                                                    ge::FORMAT_NHWC)},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({2.0f, 2.0f}))},
        &compileInfo);

    // B: origin=NHWC, storage=NHWC (基准)
    gert::TilingContextPara paraBaseline(
        "ResizeBicubicV2Grad",
        {{gradsShape, ge::DT_FLOAT, ge::FORMAT_NHWC}, {oriImageShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({2.0f, 2.0f}))},
        &compileInfo);

    TilingInfo tilingInfoA;
    TilingInfo tilingInfoB;
    ASSERT_TRUE(ExecuteTiling(paraOriNd, tilingInfoA));
    ASSERT_TRUE(ExecuteTiling(paraBaseline, tilingInfoB));
    EXPECT_EQ(tilingInfoA.tilingKey, tilingInfoB.tilingKey);
    EXPECT_EQ(tilingInfoA.blockNum, tilingInfoB.blockNum);
}

// ---------------------------------------------------------------------------
// issue-1 回归: 极端上采样反向下 (输出元素极少 + H gather 域 ~2^31) 必须走 split-K
// 确定性并行路径 (tilingKey 20002/20003), 而非原 SimtDetermine(20000/20001) 单线程
// 串扫巨大 H 域导致 vector core 看门狗超时。
// 触发条件: yShapeSize < coreNum 且 lenDstH(=grads H, gather 域) >= 4096。
// ---------------------------------------------------------------------------

// Test 08 (issue-1 复现 shape): grads (1,1,2147483649,1) -> y (1,1,2,1)。
// lenDstH = 2147483649 > INT32_MAX => 走 idx64 => 期望 tilingKey = 20003 (SPLITK_IDX64)。
// coreNum=64, yShapeSize=2 => coresPerOutput = 64/2 = 32, useCoreNum(blockNum) = 2*32 = 64。
// workspace 必须在系统预留区之外额外容纳 yShapeSize * segsPerOutput 个 float partial。
TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_splitk_issue1_idx64)
{
    gert::StorageShape inputGradsShape = {{1, 1, 2147483649LL, 1}, {1, 1, 2147483649LL, 1}};
    gert::StorageShape inputOriImageShape = {{1, 1, 2, 1}, {1, 1, 2, 1}};
    gert::StorageShape outputShape = {{1, 1, 2, 1}, {1, 1, 2, 1}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 20003);
    EXPECT_EQ(tilingInfo.blockNum, 64U);
    // idx64 => threadNum=256, segsPerOutput = coresPerOutput(32) * 256 = 8192。
    // 额外 partial 区 = yShapeSize(2) * 8192 * sizeof(float) = 65536 字节。
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    EXPECT_GE(tilingInfo.workspaceSizes[0], static_cast<int64_t>(2) * 8192 * 4);
}

// Test 09: 中等规模但同样欠并行的 split-K 场景, 索引可用 int32。
// grads (1,1,8192,1) -> y (1,1,2,1): lenDstH=8192(>=4096) 且各维 <= INT32_MAX
// => IsUseIdx32 = true => 期望 tilingKey = 20002 (SPLITK idx32)。
// coresPerOutput = 64/2 = 32, threadNum=512, segsPerOutput = 32*512 = 16384, blockNum=64。
TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_splitk_idx32)
{
    gert::StorageShape inputGradsShape = {{1, 1, 8192, 1}, {1, 1, 8192, 1}};
    gert::StorageShape inputOriImageShape = {{1, 1, 2, 1}, {1, 1, 2, 1}};
    gert::StorageShape outputShape = {{1, 1, 2, 1}, {1, 1, 2, 1}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 20002);
    EXPECT_EQ(tilingInfo.blockNum, 64U);
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    EXPECT_GE(tilingInfo.workspaceSizes[0], static_cast<int64_t>(2) * 16384 * 4);
}

// Test 10 (阈值守卫, 无回归): 输出同样欠并行(yShapeSize=2<coreNum), 但 H gather 域
// lenDstH=2000 < 4096 未达 split-K 阈值 => 必须保持原 SimtDetermine 路径。
// 各维 <= INT32_MAX => tilingKey = 20000, blockNum = min(yShapeSize, coreNum) = 2。
TEST_F(ResizeBicubicV2GradTilingTest, resize_bicubic_v2_grad_tiling_splitk_threshold_guard)
{
    gert::StorageShape inputGradsShape = {{1, 1, 2000, 1}, {1, 1, 2000, 1}};
    gert::StorageShape inputOriImageShape = {{1, 1, 2, 1}, {1, 1, 2, 1}};
    gert::StorageShape outputShape = {{1, 1, 2, 1}, {1, 1, 2, 1}};

    ResizeBicubicV2GradCompileInfo compileInfo = {64, 200704, 32, 0};

    gert::TilingContextPara tilingContextPara(
        "ResizeBicubicV2Grad",
        {{inputGradsShape, ge::DT_FLOAT, ge::FORMAT_NCHW}, {inputOriImageShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {{outputShape, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("scales", Ops::Cv::AnyValue::CreateFrom<vector<float>>({0.0f, 0.0f}))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 20000);
    EXPECT_EQ(tilingInfo.blockNum, 2U);
}
