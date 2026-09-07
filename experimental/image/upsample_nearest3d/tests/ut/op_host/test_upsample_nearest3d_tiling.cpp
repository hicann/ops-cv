/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../op_host/upsample_nearest3d_tiling.h"
#include "../../../op_kernel/upsample_nearest3d_struct.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;
using namespace optiling;
using namespace UpsampleNearest3d;

class UpsampleNearest3dTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UpsampleNearest3dTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UpsampleNearest3dTiling TearDown" << std::endl; }
};

namespace {
const uint64_t UINT8_COMMON_KEY = GET_TPL_TILING_KEY(UPSAMPLE_NEAREST3D_TPL_UINT8, UPSAMPLE_NEAREST3D_TPL_UINT8);

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

void ExecuteUint8TilingCase(const char* opType, const std::vector<int64_t>& inputShape,
                            const std::vector<int64_t>& outputShape, const std::vector<int64_t>& outputSize,
                            UpsampleNearest3dCompileInfo& compileInfo, const std::vector<uint64_t>& expectedKeys,
                            size_t& blockDim)
{
    const gert::TilingContextPara::TensorDescription inputDesc(MakeStorageShape(inputShape), ge::DT_UINT8,
                                                               ge::FORMAT_ND);
    const gert::TilingContextPara::TensorDescription outputDesc(MakeStorageShape(outputShape), ge::DT_UINT8,
                                                                ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara(
        opType, {inputDesc}, {outputDesc},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom(outputSize)),
         gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0f)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0f))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_TRUE(std::find(expectedKeys.begin(), expectedKeys.end(), tilingInfo.tilingKey) != expectedKeys.end())
        << "unexpected UINT8 tiling key: " << tilingInfo.tilingKey;
    blockDim = tilingInfo.blockNum;
}
} // namespace

TEST_F(UpsampleNearest3dTiling, Ascend910B_uint8_common_when_group_parallelism_is_insufficient)
{
    UpsampleNearest3dCompileInfo compileInfo = {48, 192 * 1024};
    size_t blockDim = 0;
    ExecuteUint8TilingCase("UpsampleNearest3d", {16, 1, 1, 1, 7}, {16, 1, 4, 4, 7}, {4, 4, 7}, compileInfo,
                           {UINT8_COMMON_KEY}, blockDim);
    EXPECT_GT(blockDim, 0);
}

TEST_F(UpsampleNearest3dTiling, Ascend910B_uint8_common_non_aligned_source_with_tail_batch)
{
    UpsampleNearest3dCompileInfo compileInfo = {48, 192 * 1024};
    size_t blockDim = 0;
    ExecuteUint8TilingCase("UpsampleNearest3d", {129, 1, 1, 1, 5}, {129, 1, 4, 4, 7}, {4, 4, 7}, compileInfo,
                           {UINT8_COMMON_KEY}, blockDim);
    EXPECT_GT(blockDim, 0);
}

TEST_F(UpsampleNearest3dTiling, Ascend910B_uint8_common_broadcast_volume_and_tail_batch)
{
    UpsampleNearest3dCompileInfo compileInfo = {48, 192 * 1024};
    size_t blockDim = 0;
    ExecuteUint8TilingCase("UpsampleNearest3d", {129, 1, 1, 1, 24}, {129, 1, 23, 4, 24}, {23, 4, 24}, compileInfo,
                           {UINT8_COMMON_KEY}, blockDim);
    EXPECT_GT(blockDim, 0);
}

TEST_F(UpsampleNearest3dTiling, Ascend910B_uint8_common_multi_unique_mapping)
{
    UpsampleNearest3dCompileInfo compileInfo = {48, 192 * 1024};
    size_t blockDim = 0;
    ExecuteUint8TilingCase("UpsampleNearestExact3d", {128, 1, 1, 2, 24}, {128, 1, 2, 2, 24}, {2, 2, 24}, compileInfo,
                           {UINT8_COMMON_KEY}, blockDim);
    EXPECT_GT(blockDim, 0);
}

TEST_F(UpsampleNearest3dTiling, Ascend910B_uint8_common_with_small_ub)
{
    UpsampleNearest3dCompileInfo compileInfo = {48, 16 * 1024};
    size_t blockDim = 0;
    ExecuteUint8TilingCase("UpsampleNearest3d", {128, 1, 1, 1, 24}, {128, 1, 23, 4, 24}, {23, 4, 24}, compileInfo,
                           {UINT8_COMMON_KEY}, blockDim);
}

TEST_F(UpsampleNearest3dTiling, Ascend910B_uint8_common_with_large_output_row_count)
{
    UpsampleNearest3dCompileInfo compileInfo = {48, 192 * 1024};
    size_t blockDim = 0;
    ExecuteUint8TilingCase("UpsampleNearest3d", {1, 1, 1, 1, 32}, {1, 1, 255, 257, 32}, {255, 257, 32}, compileInfo,
                           {UINT8_COMMON_KEY}, blockDim);
    EXPECT_EQ(255 * 257, 65535);
}

TEST_F(UpsampleNearest3dTiling, Ascend910B_upsample_nearest3d_tiling_001)
{
    UpsampleNearest3dCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3d", {{{{1, 1, 128, 128, 128}, {1, 1, 128, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 256, 256, 256}, {1, 1, 256, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({256, 256, 256})),
         gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    uint64_t expectTilingKey = 5140;
    string expectTilingData = "0 1 128 128 128 256 256 256 4539628425446424576 1056964608 2048 1028 4 4 128 128 341 16 "
                              "16368 1 1 1 48 0 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearest3dTiling, Ascend310P_upsample_nearest3d_tiling_002)
{
    UpsampleNearest3dCompileInfo compileInfo = {48};
    string socVersion = "Ascend310p";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3d", {{{{1, 1, 128, 128, 128}, {1, 1, 128, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 256, 256, 256}, {1, 1, 256, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({256, 256, 256})),
         gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 5140;
    string expectTilingData = "0 1 128 128 128 256 256 256 4539628425446424576 1056964608 2048 1028 4 4 128 128 341 16 "
                              "16368 1 1 1 48 0 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearest3dTiling, upsample_nearest3d_tiling_001)
{
    UpsampleNearest3dCompileInfo compileInfo = {48};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3d", {{{{1, 1, 128, 128, 128}, {1, 1, 128, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 256, 256, 256}, {1, 1, 256, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({256, 256, 256})),
         gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 257;
    string expectTilingData = "262144 1 1 128 128 128 256 256 256 0 4539628425446424576 1056964608 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearest3dTiling, upsample_nearest3d_tiling_002)
{
    UpsampleNearest3dCompileInfo compileInfo = {64};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3d", {{{{2, 2, 4, 5, 6}, {2, 2, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 2, 4, 5, 6}, {2, 2, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({4, 5, 6})),
         gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "32 2 2 4 5 6 4 5 6 32760 4575657222473777152 1065353216 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearest3dTiling, upsample_nearest3d_tiling_004)
{
    UpsampleNearest3dCompileInfo compileInfo = {64};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3d", {{{{2, 2, 4, 5, 6}, {2, 2, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 2, 8, 8, 7}, {2, 2, 8, 8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({8, 8, 7})),
         gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 259;
    string expectTilingData = "28 2 2 4 5 6 8 8 7 0 4548635624701165568 1062956471 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearest3dTiling, upsample_nearest3d_tiling_005)
{
    UpsampleNearest3dCompileInfo compileInfo = {64};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3d", {{{{2, 22, 20, 30, 2}, {2, 22, 20, 30, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 22, 40, 20, 8}, {2, 22, 40, 20, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({40, 20, 8})),
         gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 258;
    string expectTilingData = "2200 2 22 20 30 2 40 20 8 0 4593671620974870528 1048576000 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleNearest3dTiling, upsample_nearest3d_tiling_006)
{
    UpsampleNearest3dCompileInfo compileInfo = {64};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "UpsampleNearest3d", {{{{2, 2, 80, 30, 2}, {2, 2, 80, 30, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 2, 60, 200, 200}, {2, 2, 60, 200, 200}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("output_size",
                                         Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>({60, 200, 200})),
         gert::TilingContextPara::OpAttr("scale_d", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
         gert::TilingContextPara::OpAttr("scale_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo, socVersion);
    uint64_t expectTilingKey = 257;
    string expectTilingData = "37500 4 2 80 30 2 60 200 200 0 4474776592541461163 1008981770 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
