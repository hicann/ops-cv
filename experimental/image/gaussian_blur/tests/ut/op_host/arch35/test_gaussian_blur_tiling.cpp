/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_kernel/arch35/gaussian_blur_tiling_data.h"

namespace {

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct GaussianBlurCompileInfo {};
GaussianBlurCompileInfo g_compileInfo;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara MakeContext(const std::vector<int64_t>& shape, const std::vector<int64_t>& ksize)
{
    const gert::StorageShape storageShape = MakeStorageShape(shape);
    std::vector<TensorDesc> inputs = {TensorDesc(storageShape, ge::DT_FLOAT, ge::FORMAT_ND)};
    std::vector<TensorDesc> outputs = {TensorDesc(storageShape, ge::DT_FLOAT, ge::FORMAT_ND)};
    std::vector<OpAttr> attrs = {
        OpAttr("ksize", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)),
        OpAttr("sigma_x", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
        OpAttr("sigma_y", Ops::Cv::AnyValue::CreateFrom<float>(1.0f)),
        OpAttr("border_type", Ops::Cv::AnyValue::CreateFrom<int64_t>(1)),
    };
    return gert::TilingContextPara("GaussianBlur", inputs, outputs, attrs, &g_compileInfo, "Ascend950", 64, 262144,
                                   4096);
}

bool RunTiling(const gert::TilingContextPara& context, TilingInfo& info, GaussianBlurTilingData& tilingData)
{
    if (!ExecuteTiling(context, info) || info.tilingDataSize < sizeof(GaussianBlurTilingData)) {
        return false;
    }
    std::memcpy(&tilingData, info.tilingData.get(), sizeof(tilingData));
    return true;
}

} // namespace

class GaussianBlurTilingTest : public testing::Test {};

TEST_F(GaussianBlurTilingTest, fused_generic_channel_path)
{
    TilingInfo info;
    GaussianBlurTilingData tilingData{};
    ASSERT_TRUE(RunTiling(MakeContext({839, 383, 255}, {31, 31}), info, tilingData));
    EXPECT_EQ(tilingData.h, 839U);
    EXPECT_EQ(tilingData.w, 383U);
    EXPECT_EQ(tilingData.c, 255U);
    EXPECT_EQ(tilingData.kernelSize, 31U);
    EXPECT_EQ(tilingData.kernelSizeY, 31U);
    EXPECT_EQ(tilingData.pathMode, GAUSSIAN_BLUR_PATH_GENERIC_C);
    EXPECT_GT(info.blockNum, 0U);
    EXPECT_GT(tilingData.totalTiles, 0U);
}

TEST_F(GaussianBlurTilingTest, selects_specialized_channel_paths)
{
    TilingInfo info;
    GaussianBlurTilingData tilingData{};
    ASSERT_TRUE(RunTiling(MakeContext({64, 64}, {7, 7}), info, tilingData));
    EXPECT_EQ(tilingData.pathMode, GAUSSIAN_BLUR_PATH_C1_FAST);

    ASSERT_TRUE(RunTiling(MakeContext({64, 64, 3}, {7, 7}), info, tilingData));
    EXPECT_EQ(tilingData.pathMode, GAUSSIAN_BLUR_PATH_C3_FAST);

    ASSERT_TRUE(RunTiling(MakeContext({64, 64, 4}, {7, 7}), info, tilingData));
    EXPECT_EQ(tilingData.pathMode, GAUSSIAN_BLUR_PATH_C4_FAST);
}

TEST_F(GaussianBlurTilingTest, rejects_invalid_kernel_size)
{
    TilingInfo info;
    GaussianBlurTilingData tilingData{};
    EXPECT_FALSE(RunTiling(MakeContext({32, 32, 3}, {4, 3}), info, tilingData));
}
