/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../op_kernel/arch35/image_projective_transform_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {

struct ImageProjectiveTransformContractCompileInfo {};

gert::TilingContextPara MakeTilingCase(int64_t imagesBatch, int64_t transformsBatch)
{
    static ImageProjectiveTransformContractCompileInfo compileInfo = {};
    static std::vector<int32_t> outputShapeData = {2, 3};
    return gert::TilingContextPara(
        "ImageProjectiveTransform",
        {{{{imagesBatch, 4, 4, 3}, {imagesBatch, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{transformsBatch, 8}, {transformsBatch, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, outputShapeData.data()}},
        {{{{imagesBatch, 2, 3, 3}, {imagesBatch, 2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("interpolation", Ops::Cv::AnyValue::CreateFrom<std::string>("BILINEAR")),
         gert::TilingContextPara::OpAttr("fill_mode", Ops::Cv::AnyValue::CreateFrom<std::string>("CONSTANT"))},
        &compileInfo, "Ascend950");
}

} // namespace

TEST(ImageProjectiveTransformContractAudit, ValidBatchTilingIsAccepted)
{
    auto context = MakeTilingCase(1, 1);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_NE(info.tilingData, nullptr);
    const auto* data = reinterpret_cast<const ImageProjectiveTransformTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->batchSize, 1);
    EXPECT_EQ(data->totalPixels, 6);
}

TEST(ImageProjectiveTransformContractAudit, NegativeStaticBatchIsRejectedByTiling)
{
    for (const auto transformsBatch : {-2, -3, -4}) {
        auto context = MakeTilingCase(1, transformsBatch);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST(ImageProjectiveTransformContractAudit, NegativeStaticImagesBatchIsRejectedByTiling)
{
    for (const auto imagesBatch : {-2, -3, -4}) {
        auto context = MakeTilingCase(imagesBatch, 1);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}
