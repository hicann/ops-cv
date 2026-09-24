/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ---------------------------------------------------------------------------------------------------------
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

using OpAttr = gert::TilingContextPara::OpAttr;
using TensorDesc = gert::TilingContextPara::TensorDescription;

struct NMSWithMaskCompileInfo {};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara MakeContext(const std::vector<int64_t>& scoreShape)
{
    static NMSWithMaskCompileInfo compileInfo;
    const auto boxShape = MakeStorageShape({2, 4});
    const auto scores = MakeStorageShape(scoreShape);
    const auto outputShape = MakeStorageShape({2});
    return gert::TilingContextPara(
        "NMSWithMask",
        {TensorDesc(boxShape, ge::DT_FLOAT, ge::FORMAT_ND), TensorDesc(scores, ge::DT_FLOAT, ge::FORMAT_ND)},
        {TensorDesc(outputShape, ge::DT_UINT8, ge::FORMAT_ND)},
        {OpAttr("iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.1f)),
         OpAttr("scores_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.1f))},
        &compileInfo, "Ascend910b", 64, 262144, 4096);
}

} // namespace

TEST(NMSWithMaskScoreLengthTilingValidation, acceptsMatchingShape)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext({2}), info));
    EXPECT_EQ(info.tilingKey, 0);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], 16U * 1024U * 1024U);
}

TEST(NMSWithMaskScoreLengthTilingValidation, rejectsShortScoreVector)
{
    ExecuteTestCase(MakeContext({1}), ge::GRAPH_FAILED);
}

TEST(NMSWithMaskScoreLengthTilingValidation, rejectsLongScoreVector)
{
    ExecuteTestCase(MakeContext({3}), ge::GRAPH_FAILED);
}
