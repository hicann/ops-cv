/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <limits>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_kernel/arch35/combined_non_max_suppression_tiling_data.h"

namespace {

struct CombinedNonMaxSuppressionCompileInfoForTest {
    int32_t coreNum = 4;
    uint32_t padding = 0;
    uint64_t sysWorkspaceSize = 16777216;
};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

gert::TilingContextPara MakeContext(const std::vector<int64_t>& boxesShape, const std::vector<int64_t>& scoresShape,
                                    const std::vector<int64_t>& outputBoxesShape, int32_t* maxPerClass,
                                    int32_t* maxTotal, float* iouThreshold, float* scoreThreshold,
                                    CombinedNonMaxSuppressionCompileInfoForTest* compileInfo, bool clipBoxes = true)
{
    const std::vector<int64_t> outputVectorShape = {outputBoxesShape[0], outputBoxesShape[1]};
    const gert::StorageShape boxes = MakeStorageShape(boxesShape);
    const gert::StorageShape scores = MakeStorageShape(scoresShape);
    const gert::StorageShape scalar = MakeStorageShape({});
    const gert::StorageShape outputBoxes = MakeStorageShape(outputBoxesShape);
    const gert::StorageShape outputVector = MakeStorageShape(outputVectorShape);
    const gert::StorageShape outputValid = MakeStorageShape({outputBoxesShape[0]});
    const std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {boxes, ge::DT_FLOAT, ge::FORMAT_ND},
        {scores, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_INT32, ge::FORMAT_ND, true, maxPerClass},
        {scalar, ge::DT_INT32, ge::FORMAT_ND, true, maxTotal},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND, true, iouThreshold},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND, true, scoreThreshold},
    };
    const std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {outputBoxes, ge::DT_FLOAT, ge::FORMAT_ND},
        {outputVector, ge::DT_FLOAT, ge::FORMAT_ND},
        {outputVector, ge::DT_FLOAT, ge::FORMAT_ND},
        {outputValid, ge::DT_INT32, ge::FORMAT_ND},
    };
    return gert::TilingContextPara(
        "CombinedNonMaxSuppression", inputs, outputs,
        {gert::TilingContextPara::OpAttr("pad_per_class", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("clip_boxes", Ops::Cv::AnyValue::CreateFrom<bool>(clipBoxes))},
        compileInfo, "Ascend950", 4, 262144, 4096);
}

TEST(CombinedNonMaxSuppressionTiling, BuildsWorkspaceForSharedBoxes)
{
    int32_t maxPerClass = 2;
    int32_t maxTotal = 4;
    float iouThreshold = 0.5F;
    float scoreThreshold = 0.25F;
    CombinedNonMaxSuppressionCompileInfoForTest compileInfo;
    auto context = MakeContext({2, 5, 1, 4}, {2, 5, 3}, {2, 4, 4}, &maxPerClass, &maxTotal, &iouThreshold,
                               &scoreThreshold, &compileInfo, false);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(CombinedNonMaxSuppressionTilingData));
    const auto* tiling = reinterpret_cast<const CombinedNonMaxSuppressionTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.blockNum, 4U);
    EXPECT_EQ(tiling->batchSize, 2);
    EXPECT_EQ(tiling->numBoxes, 5);
    EXPECT_EQ(tiling->boxClasses, 1);
    EXPECT_EQ(tiling->numClasses, 3);
    EXPECT_EQ(tiling->maxOutputPerClass, 2);
    EXPECT_EQ(tiling->outputSize, 4);
    EXPECT_EQ(tiling->clipBoxes, 0);
    EXPECT_FLOAT_EQ(tiling->iouThreshold, 0.5F);
    EXPECT_FLOAT_EQ(tiling->scoreThreshold, 0.25F);
    EXPECT_EQ(tiling->selectedScoresOffset, 0U);
    EXPECT_EQ(tiling->selectedIndicesOffset, 64U);
    EXPECT_EQ(tiling->selectedCountsOffset, 128U);
    EXPECT_EQ(tiling->suppressedOffset, 160U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], static_cast<int64_t>(compileInfo.sysWorkspaceSize + 192U));
}

TEST(CombinedNonMaxSuppressionTiling, RejectsIouThresholdOutsideUnitInterval)
{
    int32_t maxPerClass = 2;
    int32_t maxTotal = 4;
    float iouThreshold = 1.1F;
    float scoreThreshold = 0.0F;
    CombinedNonMaxSuppressionCompileInfoForTest compileInfo;
    auto context = MakeContext({1, 5, 1, 4}, {1, 5, 2}, {1, 4, 4}, &maxPerClass, &maxTotal, &iouThreshold,
                               &scoreThreshold, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(CombinedNonMaxSuppressionTiling, RejectsMismatchedClassSpecificBoxes)
{
    int32_t maxPerClass = 2;
    int32_t maxTotal = 4;
    float iouThreshold = 0.5F;
    float scoreThreshold = 0.0F;
    CombinedNonMaxSuppressionCompileInfoForTest compileInfo;
    auto context = MakeContext({1, 5, 3, 4}, {1, 5, 2}, {1, 4, 4}, &maxPerClass, &maxTotal, &iouThreshold,
                               &scoreThreshold, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(CombinedNonMaxSuppressionTiling, RejectsOutputShapeLargerThanMaxTotal)
{
    int32_t maxPerClass = 2;
    int32_t maxTotal = 4;
    float iouThreshold = 0.5F;
    float scoreThreshold = 0.0F;
    CombinedNonMaxSuppressionCompileInfoForTest compileInfo;
    auto context = MakeContext({1, 5, 1, 4}, {1, 5, 2}, {1, 5, 4}, &maxPerClass, &maxTotal, &iouThreshold,
                               &scoreThreshold, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(CombinedNonMaxSuppressionTiling, RejectsTaskCountOutsideInt32Range)
{
    int32_t maxPerClass = 1;
    int32_t maxTotal = 1;
    float iouThreshold = 0.5F;
    float scoreThreshold = 0.0F;
    CombinedNonMaxSuppressionCompileInfoForTest compileInfo;
    constexpr int64_t numClasses = 200;
    const int64_t batch = std::numeric_limits<int32_t>::max() / numClasses + 1;
    auto context = MakeContext({batch, 1, 1, 4}, {batch, 1, numClasses}, {batch, 1, 4}, &maxPerClass, &maxTotal,
                               &iouThreshold, &scoreThreshold, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

} // namespace
