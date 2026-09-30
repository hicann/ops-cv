/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

using TensorDesc = gert::TilingContextPara::TensorDescription;

struct CIoUCompileInfo {};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara MakeContext(const std::vector<int64_t>& bboxes, const std::vector<int64_t>& gtboxes,
                                    ge::DataType bboxesDtype = ge::DT_FLOAT, ge::DataType gtboxesDtype = ge::DT_FLOAT)
{
    const gert::StorageShape bboxesShape = MakeStorageShape(bboxes);
    const gert::StorageShape gtboxesShape = MakeStorageShape(gtboxes);
    const int64_t n = bboxes.size() >= 2 && bboxes[1] >= 0 ? bboxes[1] : ge::UNKNOWN_DIM;
    const gert::StorageShape outputShape = MakeStorageShape({1, n});
    const std::vector<TensorDesc> inputs = {{bboxesShape, bboxesDtype, ge::FORMAT_ND},
                                            {gtboxesShape, gtboxesDtype, ge::FORMAT_ND}};
    const std::vector<TensorDesc> outputs = {{outputShape, bboxesDtype, ge::FORMAT_ND},
                                             {outputShape, bboxesDtype, ge::FORMAT_ND}};
    static CIoUCompileInfo compileInfo;
    return gert::TilingContextPara("CIoU", inputs, outputs, &compileInfo);
}

} // namespace

TEST(CIoUTilingShapeContract, accepts_matching_four_by_n)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext({4, 8}, {4, 8}), info));
}

TEST(CIoUTilingShapeContract, accepts_empty_box_list)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext({4, 0}, {4, 0}), info));
}

TEST(CIoUTilingShapeContract, rejects_invalid_input_shape)
{
    ExecuteTestCase(MakeContext({2, 8}, {4, 8}), ge::GRAPH_FAILED);
    ExecuteTestCase(MakeContext({4, 8}, {3, 8}), ge::GRAPH_FAILED);
    ExecuteTestCase(MakeContext({4}, {4, 8}), ge::GRAPH_FAILED);
    ExecuteTestCase(MakeContext({4, 8, 1}, {4, 8, 1}), ge::GRAPH_FAILED);
    ExecuteTestCase(MakeContext({4, 8}, {4, 4}), ge::GRAPH_FAILED);
    ExecuteTestCase(MakeContext({4, -1}, {4, -1}), ge::GRAPH_FAILED);
    ExecuteTestCase(MakeContext({4, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1},
                                {4, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1}),
                    ge::GRAPH_FAILED);
}

TEST(CIoUTilingDtypeValidation, accepts_matching_input_dtypes)
{
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(MakeContext({4, 8}, {4, 8}, dtype, dtype), info));
    }
}

TEST(CIoUTilingDtypeValidation, rejects_mixed_input_dtypes)
{
    for (const auto dtypes : {std::pair{ge::DT_FLOAT, ge::DT_FLOAT16}, std::pair{ge::DT_FLOAT16, ge::DT_FLOAT}}) {
        ExecuteTestCase(MakeContext({4, 8}, {4, 8}, dtypes.first, dtypes.second), ge::GRAPH_FAILED);
    }
}
