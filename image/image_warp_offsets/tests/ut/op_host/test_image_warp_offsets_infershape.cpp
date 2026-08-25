/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_image_warp_offsets_infershape.cpp
 * \brief
 */

// ---------------IMGWarpOffsets Op start-------------------

#include <gtest/gtest.h>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
void ExpectShape(const gert::StorageShape& imagesShape, const gert::StorageShape& offsetsShape,
                 ge::graphStatus expectedStatus, const std::vector<std::vector<int64_t>>& expectedShapes = {})
{
    gert::InfershapeContextPara context(
        "IMGWarpOffsets", {{imagesShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {offsetsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    ExecuteTestCase(context, expectedStatus, expectedShapes);
}
} // namespace

TEST(IMGWarpOffsetsInfershape, infers_static_shape)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{2, 4, 6, 7}, {2, 4, 6, 7}}, ge::GRAPH_SUCCESS, {{2, 4, 6, 7, 3}});
}

TEST(IMGWarpOffsetsInfershape, preserves_dynamic_dimensions)
{
    ExpectShape({{-1, 8, 9, -1}, {-1, 8, 9, -1}}, {{-1, 4, -1, 7}, {-1, 4, -1, 7}}, ge::GRAPH_SUCCESS,
                {{-1, 4, -1, 7, -1}});
}

TEST(IMGWarpOffsetsInfershape, images_unknown_rank_skips_other_validation)
{
    ExpectShape({{-2}, {-2}}, {{2, 4, 6}, {2, 4, 6}}, ge::GRAPH_SUCCESS, {{-2}});
}

TEST(IMGWarpOffsetsInfershape, offsets_unknown_rank_skips_other_validation)
{
    ExpectShape({{2, 8, 9}, {2, 8, 9}}, {{-2}, {-2}}, ge::GRAPH_SUCCESS, {{-2}});
}

TEST(IMGWarpOffsetsInfershape, rejects_images_rank)
{
    ExpectShape({{2, 8, 3}, {2, 8, 3}}, {{2, 4, 6, 7}, {2, 4, 6, 7}}, ge::GRAPH_FAILED);
}

TEST(IMGWarpOffsetsInfershape, rejects_images_channel)
{
    ExpectShape({{2, 8, 9, 2}, {2, 8, 9, 2}}, {{2, 4, 6, 7}, {2, 4, 6, 7}}, ge::GRAPH_FAILED);
}

TEST(IMGWarpOffsetsInfershape, rejects_offsets_rank)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{2, 4, 6}, {2, 4, 6}}, ge::GRAPH_FAILED);
}

TEST(IMGWarpOffsetsInfershape, rejects_offsets_point_dimension)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{2, 5, 6, 7}, {2, 5, 6, 7}}, ge::GRAPH_FAILED);
}

TEST(IMGWarpOffsetsInfershape, rejects_known_batch_mismatch)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{3, 4, 6, 7}, {3, 4, 6, 7}}, ge::GRAPH_FAILED);
}

TEST(IMGWarpOffsetsInfershape, rejects_unknown_and_known_batch_mismatch)
{
    ExpectShape({{-1, 8, 9, 3}, {-1, 8, 9, 3}}, {{2, 4, 6, 7}, {2, 4, 6, 7}}, ge::GRAPH_FAILED);
}
