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
 * \file test_upsample_bilinear2d_aa_backward_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_bilinear2d_aa_backward_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
using namespace std;
using namespace gert;
using namespace optiling;

class UpsampleBilinear2dAABackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleBilinear2dAABackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleBilinear2dAABackwardTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleBilinear2dAABackwardTiling, upsample_bilinear2d_aa_backward_tiling_001)
{
    UpsampleBilinear2dAABackwardCompileInfo compileInfo = {48};
    std::vector<int64_t> output_size = {256, 256};
    std::vector<int64_t> input_size = {1, 1, 256, 256};
    gert::TilingContextPara tilingContextPara("UpsampleBilinear2dAABackward",
        {
            {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 256, 256}, {1, 1, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)},
            {"input_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(input_size)},
            {"align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)},
            {"scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)},
            {"scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)}},
        &compileInfo,
        48,
        192 * 1024,
        8192);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "16 4611686018427387906 4539628425463201792 4611686019484352512 1407397432131584 32768 1305670058288 "
        "206158430256 257 1 1 128 128 1 1 256 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 16 16 32 32 32 48 48 48 64 64 64 80 80 80 96 96 96 112 112 112 128 128 128 144 "
        "144 144 160 160 160 176 176 176 192 192 192 208 208 208 224 224 224 240 240 240 0 0 16 16 16 32 32 32 48 48 "
        "48 64 64 64 80 80 80 96 96 96 112 112 112 128 128 128 144 144 144 160 160 160 176 176 176 192 192 192 208 208 "
        "208 224 224 224 240 240 240 256 256 256 0 0 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 "
        "43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 0 43 86 128 43 86 128 43 86 128 43 86 128 43 "
        "86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 "
        "86 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "16 16 16 32 32 32 48 48 48 64 64 64 80 80 80 96 96 96 112 112 112 128 128 128 144 144 144 160 160 160 176 176 "
        "176 192 192 192 208 208 208 224 224 224 240 240 240 0 0 16 16 16 32 32 32 48 48 48 64 64 64 80 80 80 96 96 96 "
        "112 112 112 128 128 128 144 144 144 160 160 160 176 176 176 192 192 192 208 208 208 224 224 224 240 240 240 "
        "256 256 256 0 0 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 "
        "86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 0 86 172 256 86 172 256 86 172 256 86 172 256 86 172 "
        "256 86 172 256 86 172 256 86 172 256 86 172 256 86 172 256 86 172 256 86 172 256 86 172 256 86 172 256 86 172 "
        "256 86 172 256 0 0 549755813889 549755814144 549755814016 81604378640 68719476864 4294967320 4294967297 1 0 "
        "59373627899904 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 "
        "549755814144 68719476864 81604378880 1099511627792 4294967320 4294967297 1 0 112150186033152 16384 4294967297 "
        "4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {17025024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}