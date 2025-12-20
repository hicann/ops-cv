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
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_bicubic2d_aa_grad_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
 
using namespace std;

class UpsampleBicubic2dAAGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleBicubic2dAAGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleBicubic2dAAGradTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleBicubic2dAAGradTiling, upsample_bicubic2d_aa_grad_tiling_001)
{
    struct UpsampleBicubic2dAAGradCompileInfo {
        uint32_t coreNum = 24;
    } compile_info;
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dAAGrad",
                                                {{{{1, 1, 256, 256}, {1, 1, 256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({256, 256})),
                                                gert::TilingContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({1, 1, 128, 128})),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compile_info, socVersion, 48, 192*1024, 8192);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "16 2 4539628425446424576 4575657222473777152 22548578304 22548578304 32768 5085241279648 1 1 256 256 1 1 128 128 103079215128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 16 16 32 32 32 48 48 48 64 64 64 80 80 80 96 96 96 112 112 112 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 16 16 32 32 32 48 48 48 64 64 64 80 80 80 96 96 96 112 112 112 128 128 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 86 172 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 86 172 256 86 172 256 86 172 256 86 172 256 86 172 256 86 172 256 86 172 256 86 172 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 16 16 32 32 32 48 48 48 64 64 64 80 80 80 96 96 96 112 112 112 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16 16 16 32 32 32 48 48 48 64 64 64 80 80 80 96 96 96 112 112 112 128 128 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 43 86 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 43 86 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1099511627777 1099511627904 1099511628032 317827579920 68719476864 17179869248 8589934594 1 0 598134325510144 8192 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 549755813889 1099511627904 68719476992 317827580032 549755813904 8589934656 4294967298 1 0 316659348799488 8192 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33912832};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dAAGradTiling, upsample_bicubic2d_aa_grad_tiling_002)
{
    struct UpsampleBicubic2dAAGradCompileInfo {
        uint32_t coreNum = 0;
    } compile_info;
    string socVersion = "Ascend910b";
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dAAGrad",
                                                {{{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({128, 128})),
                                                gert::TilingContextPara::OpAttr("input_size", Ops::Cv::AnyValue::CreateFrom<vector<int64_t>>({1, 1, 128, 128})),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compile_info, socVersion, 48, 192*1024, 8192);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "16 2 4575657222473777152 4575657222473777152 22548578304 22548578304 16384 2886218023584 1 1 128 128 1 1 128 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 549755814016 68719476864 180388626560 549755813904 4294967344 4294967297 1 0 118747255799808 8192 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33625344};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
