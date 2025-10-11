/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_bicubic2d_grad_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

class UpsampleBicubic2dGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleBicubic2dGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleBicubic2dGradTiling TearDown" << std::endl;
    }
};

TEST_F(UpsampleBicubic2dGradTiling, test_tiling_float16_1)
{
    struct UpsampleBicubic2dGradCompileInfo {
        uint32_t aicNum = 20;
        uint32_t aivNum = 40;
        uint64_t ubSize = 196608;
        uint64_t l1Size = 524288;
        uint64_t l2Size = 201326592;
        uint64_t l0CSize = 131072;
        uint64_t l0ASize = 65536;
        uint64_t l0BSize = 65536;
        platform_ascendc::SocVersion socVersion;
        std::string socVersionStr = "";
    } compile_info;

    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dGrad",
        {{{{1, 2, 64, 64}, {1, 2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1, 2, 32, 32}, {1, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
            gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.5))},
        &compile_info);
    uint64_t expectTilingKey = 10000001;
    string expectTilingData =
        "171798691841 0 4539628425446424576 68719476752 2 274877906944 137438953536 68719476768 4294967300 42949672960 "
        "8589934592 17179869200 1 10 281474976710658 412316860416 16 34359738416 0 0 0 0 0 0 0 0 0 68719476737 "
        "68719476768 68719476800 68719476768 137438953488 4294967312 4294967297 1 0 6597069766656 2048 4294967297 "
        "4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 274877906945 274877906960 274877906960 68719476752 "
        "68719476800 4294967312 4294967297 1 0 10995116277760 4096 4294967297 4294967297 4294967297 0 8589934594 1 0 0 "
        "0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16808448};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dGradTiling, test_tiling_float_1)
{
    struct UpsampleBicubic2dGradCompileInfo {
        uint32_t aicNum = 20;
        uint32_t aivNum = 40;
        uint64_t ubSize = 196608;
        uint64_t l1Size = 524288;
        uint64_t l2Size = 201326592;
        uint64_t l0CSize = 131072;
        uint64_t l0ASize = 65536;
        uint64_t l0BSize = 65536;
        platform_ascendc::SocVersion socVersion;
        std::string socVersionStr = "";
    } compile_info;

    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dGrad",
        {{{{1, 2, 64, 64}, {1, 2, 64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 2, 32, 32}, {1, 2, 32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.5)),
            gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.5))},
        &compile_info);
    uint64_t expectTilingKey = 10000001;
    string expectTilingData =
        "171798691840 0 4539628425446424576 68719476752 2 274877906944 137438953536 68719476768 4294967300 42949672960 "
        "8589934592 17179869200 1 10 140737488355330 412316860416 32 68719476784 0 0 0 0 0 0 0 0 0 68719476737 "
        "68719476768 68719476800 68719476768 137438953488 4294967312 4294967297 1 0 13194139533312 2048 4294967297 "
        "4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 274877906945 274877906960 274877906960 68719476752 "
        "68719476800 4294967312 4294967297 1 0 21990232555520 4096 4294967297 4294967297 4294967297 0 8589934594 1 0 0 "
        "0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16837120};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dGradTiling, test_tiling_float_2)
{
    struct UpsampleBicubic2dGradCompileInfo {
        uint32_t aicNum = 20;
        uint32_t aivNum = 40;
        uint64_t ubSize = 196608;
        uint64_t l1Size = 524288;
        uint64_t l2Size = 201326592;
        uint64_t l0CSize = 131072;
        uint64_t l0ASize = 65536;
        uint64_t l0BSize = 65536;
        platform_ascendc::SocVersion socVersion;
        std::string socVersionStr = "";
    } compile_info;

    gert::TilingContextPara tilingContextPara("UpsampleBicubic2dGrad",
        {{{{1, 2, 200, 200}, {1, 2, 200, 200}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 2, 2, 2}, {1, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.01)),
            gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.01))},
        &compile_info);
    uint64_t expectTilingKey = 10000002;
    string expectTilingData =
        "85899345920 196608 4333543705419175690 0 4294967298 858993459202 8589934792 2 2 0 0 85899345920 0 0 0 0 0 0 "
        "4294967312 1 200 200 3200 800 85899345920 0 8589934592 8589934593 858993459202 68719476936 858993459202 "
        "68719476752 4294967496 4294967297 1 0 109951162777600 1024 4294967297 4294967297 4294967297 0 8589934594 1 0 "
        "0 0 0 0 0 0 0 1717986918401 858993459202 1717986918600 858993459216 68719477136 8589934608 4294967309 1 0 "
        "277076930199552 25600 4294967297 4294967297 55834574849 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {17292416};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}