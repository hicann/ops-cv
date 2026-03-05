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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/upsample_bicubic2d_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;


class UpsampleBicubic2dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UpsampleBicubic2dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UpsampleBicubic2dTiling TearDown" << std::endl;
    }
};

struct UpsampleBicubic2dCompileInfo {
    uint16_t totalCoreNum = 0;
    uint16_t socVersionType = 220;
};

TEST_F(UpsampleBicubic2dTiling, upsample_bicubic2d_tiling_001)
{
    gert::StorageShape input_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};
    gert::StorageShape out_shape = {{1, 1, 256, 256}, {1, 1, 256, 256}};
    UpsampleBicubic2dCompileInfo compileInfo = {48, 220};
    std::vector<int64_t> output_size = {256, 256};
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2d",
                                                {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;

    string expectTilingData =
        "64 2 4539628425446424576 17180131328 131072 9895604652288 17179869192 1 1 128 128 1 1 256 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 549755814016 824633721024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 549755814016 824633721024 1099511628032 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906944 274877906944 274877906944 274877906944 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 549755813952 549755813952 549755813952 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906944 824633720960 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 1099511627968 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 549755814144 549755814016 154618822720 274877907072 4294967336 4294967297 1 0 131941395333120 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 549755814144 274877907072 154618822912 1099511627840 8589934624 4294967298 1 0 351843720888320 65536 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33759232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dTiling, upsample_bicubic2d_tiling_002)
{
    gert::StorageShape input_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};
    gert::StorageShape out_shape = {{1, 1, 256, 256}, {1, 1, 256, 256}};
    UpsampleBicubic2dCompileInfo compileInfo = {48, 220};
    std::vector<int64_t> output_size = {256, 256};
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2d",
                                                {{input_shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {{out_shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;

    string expectTilingData =
        "64 1 4539628425446424576 17180131328 65536 9895604652288 17179869192 1 1 128 128 1 1 256 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 549755814016 824633721024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 549755814016 824633721024 1099511628032 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906944 274877906944 274877906944 274877906944 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 549755813952 549755813952 549755813952 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906944 824633720960 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 1099511627968 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 549755814144 549755814016 154618822720 274877907072 4294967344 4294967297 1 0 79164837199872 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 549755814144 274877907072 154618822912 1099511627840 4294967344 4294967297 1 0 131941395333120 65536 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33656832};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
 
TEST_F(UpsampleBicubic2dTiling, upsample_bicubic2d_tiling_003)
{
    gert::StorageShape input_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};
    gert::StorageShape out_shape = {{1, 1, 256, 256}, {1, 1, 256, 256}};
    UpsampleBicubic2dCompileInfo compileInfo = {48, 220};
    std::vector<int64_t> output_size = {256, 256};
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2d",
                                                {{input_shape, ge::DT_BF16, ge::FORMAT_ND}},
                                                {{out_shape, ge::DT_BF16, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;

    string expectTilingData =
        "64 3 4539628425446424576 17180131328 65536 9895604652288 17179869192 1 1 128 128 1 1 256 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 549755814016 824633721024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 549755814016 824633721024 1099511628032 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906944 274877906944 274877906944 274877906944 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 549755813952 549755813952 549755813952 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906944 824633720960 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 1099511627968 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 549755814144 549755814016 154618822720 274877907072 4294967344 4294967297 1 0 79164837199872 32768 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 549755814144 274877907072 154618822912 1099511627840 4294967344 4294967297 1 0 131941395333120 65536 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33656832};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(UpsampleBicubic2dTiling, upsample_bicubic2d_tiling_004)
{
    gert::StorageShape input_shape = {{1, 1, 128, 128}, {1, 1, 128, 128}};
    gert::StorageShape out_shape = {{1, 1, 256, 256}, {1, 1, 256, 256}};
    UpsampleBicubic2dCompileInfo compileInfo = {48, 200};
    std::vector<int64_t> output_size = {256, 256};
    string socVersion = "Ascend310P";
    gert::TilingContextPara tilingContextPara("UpsampleBicubic2d",
                                                {{input_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{out_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
                                                gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
                                                gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("scales_h", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
                                                &compileInfo, socVersion);
    uint64_t expectTilingKey = 1;

    string expectTilingData =
        "0 2 4287426846254956544 0 0 0 48 1 1 128 128 256 256 128 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 68719476736 68719476752 137438953504 206158430240 206158430256 274877907008 343597383744 343597383760 412316860512 481036337248 481036337264 549755814016 618475290752 618475290768 687194767520 755914244256 755914244272 824633721024 893353197760 893353197776 962072674528 1030792151264 1030792151280 0 68719476752 137438953488 137438953504 206158430256 274877906992 274877907008 343597383760 412316860496 412316860512 481036337264 549755814000 549755814016 618475290768 687194767504 687194767520 755914244272 824633721008 824633721024 893353197776 962072674512 962072674528 1030792151280 1099511628016 1099511628032 0 369367187456 172 738734374998 369367187456 172 738734374998 369367187456 172 738734374998 369367187456 172 738734374998 369367187456 172 738734374998 369367187456 172 738734374998 369367187456 172 738734374998 369367187456 172 738734374998 0 738734374998 369367187712 1099511627948 738734374998 369367187712 1099511627948 738734374998 369367187712 1099511627948 738734374998 369367187712 1099511627948 738734374998 369367187712 1099511627948 738734374998 369367187712 1099511627948 738734374998 369367187712 1099511627948 738734374998 369367187712 1099511627948 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
