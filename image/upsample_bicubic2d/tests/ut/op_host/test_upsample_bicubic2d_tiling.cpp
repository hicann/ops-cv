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
    uint32_t coreNum = 0;
    uint16_t socVersionType = 0;
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
        {gert::TilingContextPara::OpAttr(
             "output_size", Ops::Cv::AnyValue::CreateFrom<std::vector<int64_t>>(output_size)),
            gert::TilingContextPara::OpAttr("scales_w", Ops::Cv::AnyValue::CreateFrom<float>(0.0))},
        &compileInfo);
    uint64_t expectTilingKey = 1;

    string expectTilingData =
        "64 2 4539628425446424576 17180131328 131072 9895604652288 17179869192 1 1 128 128 1 1 256 256 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 "
        "549755814016 824633721024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 549755814016 824633721024 "
        "1099511628032 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906944 274877906944 274877906944 274877906944 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 549755813952 549755813952 549755813952 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 274877906944 824633720960 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813952 "
        "1099511627968 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "4294967297 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 549755814144 549755814016 154618822720 "
        "274877907072 4294967336 4294967297 1 0 131941395333120 32768 4294967297 4294967297 4294967297 0 8589934594 1 "
        "0 0 0 0 0 0 0 0 1099511627777 549755814144 274877907072 154618822912 1099511627840 8589934624 4294967298 1 0 "
        "351843720888320 65536 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33759232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
