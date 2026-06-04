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
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_host/mrgba_custom_tiling.h"

using namespace ge;
using namespace std;

class TilingForMrgbaCustom : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "TilingForMrgbaCustom SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "TilingForMrgbaCustom TearDown" << std::endl;
  }
};

struct MrgbaCustomCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

TEST_F(TilingForMrgbaCustom, mrgba_custom_tiling_0)
{
    MrgbaCustomCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("MrgbaCustom",
                                                {{{{480, 640, 3}, {480, 640, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                {{{480, 640, 1}, {480, 640, 1}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                                {{{{480, 640, 4}, {480, 640, 4}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                                {},
                                                &compileInfo);

    std::vector<size_t> expectWorkspaces = {};
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    uint8_t* tilingDataPtr = tilingInfo.tilingData.get();
    EXPECT_EQ(tilingInfo.workspaceSizes.size(), expectWorkspaces.size());
    for (size_t i = 0; i < tilingInfo.workspaceSizes.size(); i++) {
        EXPECT_EQ(tilingInfo.workspaceSizes[i], expectWorkspaces[i]);
    }
    uint32_t expectedAlphaLen = 480 * 640 * 1;
    EXPECT_EQ(*reinterpret_cast<uint32_t*>(tilingDataPtr), expectedAlphaLen);
}

TEST_F(TilingForMrgbaCustom, mrgba_custom_tiling_1)
{
    MrgbaCustomCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("MrgbaCustom",
                                                {{{{1080, 1920, 3}, {1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                {{{1080, 1920, 1}, {1080, 1920, 1}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                                {{{{1080, 1920, 4}, {1080, 1920, 4}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                                {},
                                                &compileInfo);

    std::vector<size_t> expectWorkspaces = {};
    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    uint8_t* tilingDataPtr = tilingInfo.tilingData.get();
    EXPECT_EQ(tilingInfo.workspaceSizes.size(), expectWorkspaces.size());
    for (size_t i = 0; i < tilingInfo.workspaceSizes.size(); i++) {
        EXPECT_EQ(tilingInfo.workspaceSizes[i], expectWorkspaces[i]);
    }
    uint32_t expectedAlphaLen = 1080 * 1920 * 1;
    EXPECT_EQ(*reinterpret_cast<uint32_t*>(tilingDataPtr), expectedAlphaLen);
}