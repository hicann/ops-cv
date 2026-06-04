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
#include "../../../op_host/blend_images_custom_tiling.h"

using namespace ge;
using namespace std;

class TilingForBlendImagesCustom : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "TilingForBlendImagesCustom SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "TilingForBlendImagesCustom TearDown" << std::endl;
  }
};

struct BlendImagesCustomCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

TEST_F(TilingForBlendImagesCustom, blend_images_custom_tiling_0)
{
    BlendImagesCustomCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("BlendImagesCustom",
                                                {{{{480, 640, 3}, {480, 640, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                {{{480, 640, 1}, {480, 640, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                {{{480, 640, 3}, {480, 640, 3}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                                {{{{480, 640, 3}, {480, 640, 3}}, ge::DT_UINT8, ge::FORMAT_ND}},
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
    uint32_t expectedTotalAlphaLength = 480 * 640 * 1;
    EXPECT_EQ(*reinterpret_cast<uint32_t*>(tilingDataPtr), expectedTotalAlphaLength);
}

TEST_F(TilingForBlendImagesCustom, blend_images_custom_tiling_1)
{
    BlendImagesCustomCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("BlendImagesCustom",
                                                {{{{1080, 1920, 3}, {1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                {{{1080, 1920, 1}, {1080, 1920, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                {{{1080, 1920, 3}, {1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                                {{{{1080, 1920, 3}, {1080, 1920, 3}}, ge::DT_UINT8, ge::FORMAT_ND}},
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
    uint32_t expectedTotalAlphaLength = 1080 * 1920 * 1;
    EXPECT_EQ(*reinterpret_cast<uint32_t*>(tilingDataPtr), expectedTotalAlphaLength);
}