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
#include <iostream>
#include <vector>
#include "../../../op_host/arch35/nms_with_mask_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

constexpr uint64_t TILING_KEY_FOR_MULTICORE = 10000UL;
constexpr int64_t BIT_PER_BYTE = 8;

class NMSWithMaskTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NMSWithMaskTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NMSWithMaskTiling TearDown" << std::endl;
    }
};

void CalculateTilingData(NMSWithMaskTilingData &tilingData, int64_t boxesNum, int64_t vectorCoreNum, int64_t groupSize) 
{
    tilingData.boxesNum = boxesNum;
    tilingData.groupSize = groupSize;
    tilingData.groupNum = (boxesNum + groupSize - 1) / groupSize;
    tilingData.blockNum = tilingData.groupNum * (tilingData.groupNum + 1) / 2;
    tilingData.usedCoreNum = tilingData.blockNum > vectorCoreNum ? vectorCoreNum : tilingData.blockNum;
    tilingData.headCoreNum = tilingData.blockNum % tilingData.usedCoreNum;
    tilingData.headCoreNum = tilingData.headCoreNum == 0 ? tilingData.usedCoreNum : tilingData.headCoreNum;
    tilingData.blockPerHead = (tilingData.blockNum + tilingData.usedCoreNum - 1) / tilingData.usedCoreNum;
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_test_float32_case1)
{
    const uint32_t boxesNum = 16;
    const uint32_t m_dim = 5;
    gert::StorageShape box_scores = {{boxesNum, m_dim}, {boxesNum, m_dim}};
    gert::StorageShape selected_boxes = {{boxesNum, m_dim}, {boxesNum, m_dim}};
    gert::StorageShape selected_idx = {{boxesNum}, {boxesNum}};
    gert::StorageShape selected_mask = {{boxesNum}, {boxesNum}};
    
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    NMSWithMaskTilingData tilingData;
    CalculateTilingData(tilingData, boxesNum, compileInfo.coreNum, 256);

    gert::TilingContextPara tilingContextPara(
        "NMSWithMask",
        {{box_scores, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {selected_boxes, ge::DT_FLOAT, ge::FORMAT_ND},
            {selected_idx, ge::DT_INT32, ge::FORMAT_ND},
            {selected_mask, ge::DT_UINT8, ge::FORMAT_ND}
        },
        {gert::TilingContextPara::OpAttr("iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.5))},
        &compileInfo
    );

    uint64_t expectedTilingKey = TILING_KEY_FOR_MULTICORE;
    string expectTilingData = "16 1 256 1 1 1 1 1056964608 ";
    // size_t totalWorkspaceBytes = tilingData.blockNum * tilingData.groupSize * tilingData.groupSize / BIT_PER_BYTE;
    std::vector<size_t> expectWorkspaces = {4294975487};
    // expectWorkspaces[0] += totalWorkspaceBytes;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectTilingData, expectWorkspaces);
}