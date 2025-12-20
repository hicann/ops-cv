/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include <cstdint>
#include "../../../op_host/iou_v2_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void iou_v2(
    GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR overlap, GM_ADDR workspace, GM_ADDR tiling);

class iou_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "iou_v2 SetUp\n"
             << endl;
    }
    static void TearDownTestCase()
    {
        cout << "iou_v2_test TearDown\n"
             << endl;
    }
};

// [4, 8], aligned=true, iou, float32
TEST_F(iou_v2_test, test_aligned_iou_fp32)
{

    size_t bboxesByteSize = 4 * 8 * sizeof(float);
    size_t gtboxesByteSize = 4 * 8 * sizeof(float);
    size_t overlapByteSize = 8 * 1 * sizeof(float);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    IouV2TilingData* tilingData = reinterpret_cast<IouV2TilingData*>(tiling);
    tilingData->bBoxLength = 8;
    tilingData->gtBoxLength = 8;
    tilingData->frontCoreNum = 0;
    tilingData->loopNum = 1;
    tilingData->tileLength = 8;
    tilingData->subTileLen = 8;
    tilingData->eps = 1;

    ICPU_SET_TILING_KEY(4);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, tiling);

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}