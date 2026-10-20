/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_kernel/roi_align_v2.cpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "tikicpulib.h"

TEST(RoiAlignV2AdaptiveGridTest, roundsFloatRatioAboveIntegerUp)
{
    constexpr uint32_t numRois = 2;
    constexpr std::size_t featureBytes = 6 * sizeof(float);
    constexpr std::size_t roiBytes = numRois * 5 * sizeof(float);
    constexpr std::size_t outputBytes = numRois * sizeof(float);
    constexpr uint32_t numBlocks = 1;

    uint8_t* features = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(featureBytes));
    uint8_t* rois = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(roiBytes));
    uint8_t* output = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputBytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(RoiAlignV2TilingData)));
    ASSERT_NE(features, nullptr);
    ASSERT_NE(rois, nullptr);
    ASSERT_NE(output, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    const float featureValues[] = {0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 10.0f};
    const float roiValues[] = {
        0.0f, 0.5f, 0.5f, 1.505f, 1.5f, // width ratio is just above 1.0f
        0.0f, 0.5f, 0.5f, 1.5f,   1.5f, // exact-integer control
    };
    std::memcpy(features, featureValues, featureBytes);
    std::memcpy(rois, roiValues, roiBytes);
    std::memset(output, 0, outputBytes);
    std::memset(tiling, 0, sizeof(RoiAlignV2TilingData));

    auto* tilingData = reinterpret_cast<RoiAlignV2TilingData*>(tiling);
    tilingData->baseRoisPerCore = numRois;
    tilingData->bigTotalRois = numRois;
    tilingData->tailRoiNum = 0;
    tilingData->featureTotalSize = 6;
    tilingData->roiLength = 5;
    tilingData->featureMapSize = 6;
    tilingData->outRoiSize = 1;
    tilingData->numRois = numRois;
    tilingData->batch = 1;
    tilingData->channels = 1;
    tilingData->height = 2;
    tilingData->width = 3;
    tilingData->pooledHeight = 1;
    tilingData->pooledWidth = 1;
    tilingData->spatialScale = 1.0f;
    tilingData->samplingRatio = 0;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(roi_align_v2<0>, numBlocks, features, rois, output, workspace, tiling);

    const auto* actual = reinterpret_cast<const float*>(output);
    EXPECT_NEAR(actual[0], 1.26875f, 1e-5f);
    EXPECT_NEAR(actual[1], 0.0f, 1e-6f);

    AscendC::GmFree(features);
    AscendC::GmFree(rois);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
