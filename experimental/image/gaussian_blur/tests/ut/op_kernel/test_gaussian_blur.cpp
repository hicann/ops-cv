/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "gaussian_blur_tiling.h"
#include "../../../op_kernel/arch35/gaussian_blur_kernel.h"

namespace {

__global__ __aicore__ void gaussian_blur_row_ut(GM_ADDR src, GM_ADDR dst, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA_WITH_STRUCT(GaussianBlurTilingData, tilingData, tiling);
    NsGaussianBlurW128::ProcessRow(src, dst, &tilingData);
}

__global__ __aicore__ void gaussian_blur_column_ut(GM_ADDR src, GM_ADDR dst, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA_WITH_STRUCT(GaussianBlurTilingData, tilingData, tiling);
    NsGaussianBlurW128::ProcessColumn(src, dst, &tilingData);
}

GaussianBlurTilingData MakeIdentityTiling()
{
    GaussianBlurTilingData tilingData{};
    tilingData.h = 1U;
    tilingData.w = 1U;
    tilingData.c = 1U;
    tilingData.totalTiles = 1U;
    tilingData.tilesX = 1U;
    tilingData.tilesY = 1U;
    tilingData.kernelSize = 1U;
    tilingData.kernelSizeY = 1U;
    tilingData.pathMode = GAUSSIAN_BLUR_PATH_C1_FAST;
    tilingData.reserved[0] = 1U;
    tilingData.reserved[1] = 1U;
    tilingData.weights[0] = 1.0F;
    tilingData.weightsY[0] = 1.0F;
    return tilingData;
}

template <typename Kernel>
void RunIdentityKernel(Kernel kernel)
{
    auto* src = static_cast<float*>(AscendC::GmAlloc(sizeof(float)));
    auto* dst = static_cast<float*>(AscendC::GmAlloc(sizeof(float)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(GaussianBlurTilingData)));
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_NE(tiling, nullptr);

    src[0] = 3.25F;
    dst[0] = 0.0F;
    const auto tilingData = MakeIdentityTiling();
    std::memcpy(tiling, &tilingData, sizeof(tilingData));

    ICPU_RUN_KF(kernel, 1U, reinterpret_cast<uint8_t*>(src), reinterpret_cast<uint8_t*>(dst), nullptr, tiling);
    EXPECT_FLOAT_EQ(dst[0], src[0]);

    AscendC::GmFree(src);
    AscendC::GmFree(dst);
    AscendC::GmFree(tiling);
}

TEST(GaussianBlurKernelTest, row_identity_k1) { RunIdentityKernel(gaussian_blur_row_ut); }

TEST(GaussianBlurKernelTest, column_identity_k1) { RunIdentityKernel(gaussian_blur_column_ut); }

} // namespace
