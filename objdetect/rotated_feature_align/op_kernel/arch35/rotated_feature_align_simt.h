/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTATED_FEATURE_ALIGN_SIMT_H_
#define ROTATED_FEATURE_ALIGN_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/math_functions.h"
#include "rotated_feature_align_tiling_data.h"
#include "rotated_feature_align_tiling_key.h"

#pragma clang fp contract(off)

namespace NsRotatedFeatureAlign {
using namespace AscendC;

static constexpr uint32_t THREAD_NUM = 512;
static constexpr int32_t OUT_OF_RANGE_BOUND = -1;
static constexpr float ZERO_F = 0.0f;
static constexpr float ONE_F = 1.0f;
static constexpr float HALF_F = 0.5f;
static constexpr int32_t MAX_SAMPLE_POINTS = 5;
static constexpr int32_t POINTS_MODE_1_NUM = 1;
static constexpr int32_t POINTS_MODE_5_NUM = 5;
static constexpr int32_t PARAM_Y_IDX = 0;
static constexpr int32_t PARAM_X_IDX = 1;
static constexpr int32_t PARAM_W_IDX = 2;
static constexpr int32_t PARAM_H_IDX = 3;
static constexpr int32_t PARAM_A_IDX = 4;
static constexpr uint64_t UB_DIV_SLOTS = 8;
static constexpr uint32_t UB_SLOT_DIV_HW = 0;
static constexpr uint32_t UB_SLOT_MAGIC_HW = 1;
static constexpr uint32_t UB_SLOT_SHIFT_HW = 2;
static constexpr uint32_t UB_SLOT_MAGIC_C = 4;
static constexpr uint32_t UB_SLOT_SHIFT_C = 5;

template <typename T>
__simt_callee__ inline float BilinearInterpolate(__gm__ const T* plane, int64_t height, int64_t width, float heightF,
                                                 float widthF, float y, float x)
{
    if (y < static_cast<float>(OUT_OF_RANGE_BOUND) || y > heightF || x < static_cast<float>(OUT_OF_RANGE_BOUND) ||
        x > widthF) {
        return ZERO_F;
    }
    y = fmaxf(y, ZERO_F);
    x = fmaxf(x, ZERO_F);
    int32_t yLow = static_cast<int32_t>(y);
    int32_t xLow = static_cast<int32_t>(x);
    int32_t yHigh = 0;
    int32_t xHigh = 0;
    if (yLow >= height - 1) {
        yLow = yHigh = static_cast<int32_t>(height - 1);
        y = static_cast<float>(yLow);
    } else {
        yHigh = yLow + 1;
    }
    if (xLow >= width - 1) {
        xLow = xHigh = static_cast<int32_t>(width - 1);
        x = static_cast<float>(xLow);
    } else {
        xHigh = xLow + 1;
    }
    const float ly = y - static_cast<float>(yLow);
    const float lx = x - static_cast<float>(xLow);
    const float hy = ONE_F - ly;
    const float hx = ONE_F - lx;
    const __gm__ T* rowLo = plane + static_cast<int64_t>(yLow) * width;
    const __gm__ T* rowHi = plane + static_cast<int64_t>(yHigh) * width;
    return (hy * hx) * static_cast<float>(rowLo[xLow]) + (hy * lx) * static_cast<float>(rowLo[xHigh]) +
           (ly * hx) * static_cast<float>(rowHi[xLow]) + (ly * lx) * static_cast<float>(rowHi[xHigh]);
}

template <uint32_t pointsMode, typename T>
__simt_callee__ inline float ProcessElement(uint64_t index, uint64_t bboxBase, uint64_t planeBase, uint64_t divHw,
                                            int64_t height, int64_t width, float spatialScale, __gm__ const T* xGm,
                                            __gm__ const T* bboxesGm)
{
    const float heightF = static_cast<float>(height);
    const float widthF = static_cast<float>(width);
    const float roiY = static_cast<float>(bboxesGm[bboxBase + PARAM_Y_IDX * divHw]) * spatialScale;
    const float roiX = static_cast<float>(bboxesGm[bboxBase + PARAM_X_IDX * divHw]) * spatialScale;
    float px[MAX_SAMPLE_POINTS];
    float py[MAX_SAMPLE_POINTS];
    px[0] = roiX;
    py[0] = roiY;
    if constexpr (pointsMode == RFA_POINTS_MODE_5) {
        const float roiW = static_cast<float>(bboxesGm[bboxBase + PARAM_W_IDX * divHw]) * spatialScale;
        const float roiH = static_cast<float>(bboxesGm[bboxBase + PARAM_H_IDX * divHw]) * spatialScale;
        const float roiA = static_cast<float>(bboxesGm[bboxBase + PARAM_A_IDX * divHw]);
        const float w2 = roiW * HALF_F;
        const float h2 = roiH * HALF_F;
        const float cosa = cosf(roiA);
        const float sina = sinf(roiA);
        volatile float wx = cosa * w2;
        volatile float wy = sina * w2;
        volatile float hx = -sina * h2;
        volatile float hy = cosa * h2;
        px[1] = roiX + wx + hx;
        py[1] = roiY + wy + hy;
        px[2] = roiX - wx + hx;
        py[2] = roiY - wy + hy;
        px[3] = roiX - wx - hx;
        py[3] = roiY - wy - hy;
        px[4] = roiX + wx - hx;
        py[4] = roiY + wy - hy;
    }
    constexpr int32_t kPoints = (pointsMode == RFA_POINTS_MODE_5) ? POINTS_MODE_5_NUM : POINTS_MODE_1_NUM;
    float outputVal = static_cast<float>(xGm[index]);
#pragma unroll
    for (int32_t i = 0; i < kPoints; ++i) {
        outputVal += BilinearInterpolate<T>(xGm + planeBase, height, width, heightF, widthF, py[i], px[i]);
    }
    return outputVal;
}

template <uint32_t pointsMode, typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpRotatedFeatureAlignSimt(
    uint64_t totalNum, int64_t height, int64_t width, float spatialScale, __ubuf__ const uint64_t* divUb,
    __gm__ const T* xGm, __gm__ const T* bboxesGm, __gm__ T* yGm)
{
    const uint64_t divHw = divUb[UB_SLOT_DIV_HW];
    const uint64_t magicHw = divUb[UB_SLOT_MAGIC_HW];
    const uint64_t shiftHw = divUb[UB_SLOT_SHIFT_HW];
    const uint64_t magicC = divUb[UB_SLOT_MAGIC_C];
    const uint64_t shiftC = divUb[UB_SLOT_SHIFT_C];

    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < totalNum;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint64_t u = Simt::UintDiv<uint64_t>(index, magicHw, shiftHw);
        const uint64_t t = index - u * divHw;
        const uint64_t n = Simt::UintDiv<uint64_t>(u, magicC, shiftC);
        const uint64_t bboxBase = n * static_cast<uint64_t>(BBOX_PARAM_NUM) * divHw + t;
        const uint64_t planeBase = index - t;
        const float outputVal = ProcessElement<pointsMode, T>(index, bboxBase, planeBase, divHw, height, width,
                                                              spatialScale, xGm, bboxesGm);
        yGm[index] = static_cast<T>(outputVal);
    }
}

template <uint32_t pointsMode, typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR bboxes, GM_ADDR y, const RotatedFeatureAlignTilingData* tilingData)
{
    const uint64_t totalNum = static_cast<uint64_t>(tilingData->totalNum);
    if (totalNum == 0) {
        return;
    }
    const uint64_t cDim = static_cast<uint64_t>(tilingData->cDim);
    const uint64_t hw = static_cast<uint64_t>(tilingData->hDim) * static_cast<uint64_t>(tilingData->wDim);
    const float spatialScale = tilingData->spatialScale;
    __gm__ const T* xGm = (__gm__ const T*)x;
    __gm__ const T* bboxesGm = (__gm__ const T*)bboxes;
    __gm__ T* yGm = (__gm__ T*)y;

    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    LocalTensor<uint64_t> divUbTensor = ubAlloc.Alloc<uint64_t>(UB_DIV_SLOTS);
    __ubuf__ uint64_t* divUb = (__ubuf__ uint64_t*)divUbTensor.GetPhyAddr();
    uint64_t magic = 0;
    uint64_t shift = 0;
    GetUintDivMagicAndShift<uint64_t>(magic, shift, hw);
    divUb[UB_SLOT_DIV_HW] = hw;
    divUb[UB_SLOT_MAGIC_HW] = magic;
    divUb[UB_SLOT_SHIFT_HW] = shift;
    GetUintDivMagicAndShift<uint64_t>(magic, shift, cDim);
    divUb[3] = cDim;
    divUb[UB_SLOT_MAGIC_C] = magic;
    divUb[UB_SLOT_SHIFT_C] = shift;
    DataSyncBarrier<MemDsbT::UB>();

    asc_vf_call<OpRotatedFeatureAlignSimt<pointsMode, T>>(dim3(THREAD_NUM), totalNum, tilingData->hDim,
                                                          tilingData->wDim, spatialScale, divUb, xGm, bboxesGm, yGm);
}
} // namespace NsRotatedFeatureAlign
#endif // ROTATED_FEATURE_ALIGN_SIMT_H_
