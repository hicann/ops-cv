/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STACK_GROUP_POINTS_H_
#define STACK_GROUP_POINTS_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "stack_group_points_tiling_data.h"
#include "stack_group_points_tiling_key.h"

namespace NsStackGroupPoints {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpStackGroupPointsSimt(
    int64_t m, int64_t c, int64_t nsample, int64_t b, int64_t n, int64_t totalElements, __gm__ T* features,
    __gm__ int32_t* featuresBatchCnt, __gm__ int32_t* indices, __gm__ int32_t* indicesBatchCnt, __gm__ T* y)
{
    int64_t nsampleC = nsample * c;
    int64_t mNsample = m * nsample;
    uint64_t stride = static_cast<uint64_t>(blockDim.x * gridDim.x);
    int64_t featLen = n * c;

    if (b <= 0) {
        for (uint64_t i = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
             i < static_cast<uint64_t>(totalElements); i += stride) {
            y[i] = static_cast<T>(0);
        }
        return;
    }

    for (uint64_t index = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
         index < static_cast<uint64_t>(totalElements); index += stride) {
        int64_t sampleIdx = index % nsample;
        int64_t cIdx = (index / nsample) % c;
        int64_t ptIdx = index / nsampleC;

        int32_t bsIdx = 0;
        int32_t ptCnt = indicesBatchCnt[0];
        for (int32_t k = 1; k < b; k++) {
            if (ptIdx >= ptCnt) {
                ptCnt += indicesBatchCnt[k];
                bsIdx = k;
            }
        }

        int32_t featuresBatchStartIdx = 0;
        int32_t featuresBatchEndIdx = featuresBatchCnt[0];
        for (int32_t k = 0; k < bsIdx; k++) {
            featuresBatchStartIdx += featuresBatchCnt[k];
            featuresBatchEndIdx = featuresBatchStartIdx + featuresBatchCnt[k + 1];
        }

        int64_t tmpCin = ptIdx * nsample + sampleIdx;
        int32_t cin = 0;
        if (tmpCin < mNsample) {
            cin = indices[tmpCin];
        }

        T result = static_cast<T>(0);
        if (cin < featuresBatchEndIdx) {
            int64_t inIdx = static_cast<int64_t>(cin) * c + cIdx;
            int64_t fsIdx = inIdx + static_cast<int64_t>(featuresBatchStartIdx) * c;
            if (fsIdx >= 0 && fsIdx < featLen) {
                result = features[fsIdx];
            }
        }
        y[index] = result;
    }
}

template <typename T>
__aicore__ inline void Process(GM_ADDR features, GM_ADDR features_batch_cnt, GM_ADDR indices, GM_ADDR indices_batch_cnt,
                               GM_ADDR y, int64_t m, int64_t c, int64_t nsample, int64_t b, int64_t n,
                               int64_t totalElements)
{
    __gm__ T* featuresGm = (__gm__ T*)features;
    __gm__ int32_t* featuresBatchCntGm = (__gm__ int32_t*)features_batch_cnt;
    __gm__ int32_t* indicesGm = (__gm__ int32_t*)indices;
    __gm__ int32_t* indicesBatchCntGm = (__gm__ int32_t*)indices_batch_cnt;
    __gm__ T* yGm = (__gm__ T*)y;

    asc_vf_call<OpStackGroupPointsSimt<T>>(dim3(THREAD_NUM), m, c, nsample, b, n, totalElements, featuresGm,
                                           featuresBatchCntGm, indicesGm, indicesBatchCntGm, yGm);
}

} // namespace NsStackGroupPoints
#endif // STACK_GROUP_POINTS_H_
