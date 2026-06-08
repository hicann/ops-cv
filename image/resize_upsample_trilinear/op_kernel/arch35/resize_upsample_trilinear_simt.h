/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_upsample_trilinear_simt.h
 * \brief ResizeUpsampleTrilinear SIMT kernel implementation for A5
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_SIMT_H
#define RESIZE_UPSAMPLE_TRILINEAR_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./resize_upsample_trilinear_simt_base.h"
#include "./resize_upsample_trilinear_tiling_data.h"

namespace ResizeUpsampleTrilinear {
using namespace AscendC;

template <typename T>
class ResizeUpsampleTrilinearSimt {
public:
    __aicore__ inline ResizeUpsampleTrilinearSimt() = default;
    __aicore__ inline void Init(GM_ADDR output, GM_ADDR input,
        const ResizeUpsampleTrilinearArch35TilingData* __restrict tiling);
    __aicore__ inline void Process();

private:
    const ResizeUpsampleTrilinearArch35TilingData* tilingData;
    uint32_t bid = 0;
    GlobalTensor<T> outputGm;
    GlobalTensor<T> inputGm;
};

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimt<T>::Init(
    GM_ADDR output, GM_ADDR input, const ResizeUpsampleTrilinearArch35TilingData* __restrict tiling)
{
    outputGm.SetGlobalBuffer((__gm__ T*)output);
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    tilingData = tiling;
    bid = static_cast<uint32_t>(GetBlockIdx());
}

template <typename T>
__aicore__ inline void ResizeUpsampleTrilinearSimt<T>::Process()
{
    uint32_t baseElementsPerBlock = tilingData->base_elements_per_block;
    uint32_t blockCount = tilingData->block_count;
    uint32_t tailElements = tilingData->tail_elements;

    if (bid >= blockCount) {
        return;
    }

    __gm__ T* outputPtr = (__gm__ T*)outputGm.GetAddr();
    __gm__ T* inputPtr = (__gm__ T*)inputGm.GetAddr();

    if (tilingData->use_int32 != 0) {
        int32_t blkStartOffset = static_cast<int32_t>(bid) * static_cast<int32_t>(baseElementsPerBlock);
        int32_t blkProcessNum;
        if (bid == blockCount - 1) {
            blkProcessNum = static_cast<int32_t>(tailElements);
        } else {
            blkProcessNum = static_cast<int32_t>(baseElementsPerBlock);
        }

        if (blkProcessNum <= 0) {
            return;
        }

        calleeInt32<T, int32_t>(outputPtr, inputPtr, blkStartOffset, blkProcessNum, tilingData);
    } else {
        int64_t blkStartOffset = static_cast<int64_t>(bid) * static_cast<int64_t>(baseElementsPerBlock);
        int64_t blkProcessNum;
        if (bid == blockCount - 1) {
            blkProcessNum = static_cast<int64_t>(tailElements);
        } else {
            blkProcessNum = static_cast<int64_t>(baseElementsPerBlock);
        }

        if (blkProcessNum <= 0) {
            return;
        }

        calleeInt64<T, int64_t>(outputPtr, inputPtr, blkStartOffset, blkProcessNum, tilingData);
    }
}
} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_SIMT_H