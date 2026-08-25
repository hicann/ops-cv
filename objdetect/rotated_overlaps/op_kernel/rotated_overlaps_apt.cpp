/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/rotated_overlaps_kernel.h"

template <uint32_t trans, uint32_t use32Bit>
__global__ __aicore__ void rotated_overlaps(GM_ADDR boxes, GM_ADDR query_boxes, GM_ADDR overlaps, GM_ADDR workspace,
                                            GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(RotatedOverlapsTilingData);
    GET_TILING_DATA_WITH_STRUCT(RotatedOverlapsTilingData, tilingData, tiling);
    constexpr bool kTrans = trans != 0U;
    constexpr bool kUse32Bit = use32Bit != 0U;
    if (tilingData.usePairParallelSimt != 0U) {
        NsRotatedOverlaps::ProcessPairParallelSimt<kTrans, kUse32Bit>(boxes, query_boxes, overlaps, &tilingData);
        return;
    }
    NsRotatedOverlaps::RotatedOverlapsKernel<kTrans, kUse32Bit> op;
    op.Init(boxes, query_boxes, overlaps, &tilingData);
    op.Process();
}
