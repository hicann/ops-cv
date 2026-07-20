/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_upsample_trilinear_apt.cpp
 * \brief ResizeUpsampleTrilinear A950 SIMD/SIMT kernel entry.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./arch35/resize_upsample_trilinear_simt.h"
#include "./arch35/resize_upsample_trilinear_simd.h"
#include "./arch35/resize_upsample_trilinear_full3d_simd.h"
#include "./arch35/resize_upsample_trilinear_tiling_data.h"
#include "./arch35/resize_upsample_trilinear_tiling_key.h"

using namespace ResizeUpsampleTrilinear;

__aicore__ inline bool CanUseDOnlySimd(const ResizeUpsampleTrilinearRegBaseTilingData& tilingData)
{
    constexpr int64_t D_ONLY_MIN_ELEMENTS = 1024 * 1024;
    bool isDOnly = tilingData.inD != tilingData.outD && tilingData.inH == tilingData.outH &&
                   tilingData.inW == tilingData.outW && tilingData.scaleD != 1.0f && tilingData.scaleH == 1.0f &&
                   tilingData.scaleW == 1.0f;
    bool coordinateSafe = tilingData.alignCorners == 1 || tilingData.scaleD == 0.5f;
    int64_t totalNc = tilingData.lenN * tilingData.lenC;
    int64_t outputElements = totalNc * tilingData.outD * tilingData.outH * tilingData.outW;
    return isDOnly && coordinateSafe && outputElements >= D_ONLY_MIN_ELEMENTS &&
           totalNc >= static_cast<int64_t>(AscendC::GetBlockNum());
}

__aicore__ inline bool CanUseFull3dSimd(const ResizeUpsampleTrilinearRegBaseTilingData& tilingData)
{
    bool inputInRange = tilingData.inD >= FULL3D_MIN_IN_D && tilingData.inD <= FULL3D_IN_D &&
                        tilingData.inH >= FULL3D_MIN_IN_H && tilingData.inH <= FULL3D_IN_H &&
                        tilingData.inW >= FULL3D_MIN_IN_W && tilingData.inW <= FULL3D_IN_W;
    bool outputInRange = tilingData.outD >= FULL3D_MIN_OUT_D && tilingData.outD <= FULL3D_OUT_D &&
                         tilingData.outH >= FULL3D_MIN_OUT_H && tilingData.outH <= FULL3D_OUT_H &&
                         tilingData.outW >= FULL3D_MIN_OUT_W && tilingData.outW <= FULL3D_OUT_W;
    bool widthAligned = tilingData.inW % FULL3D_WIDTH_ALIGN == 0 && tilingData.outW % FULL3D_WIDTH_ALIGN == 0;
    bool isFull3dUpsample = tilingData.outD > tilingData.inD && tilingData.outH > tilingData.inH &&
                            tilingData.outW > tilingData.inW;
    bool scaleValid = tilingData.scaleD > 0.0f && tilingData.scaleH > 0.0f && tilingData.scaleW > 0.0f;
    return inputInRange && outputInRange && widthAligned && isFull3dUpsample && scaleValid;
}

__aicore__ inline bool IsFixedFull3dShape(const ResizeUpsampleTrilinearRegBaseTilingData& tilingData)
{
    return tilingData.inD == FULL3D_IN_D && tilingData.inH == FULL3D_IN_H && tilingData.inW == FULL3D_IN_W &&
           tilingData.outD == FULL3D_OUT_D && tilingData.outH == FULL3D_OUT_H && tilingData.outW == FULL3D_OUT_W;
}

__aicore__ inline void RunFull3dSimd(GM_ADDR input, GM_ADDR output,
                                     const ResizeUpsampleTrilinearRegBaseTilingData& tilingData)
{
    if (IsFixedFull3dShape(tilingData)) {
        ResizeUpsampleTrilinearFull3dSimd<DTYPE_INPUT, true> op;
        op.Init(input, output, &tilingData);
        op.Process();
    } else {
        ResizeUpsampleTrilinearFull3dSimd<DTYPE_INPUT, false> op;
        op.Init(input, output, &tilingData);
        op.Process();
    }
}

__aicore__ inline bool CanUseDReuseNcHw(const ResizeUpsampleTrilinearRegBaseTilingData& tilingData)
{
    constexpr int64_t D_REUSE_MIN_OUTPUT_ELEMENTS = 60 * 1024 * 1024;
    constexpr int64_t D_REUSE_MIN_EXPANSION = 4;
    int64_t outputElements = tilingData.lenN * tilingData.lenC * tilingData.outD * tilingData.outH * tilingData.outW;
    return outputElements >= D_REUSE_MIN_OUTPUT_ELEMENTS && tilingData.outD >= tilingData.inD * D_REUSE_MIN_EXPANSION &&
           (tilingData.outH != tilingData.inH || tilingData.outW != tilingData.inW);
}

template <typename IndexT>
__aicore__ inline void RunSimt(GM_ADDR input, GM_ADDR output,
                               const ResizeUpsampleTrilinearRegBaseTilingData& tilingData)
{
    ResizeUpsampleTrilinearSimt<DTYPE_INPUT, IndexT> op;
    op.Init(input, output, &tilingData);
    op.Process();
}

__aicore__ inline void RunNcHw(GM_ADDR input, GM_ADDR output,
                               const ResizeUpsampleTrilinearRegBaseTilingData& tilingData)
{
    GlobalTensor<DTYPE_INPUT> inputGm;
    GlobalTensor<DTYPE_INPUT> outputGm;
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INPUT*>(input));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INPUT*>(output));
    ProcessNcHw(inputGm, outputGm, &tilingData);
}

template <bool UseTilingSplit>
__aicore__ inline void RunDReuseNcHw(GM_ADDR input, GM_ADDR output,
                                     const ResizeUpsampleTrilinearRegBaseTilingData& tilingData)
{
    GlobalTensor<DTYPE_INPUT> inputGm;
    GlobalTensor<DTYPE_INPUT> outputGm;
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INPUT*>(input));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INPUT*>(output));
    ProcessDReuseNcHw<DTYPE_INPUT, UseTilingSplit>(inputGm, outputGm, &tilingData);
}

template <uint64_t isInt32>
__aicore__ inline void RunGeneric(GM_ADDR input, GM_ADDR output,
                                  const ResizeUpsampleTrilinearRegBaseTilingData& tilingData)
{
    if constexpr (isInt32 == 1) {
        if (CanUseFull3dSimd(tilingData)) {
            RunFull3dSimd(input, output, tilingData);
            return;
        }
        if (CanUseDReuseNcHw(tilingData)) {
            RunDReuseNcHw<false>(input, output, tilingData);
            return;
        }
    }
    if (CanUseDOnlySimd(tilingData)) {
        ResizeUpsampleTrilinearSimd<DTYPE_INPUT> op;
        op.Init(input, output, &tilingData);
        op.Process();
        return;
    }
    if constexpr (isInt32 == 1) {
        RunSimt<uint32_t>(input, output, tilingData);
    } else {
        RunSimt<uint64_t>(input, output, tilingData);
    }
}

template <uint64_t schId, uint64_t isInt32>
__global__ __aicore__ void resize_upsample_trilinear(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ResizeUpsampleTrilinearRegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(ResizeUpsampleTrilinearRegBaseTilingData, tilingData, tiling);
    (void)workspace;

    if constexpr (schId == RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_D_ONLY) {
        ResizeUpsampleTrilinearSimd<DTYPE_INPUT> op;
        op.Init(input, output, &tilingData);
        op.Process();
    } else if constexpr (schId == RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_FULL_3D_SIMD) {
        if constexpr (isInt32 == 1) {
            RunFull3dSimd(input, output, tilingData);
        } else {
            RunSimt<uint64_t>(input, output, tilingData);
        }
    } else if constexpr (schId == RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_NC_HW) {
        RunNcHw(input, output, tilingData);
    } else if constexpr (schId == RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_D_REUSE_NC_HW) {
        if constexpr (isInt32 == 1) {
            RunDReuseNcHw<true>(input, output, tilingData);
        } else {
            RunSimt<uint64_t>(input, output, tilingData);
        }
    } else if constexpr (schId == RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_NCDHW ||
                         schId == RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_COPY) {
        // Some binary-generation/runtime combinations resolve a valid D-only
        // request to the generic template variant. Recheck the data-dependent
        // predicate here so the optimized path does not depend solely on the
        // template key selected by the launcher.
        RunGeneric<isInt32>(input, output, tilingData);
    }
}
