/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_pooling_with_arg_max_apt.cpp
 * \brief roi_pooling_with_arg_max kernel entry
 */

#include "./arch35/roi_pooling_with_arg_max_simt.h"
#include "./arch35/roi_pooling_with_arg_max_tiling_data.h"
#include "./arch35/roi_pooling_with_arg_max_tiling_key.h"
using namespace AscendC;

template <uint64_t dType>
__global__ __aicore__ void roi_pooling_with_arg_max(GM_ADDR x, GM_ADDR rois, GM_ADDR roi_actual_num, GM_ADDR y,
                                                    GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(RoiPoolingWithArgMaxRegBaseTilingData);
    GET_TILING_DATA(tilingDataIn, tiling);
    const RoiPoolingWithArgMaxRegBaseTilingData* __restrict tilingData = &tilingDataIn;

    uint32_t mPoolW = 0, shiftPoolW = 0, mPoolH = 0, shiftPoolH = 0, mCH = 0, shiftCH = 0;
    GetUintDivMagicAndShift(mPoolW, shiftPoolW, static_cast<uint32_t>(tilingData->poolW));
    GetUintDivMagicAndShift(mPoolH, shiftPoolH, static_cast<uint32_t>(tilingData->poolH));
    GetUintDivMagicAndShift(mCH, shiftCH, static_cast<uint32_t>(tilingData->channels));

    const int32_t ch = static_cast<int32_t>(tilingData->channels);
    const int32_t fh = static_cast<int32_t>(tilingData->fmHeight);
    const int32_t fw = static_cast<int32_t>(tilingData->fmWidth);
    const int32_t rn = static_cast<int32_t>(tilingData->roiNumber);
    const int32_t ph = static_cast<int32_t>(tilingData->poolH);
    const int32_t pw = static_cast<int32_t>(tilingData->poolW);
    const float sh = tilingData->spatialH;
    const float sw = tilingData->spatialW;
    const auto indicesPtr = (__gm__ int32_t*)indices;

    if constexpr (dType == ROI_POOLING_WITH_ARG_MAX_TPL_FP32) {
        Simt::VF_CALL<RoiPoolingWithArgMaxCompute<float>>(
            dim3{1024, 1, 1},
            (__gm__ float*)x, (__gm__ float*)rois, (__gm__ float*)roi_actual_num, (__gm__ float*)y,
            indicesPtr, ch, fh, fw, rn, ph, pw, sh, sw,
            mPoolW, shiftPoolW, mPoolH, shiftPoolH, mCH, shiftCH);
    } else if constexpr (dType == ROI_POOLING_WITH_ARG_MAX_TPL_FP16) {
        Simt::VF_CALL<RoiPoolingWithArgMaxCompute<half>>(
            dim3{1024, 1, 1},
            (__gm__ half*)x, (__gm__ half*)rois, (__gm__ half*)roi_actual_num, (__gm__ half*)y,
            indicesPtr, ch, fh, fw, rn, ph, pw, sh, sw,
            mPoolW, shiftPoolW, mPoolH, shiftPoolH, mCH, shiftCH);
    }
}
