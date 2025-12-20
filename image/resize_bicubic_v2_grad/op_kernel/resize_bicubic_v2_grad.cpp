/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file resize_bicubic_v2_grad.cpp
 * \brief resize_bicubic_v2_grad
 */

#include "./arch35/resize_bicubic_v2_grad_all_copy.h"
#include "./arch35/resize_bicubic_v2_grad_simt.h"
#include "./arch35/resize_bicubic_v2_grad_simt_determine.h"

#define TILING_KEY_SIMT 10000
#define TILING_KEY_SIMT_IDX64 10001
#define TILING_KEY_SIMT_DETERMINE 20000
#define TILING_KEY_SIMT_DETERMINE_IDX64 20001
#define TILING_KEY_ALL_COPY 30000

using namespace AscendC;

extern "C" __global__ __aicore__ void resize_bicubic_v2_grad(
    GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_ALL_COPY)) {
        GET_TILING_DATA_WITH_STRUCT(ResizeBicubicV2GradAllCopyTilingData, tilingDataIn, tiling);
        const ResizeBicubicV2GradAllCopyTilingData *__restrict__ tilingData = &tilingDataIn;
        ResizeBicubicV2Grad::ResizeBicubicV2GradAllCopy<DTYPE_Y> op;
        op.Init(grads, y, &pipe, tilingData);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT)) {
        GET_TILING_DATA_WITH_STRUCT(ResizeBicubicV2GradSimtTilingData, tilingDataIn, tiling);
        const ResizeBicubicV2GradSimtTilingData *__restrict__ tilingData = &tilingDataIn;
        if (tilingData->format == 0) {
            if (tilingData->alignCorners > 0) {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimt<DTYPE_Y, uint32_t, int32_t, FORMAT_NCHW, true> op;
                op.Init(grads, y, tilingData);
                op.Process();
            } else {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimt<DTYPE_Y, uint32_t, int32_t, FORMAT_NCHW, false> op;
                op.Init(grads, y, tilingData);
                op.Process();
            }
        } else {
            if (tilingData->alignCorners > 0) {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimt<DTYPE_Y, uint32_t, int32_t, FORMAT_NHWC, true> op;
                op.Init(grads, y, tilingData);
                op.Process();
            } else {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimt<DTYPE_Y, uint32_t, int32_t, FORMAT_NHWC, false> op;
                op.Init(grads, y, tilingData);
                op.Process();
            }
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_IDX64)) {
        GET_TILING_DATA_WITH_STRUCT(ResizeBicubicV2GradSimtTilingData, tilingDataIn, tiling);
        const ResizeBicubicV2GradSimtTilingData *__restrict__ tilingData = &tilingDataIn;
        if (tilingData->format == 0) {
            if (tilingData->alignCorners > 0) {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimt<DTYPE_Y, uint64_t, int64_t, FORMAT_NCHW, true> op;
                op.Init(grads, y, tilingData);
                op.Process();
            } else {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimt<DTYPE_Y, uint64_t, int64_t, FORMAT_NCHW, false> op;
                op.Init(grads, y, tilingData);
                op.Process();
            }
        } else {
            if (tilingData->alignCorners > 0) {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimt<DTYPE_Y, uint64_t, int64_t, FORMAT_NHWC, true> op;
                op.Init(grads, y, tilingData);
                op.Process();
            } else {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimt<DTYPE_Y, uint64_t, int64_t, FORMAT_NHWC, false> op;
                op.Init(grads, y, tilingData);
                op.Process();
            }
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_DETERMINE)) {
        GET_TILING_DATA_WITH_STRUCT(ResizeBicubicV2GradSimtDetermineTilingData, tilingDataIn, tiling);
        const ResizeBicubicV2GradSimtDetermineTilingData *__restrict__ tilingData = &tilingDataIn;
        if (tilingData->format == 0) {
            if (tilingData->alignCorners > 0) {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimtDetermine<DTYPE_Y, uint32_t, int32_t, FORMAT_NCHW, true> op;
                op.Init(grads, y, tilingData);
                op.Process();
            } else {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimtDetermine<DTYPE_Y, uint32_t, int32_t, FORMAT_NCHW, false>
                    op;
                op.Init(grads, y, tilingData);
                op.Process();
            }
        } else {
            if (tilingData->alignCorners > 0) {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimtDetermine<DTYPE_Y, uint32_t, int32_t, FORMAT_NHWC, true> op;
                op.Init(grads, y, tilingData);
                op.Process();
            } else {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimtDetermine<DTYPE_Y, uint32_t, int32_t, FORMAT_NHWC, false>
                    op;
                op.Init(grads, y, tilingData);
                op.Process();
            }
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_DETERMINE_IDX64)) {
        GET_TILING_DATA_WITH_STRUCT(ResizeBicubicV2GradSimtDetermineTilingData, tilingDataIn, tiling);
        const ResizeBicubicV2GradSimtDetermineTilingData *__restrict__ tilingData = &tilingDataIn;
        if (tilingData->format == 0) {
            if (tilingData->alignCorners > 0) {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimtDetermine<DTYPE_Y, uint64_t, int64_t, FORMAT_NCHW, true> op;
                op.Init(grads, y, tilingData);
                op.Process();
            } else {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimtDetermine<DTYPE_Y, uint64_t, int64_t, FORMAT_NCHW, false>
                    op;
                op.Init(grads, y, tilingData);
                op.Process();
            }
        } else {
            if (tilingData->alignCorners > 0) {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimtDetermine<DTYPE_Y, uint64_t, int64_t, FORMAT_NHWC, true> op;
                op.Init(grads, y, tilingData);
                op.Process();
            } else {
                ResizeBicubicV2Grad::ResizeBicubicV2GradSimtDetermine<DTYPE_Y, uint64_t, int64_t, FORMAT_NHWC, false>
                    op;
                op.Init(grads, y, tilingData);
                op.Process();
            }
        }
        return;
    }
}