/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_bilinear_v2_grad.cpp
 * \brief resize_bilinear_v2_grad
 */
#include "./arch35/resize_bilinear_v2_grad_base.h"
#include "./arch35/resize_bilinear_v2_grad_all_copy.h"
#include "./arch35/resize_bilinear_v2_grad_point_copy.h"
#include "./arch35/resize_bilinear_v2_grad_c_parallel.h"
#include "./arch35/resize_bilinear_v2_grad_simt.h"
#include "./arch35/resize_bilinear_v2_grad_simt_determine.h"
#include "./arch35/resize_bilinear_v2_grad_simt_determine_scales.h"

#define TILING_KEY_SIMT_NCHW 10000
#define TILING_KEY_SIMT_NHWC 10001
#define TILING_KEY_SIMT_NCHW_DETERMINE 10002
#define TILING_KEY_SIMT_NCHW_DETERMINE_SCALES 10012
#define TILING_KEY_SIMT_NHWC_DETERMINE 10003
#define TILING_KEY_SIMT_NHWC_DETERMINE_SCALES 10013
#define TILING_KEY_SIMT_NCHW_IDX64 10004
#define TILING_KEY_SIMT_NHWC_IDX64 10005
#define TILING_KEY_SIMT_NCHW_DETERMINE_IDX64 10006
#define TILING_KEY_SIMT_NCHW_DETERMINE_SCALES_IDX64 10016
#define TILING_KEY_SIMT_NHWC_DETERMINE_IDX64 10007
#define TILING_KEY_SIMT_NHWC_DETERMINE_SCALES_IDX64 10017
#define TILING_KEY_C_PARALLEL 20000
#define TILING_KEY_ALL_COPY 30000
#define TILING_KEY_POINT_COPY 30001

using namespace AscendC;

extern "C" __global__ __aicore__ void resize_bilinear_v2_grad(
    GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_ALL_COPY)) {
        ResizeBilinearV2Grad::ResizeBilinearV2GradAllCopy<DTYPE_GRADS, DTYPE_Y> op;
        op.Init(grads, originalImage, y, &pipe, &tilingData);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_POINT_COPY)) {
        ResizeBilinearV2Grad::ResizeBilinearV2GradPointCopy<DTYPE_GRADS, DTYPE_Y> op;
        op.Init(grads, originalImage, y, &pipe, &tilingData);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_C_PARALLEL)) {
        ResizeBilinearV2Grad::ResizeBilinearV2GradNc<DTYPE_GRADS, DTYPE_Y> op;
        op.Init(grads, originalImage, y, &pipe, &tilingData);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NCHW)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2GradSimt::ResizeBilinearV2GradSimt<DTYPE_GRADS, DTYPE_Y, true, uint32_t, FORMAT_NCHW> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2GradSimt::ResizeBilinearV2GradSimt<DTYPE_GRADS, DTYPE_Y, false, uint32_t, FORMAT_NCHW> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NHWC)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2GradSimt::ResizeBilinearV2GradSimt<DTYPE_GRADS, DTYPE_Y, true, uint32_t, FORMAT_NHWC> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2GradSimt::ResizeBilinearV2GradSimt<DTYPE_GRADS, DTYPE_Y, false, uint32_t, FORMAT_NHWC> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NCHW_IDX64)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2GradSimt::ResizeBilinearV2GradSimt<DTYPE_GRADS, DTYPE_Y, true, uint64_t, FORMAT_NCHW> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2GradSimt::ResizeBilinearV2GradSimt<DTYPE_GRADS, DTYPE_Y, false, uint64_t, FORMAT_NCHW> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NHWC_IDX64)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2GradSimt::ResizeBilinearV2GradSimt<DTYPE_GRADS, DTYPE_Y, true, uint64_t, FORMAT_NHWC> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2GradSimt::ResizeBilinearV2GradSimt<DTYPE_GRADS, DTYPE_Y, false, uint64_t, FORMAT_NHWC> op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NCHW_DETERMINE)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2Grad::ResizeBilinearV2GradSimtDetermine<DTYPE_GRADS, DTYPE_Y, true, uint32_t, FORMAT_NCHW>
                op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2Grad::ResizeBilinearV2GradSimtDetermine<DTYPE_GRADS, DTYPE_Y, false, uint32_t, FORMAT_NCHW>
                op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NHWC_DETERMINE)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2Grad::ResizeBilinearV2GradSimtDetermine<DTYPE_GRADS, DTYPE_Y, true, uint32_t, FORMAT_NHWC>
                op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2Grad::ResizeBilinearV2GradSimtDetermine<DTYPE_GRADS, DTYPE_Y, false, uint32_t, FORMAT_NHWC>
                op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NCHW_DETERMINE_SCALES)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2Grad::
                ResizeBilinearV2GradSimtDetermineScales<DTYPE_GRADS, DTYPE_Y, true, uint32_t, FORMAT_NCHW>
                    op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2Grad::
                ResizeBilinearV2GradSimtDetermineScales<DTYPE_GRADS, DTYPE_Y, false, uint32_t, FORMAT_NCHW>
                    op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NHWC_DETERMINE_SCALES)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2Grad::
                ResizeBilinearV2GradSimtDetermineScales<DTYPE_GRADS, DTYPE_Y, true, uint32_t, FORMAT_NHWC>
                    op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2Grad::
                ResizeBilinearV2GradSimtDetermineScales<DTYPE_GRADS, DTYPE_Y, false, uint32_t, FORMAT_NHWC>
                    op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NCHW_DETERMINE_IDX64)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2Grad::ResizeBilinearV2GradSimtDetermine<DTYPE_GRADS, DTYPE_Y, true, uint64_t, FORMAT_NCHW>
                op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2Grad::ResizeBilinearV2GradSimtDetermine<DTYPE_GRADS, DTYPE_Y, false, uint64_t, FORMAT_NCHW>
                op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NCHW_DETERMINE_SCALES_IDX64)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2Grad::
                ResizeBilinearV2GradSimtDetermineScales<DTYPE_GRADS, DTYPE_Y, true, uint64_t, FORMAT_NCHW>
                    op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2Grad::
                ResizeBilinearV2GradSimtDetermineScales<DTYPE_GRADS, DTYPE_Y, false, uint64_t, FORMAT_NCHW>
                    op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NHWC_DETERMINE_IDX64)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2Grad::ResizeBilinearV2GradSimtDetermine<DTYPE_GRADS, DTYPE_Y, true, uint64_t, FORMAT_NHWC>
                op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2Grad::ResizeBilinearV2GradSimtDetermine<DTYPE_GRADS, DTYPE_Y, false, uint64_t, FORMAT_NHWC>
                op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NHWC_DETERMINE_SCALES_IDX64)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2Grad::
                ResizeBilinearV2GradSimtDetermineScales<DTYPE_GRADS, DTYPE_Y, true, uint64_t, FORMAT_NHWC>
                    op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2Grad::
                ResizeBilinearV2GradSimtDetermineScales<DTYPE_GRADS, DTYPE_Y, false, uint64_t, FORMAT_NHWC>
                    op;
            op.Init(grads, originalImage, y, &tilingData);
            op.Process();
        }
        return;
    }
}