/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch35/bounding_box_decode_tiling_data.h"
#include "arch35/bounding_box_decode_struct.h"
#include "arch35/bounding_box_decode_kernel.h"

template <typename T>
__global__ __aicore__ void bounding_box_decode(GM_ADDR anchor_box, GM_ADDR deltas, GM_ADDR boxes, GM_ADDR workspace,
                                               GM_ADDR tiling)
{
    // §10.1: TilingData registration + task type (pure Vector AIV_ONLY)
    REGISTER_TILING_DEFAULT(BoundingBoxDecodeTilingData);
    GET_TILING_DATA_WITH_STRUCT(BoundingBoxDecodeTilingData, td, tiling);

    // §10.1: instantiate kernel class and run
    BoundingBoxDecodeKernel<T> kernel;
    kernel.Init(anchor_box, deltas, boxes, &td);
    kernel.Process();

    (void)workspace; // workspaceSize=0 (DESIGN §9.6, no cross-core partial merge)
}

// [REF_SAMPLE] --- original sample kernel logic (element-wise add) ---
// The block below is the reference sample implementation from the AddCustom
// template, preserved for traceability.
#if 0
    AscendC::InitSocState();

    REGISTER_TILING_DEFAULT(BoundingBoxDecodeTilingData);

    GET_TILING_DATA(tilingData, tiling);

    KernelBoundingBoxDecode<DTYPE_X> op;

    op.Init(x, y, z, tilingData.totalLength, tilingData.blockLength, tilingData.tileLength);

    op.Process();

    AscendC::PipeBarrier<PIPE_ALL>();
#endif
// [REF_SAMPLE] --- end ---
