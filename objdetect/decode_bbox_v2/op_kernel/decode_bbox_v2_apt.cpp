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
#include "arch35/decode_bbox_v2_struct.h"
#include "arch35/decode_bbox_v2_tiling_struct.h"
#include "arch35/decode_bbox_v2_kernel.h"

template <int LAYOUT>
__global__ __aicore__ void decode_bbox_v2(GM_ADDR boxes, GM_ADDR anchors, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR ins[kMaxInputSlots] = {boxes, anchors};
    GM_ADDR outs[kMaxOutputSlots] = {y};

    REGISTER_TILING_DEFAULT(DecodeBboxV2TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GET_TILING_DATA_WITH_STRUCT(DecodeBboxV2TilingData, td, tiling);

    if (td.ubFormer > 0) {
        DecodeBboxV2Kernel<DTYPE_BOXES, LAYOUT> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
