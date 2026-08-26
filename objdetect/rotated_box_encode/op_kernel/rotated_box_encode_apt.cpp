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
 * \file rotated_box_encode_apt.cpp
 * \brief RotatedBoxEncode AscendC kernel entry point (arch35 / Ascend 950)
 */

#include "kernel_operator.h"
#include "arch35/rotated_box_encode_struct.h"
#include "arch35/rotated_box_encode_tiling_data.h"
#include "arch35/rotated_box_encode_kernel.h"

template <int DTYPE>
__global__ __aicore__ void rotated_box_encode(GM_ADDR anchor_box, GM_ADDR gt_box, GM_ADDR y, GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RotatedBoxEncodeTilingData);
    GET_TILING_DATA_WITH_STRUCT(RotatedBoxEncodeTilingData, tilingData, tiling);

    // Empty-tensor short-circuit (DESIGN-BRANCH-0 §0 / §2.6):
    // dim0 == 0 → kernel does nothing; host set coreNum=0, SetBlockDim(0).
    if (tilingData.dim0 == 0) {
        return;
    }

    KernelRotatedBoxEncode<DTYPE> op;
    op.Init(anchor_box, gt_box, y, tilingData);
    op.Process();

    AscendC::PipeBarrier<PIPE_ALL>();
}
