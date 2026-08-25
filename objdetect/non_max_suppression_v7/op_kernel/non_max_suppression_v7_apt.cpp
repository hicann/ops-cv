/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/non_max_suppression_v7_kernel.h"

extern "C" __global__ __aicore__ void non_max_suppression_v7(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_size,
                                                             GM_ADDR iou_threshold, GM_ADDR score_threshold,
                                                             GM_ADDR index_id, GM_ADDR selected_indices,
                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (workspace == nullptr) {
        return;
    }
    AscendC::SetSysWorkspace(workspace);
    if (AscendC::GetUserWorkspace(workspace) == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(NonMaxSuppressionV7TilingData);
    GET_TILING_DATA_WITH_STRUCT(NonMaxSuppressionV7TilingData, tilingData, tiling);
    NonMaxSuppressionV7Op::Kernel<DTYPE_BOXES, DTYPE_SCORES> op;
    op.Init(boxes, scores, max_output_size, iou_threshold, score_threshold, index_id, selected_indices, workspace,
            &tilingData);
    op.Process();
}
