/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/batch_multi_class_non_max_suppression_kernel.h"
#include "arch35/batch_multi_class_non_max_suppression_tiling_key.h"

extern "C" __global__ __aicore__ void batch_multi_class_non_max_suppression(
    GM_ADDR boxes, GM_ADDR scores, GM_ADDR clip_window, GM_ADDR num_valid_boxes, GM_ADDR nmsed_boxes,
    GM_ADDR nmsed_scores, GM_ADDR nmsed_classes, GM_ADDR nmsed_num, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (workspace == nullptr) {
        return;
    }
    AscendC::SetSysWorkspace(workspace);
    if (AscendC::GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(BatchMultiClassNonMaxSuppressionTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchMultiClassNonMaxSuppressionTilingData, tilingData, tiling);
    if (TILING_KEY_IS(BATCH_MULTI_CLASS_NMS_TILING_KEY)) {
        BatchMultiClassNonMaxSuppressionOp::BatchMultiClassNonMaxSuppressionKernel<DTYPE_BOXES> op;
        op.Init(boxes, scores, clip_window, num_valid_boxes, nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num,
                workspace, &tilingData);
        op.Process();
    }
}
