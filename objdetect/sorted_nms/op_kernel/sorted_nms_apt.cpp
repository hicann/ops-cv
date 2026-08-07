/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "./arch35/sorted_nms_simt.h"

extern "C" __global__ __aicore__ void sorted_nms(GM_ADDR boxes, GM_ADDR sorted_scores, GM_ADDR input_indices,
                                                 GM_ADDR max_output_size, GM_ADDR iou_threshold,
                                                 GM_ADDR score_threshold, GM_ADDR selected_indices, GM_ADDR shape_out,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (workspace == nullptr) {
        return;
    }
    AscendC::SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(SortedNMSTilingData);
    GET_TILING_DATA_WITH_STRUCT(SortedNMSTilingData, tilingData, tiling);

    AscendC::TPipe pipe;
    NsSortedNMS::Process<DTYPE_BOXES, DTYPE_SORTED_SCORES, DTYPE_IOU_THRESHOLD, DTYPE_SCORE_THRESHOLD>(
        boxes, sorted_scores, input_indices, max_output_size, iou_threshold, score_threshold, selected_indices,
        shape_out, userWorkspace, &tilingData, &pipe);
}
