/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/combined_non_max_suppression_simt.h"

extern "C" __global__ __aicore__ void combined_non_max_suppression(GM_ADDR boxes, GM_ADDR scores,
                                                                   GM_ADDR maxOutputSizePerClass, GM_ADDR maxTotalSize,
                                                                   GM_ADDR iouThreshold, GM_ADDR scoreThreshold,
                                                                   GM_ADDR nmsedBoxes, GM_ADDR nmsedScores,
                                                                   GM_ADDR nmsedClasses, GM_ADDR validDetections,
                                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    (void)maxOutputSizePerClass;
    (void)maxTotalSize;
    (void)iouThreshold;
    (void)scoreThreshold;
    if (workspace == nullptr) {
        return;
    }
    AscendC::SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(CombinedNonMaxSuppressionTilingData);
    GET_TILING_DATA_WITH_STRUCT(CombinedNonMaxSuppressionTilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    CombinedNonMaxSuppressionOps::CombinedNonMaxSuppressionKernel op;
    op.Init(boxes, scores, nmsedBoxes, nmsedScores, nmsedClasses, validDetections, userWorkspace, &tilingData, &pipe);
    op.Process();
}
