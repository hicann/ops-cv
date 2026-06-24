/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file blend_face_bg_part_two.cpp
 * \brief
 */
#include "blend_face_bg_part_two.h"

template <uint32_t schMode>
__global__ __aicore__ void blend_face_bg_part_two(GM_ADDR acc_face, GM_ADDR acc_mask, GM_ADDR max_mask, GM_ADDR bg_img,
                                                  GM_ADDR fused_img, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(BlendFaceBgPartTwoTilingData);
    GET_TILING_DATA_WITH_STRUCT(BlendFaceBgPartTwoTilingData, tilingData, tiling);

    if constexpr (schMode == BLEND_FACE_BG_PART_TWO_SCH_MODE_DEFAULT) {
        AscendC::TPipe pipe;
        NsBlendFaceBgPartTwo::KernelBlendFaceBgPartTwo<DTYPE_ACC_FACE, DTYPE_BG_IMG> op;
        op.Init(acc_face, acc_mask, max_mask, bg_img, fused_img, &tilingData, &pipe);
        op.Process();
    }
}
