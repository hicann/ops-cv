/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_bilinear2d_aa_apt.cpp
 * \brief
 */

#include "./arch35/upsample_bilinear2d_aa_data_copy.h"
#include "./arch35/upsample_bilinear2d_aa_simt.h"
#include "./arch35/upsample_bilinear2d_aa_tiling_key.h"
#include "./arch35/upsample_bilinear2d_aa_tiling_data.h"

using namespace UpsampleBilinear2dAA;

template <uint64_t schId, uint64_t isInt32>
__global__ __aicore__ void upsample_bilinear2d_aa(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(UpsampleBilinear2dAARegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(UpsampleBilinear2dAARegBaseTilingData, tilingData, tiling);

    TPipe pipe;
    if constexpr (schId == 0) {
        Bilinear2dAADataCopy<DTYPE_INPUT> op;
        op.Init(x, y, &pipe, &tilingData);
        op.Process();
        return;
    } else {
        if constexpr(isInt32 == 1) {
            Bilinear2dAASimt<DTYPE_INPUT, uint32_t, int32_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        } else {
            Bilinear2dAASimt<DTYPE_INPUT, uint64_t, int64_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        return;
    }
}
