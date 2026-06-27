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
 * \file upsample_nearest2d_grad_apt.cpp
 * \brief Kernel entry for upsample_nearest2d_grad operator
 *
 * Single template parameter:
 *   schMode (uint32_t): scene mode (only MODE_DEFAULT = 0)
 *   dtype is handled by DTYPE_GRAD_OUTPUT macro automatically
 */

#include "./arch35/upsample_nearest2d_grad_simt.h"

template <uint32_t schMode>
__global__ __aicore__ void upsample_nearest2d_grad(GM_ADDR gradOutput, GM_ADDR gradInput, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(UpsampleNearest2dGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(UpsampleNearest2dGradTilingData, tilingData, tiling);

    NsUpsampleNearest2dGrad::Process<DTYPE_GRAD_OUTPUT>(gradOutput, gradInput, &tilingData);
}
