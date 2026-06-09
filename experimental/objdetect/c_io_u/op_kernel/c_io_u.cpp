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
 * \file c_io_u.cpp
 * \brief CIoU kernel entry.
 */

#include "c_io_u.h"

template <uint32_t schMode>
__global__ __aicore__ void c_io_u(GM_ADDR bboxes, GM_ADDR gtboxes,
                                GM_ADDR overlap, GM_ADDR atan_sub,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CIoUTilingData);
    GET_TILING_DATA_WITH_STRUCT(CIoUTilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    NsCIoU::CIoU<DTYPE_BBOXES> op;
    op.Init(bboxes, gtboxes, overlap, atan_sub, &tilingData, &pipe);
    op.Process();
}
