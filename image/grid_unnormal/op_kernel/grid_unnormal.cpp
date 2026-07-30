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
 * \file grid_unnormal.cpp
 * \brief GridUnnormal 算子 kernel 入口（arch35，ops-cv 非模板 extern "C" 约定）。
 *
 * dtype 由 DTYPE_GRID 编译期实例化（grid 输入名为 grid → 框架自动定义 DTYPE_GRID，
 * fp16 / fp32 各一份）；align_corners 走 tilingdata 运行时分支。
 */
#include "arch35/grid_unnormal.h"

#ifndef __CCE_KT_TEST__
extern "C" __global__ __aicore__ void grid_unnormal(GM_ADDR grid, GM_ADDR assist, GM_ADDR diff, GM_ADDR position,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GridUnnormalTilingData);
    GET_TILING_DATA_WITH_STRUCT(GridUnnormalTilingData, tilingData, tiling);

    NsGridUnnormal::GridUnnormalKernel<DTYPE_GRID> op;
    op.Init(grid, assist, diff, position, &tilingData);
    op.Process();
    (void)workspace;
}
#endif // __CCE_KT_TEST__
