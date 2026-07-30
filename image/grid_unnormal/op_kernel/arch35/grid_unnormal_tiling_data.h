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
 * \file grid_unnormal_tiling_data.h
 * \brief GridUnnormal TilingData 结构体（纯 elementwise，按总元素数扁平分核）
 *
 * grid/assist/diff/position 同 shape，逐元素计算：
 *   t = (grid + 1) * 0.5
 *   pos_base = align_corners ? t * (assist - 1) : t * assist - 0.5
 *   position = floor(pos_base)   (int32)
 *   diff     = pos_base - floor(pos_base)
 * 分核策略：把总元素数 totalNum 均分到各 AIV 核（perCoreNum 向上取整），
 * 每核内再按 ubFactor 元素做 UB 切分循环。align_corners 走 tilingdata 运行时分支。
 */
#ifndef GRID_UNNORMAL_TILING_DATA_H_
#define GRID_UNNORMAL_TILING_DATA_H_

#include <cstdint>

struct GridUnnormalTilingData {
    int64_t totalNum = 0;     // 总元素数 = prod(grid.shape)
    int64_t perCoreNum = 0;   // 主核每核元素数（向上取整分核）
    int64_t ubFactor = 0;     // 单次 UB tile 元素数（64 元素寄存器读宽对齐）
    int32_t alignCorners = 0; // 0/1，坐标反归一化公式分支
};

#endif // GRID_UNNORMAL_TILING_DATA_H_
