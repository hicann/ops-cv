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
 * \file crop_and_resize_constraints.h
 * \brief Shared constraint constants for crop_and_resize operator
 *
 * 被以下文件共同 include，消除跨文件硬编码不一致风险:
 *   - op_host/crop_and_resize_def.cpp         (check_supported 回调)
 *   - op_host/crop_and_resize_tiling_arch35.cpp (tiling 约束检查)
 *   - op_host/crop_and_resize_infershape.cpp   (infershape 约束检查)
 *
 * 注意: BOX_COORDS/CROP_SIZE_LEN/CROP_DIM_MAX 使用 int64_t，与 infershape.cpp 原类型
 * (Shape::GetDim 返回 int64_t) 保持一致，避免隐式转换。tiling_arch35.cpp 原为 int32_t，
 * 替换后隐式提升安全，但 format string 需相应使用 %ld。
 */
#pragma once

#include <cstdint>

// boxes.shape[1] == 4 [y1, x1, y2, x2]
constexpr int64_t BOX_COORDS = 4;
// crop_size.shape == (2,)
constexpr int64_t CROP_SIZE_LEN = 2;
// max(crop_h, crop_w) <= 16
constexpr int64_t CROP_DIM_MAX = 16;
// H*W <= 65530 (FP16)
constexpr int64_t HW_MAX = 65530;
// FP32 时 H*W <= 32765
constexpr int64_t HW_FP32_MAX = 32765;
// crop_h * crop_w <= 32765
constexpr int64_t CROP_AREA_MAX = 32765;
// C (depth) >= 256
constexpr int32_t DEPTH_MIN = 256;
// C (depth) <= 2048
constexpr int32_t DEPTH_MAX = 2048;
// num_boxes > 50
constexpr int32_t NUM_BOXES_MIN = 50;
// num_boxes <= 4000
constexpr int32_t NUM_BOXES_MAX = 4000;

// x 必须为 4D (N, H, W, C)
constexpr int64_t X_DIM = 4;
// boxes 必须为 2D (num_boxes, 4)
constexpr int64_t BOXES_DIM = 2;

// 输入索引（与 def.cpp/proto.h 输入顺序一致）
constexpr int64_t IDX_X = 0;
constexpr int64_t IDX_BOXES = 1;
constexpr int64_t IDX_BOX_INDEX = 2;
constexpr int64_t IDX_CROP_SIZE = 3;
