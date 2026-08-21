/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECK_VALID_TILING_DATA_H_
#define CHECK_VALID_TILING_DATA_H_

#include <cstdint>

struct CheckValidTilingData {
    int64_t N;
    int64_t tile_n;
    int64_t tile_n_tail;
    int64_t num_tiles;
    int64_t num_cores;
    int64_t tiles_main;
    int64_t cores_tail;
    int64_t per_buf_bytes;
    float img_width_x;
    float img_height_y;
};

#endif // CHECK_VALID_TILING_DATA_H_
