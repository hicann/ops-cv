/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "check_valid_tiling_host.h"

namespace optiling {

uint32_t ComputePublicTiling(const CheckValidPublicInputs& in, CheckValidPublicResult& out)
{
    bool bbox_dtype_ok = (in.bbox_dtype == CV_DTYPE_FLOAT || in.bbox_dtype == CV_DTYPE_FLOAT16);
    bool dtype_match = (in.bbox_dtype == in.img_metas_dtype);
    if (!bbox_dtype_ok || !dtype_match) {
        return CV_STATUS_FAILED;
    }

    bool format_ok = (in.bbox_format == CV_FMT_ND && in.img_metas_format == CV_FMT_ND && in.out_format == CV_FMT_ND);
    if (!format_ok) {
        return CV_STATUS_FAILED;
    }

    if (in.bbox_rank != 2) {
        return CV_STATUS_FAILED;
    }

    if (in.bbox_cols != 4) {
        return CV_STATUS_FAILED;
    }

    if (in.img_metas_numel < 3) {
        return CV_STATUS_FAILED;
    }

    float img_width_x = in.W * in.r - 1.0f;
    float img_height_y = in.H * in.r - 1.0f;

    out.td.N = 0;
    out.td.tile_n = 0;
    out.td.tile_n_tail = 0;
    out.td.num_tiles = 0;
    out.td.num_cores = 0;
    out.td.tiles_main = 0;
    out.td.cores_tail = 0;
    out.td.per_buf_bytes = 0;

    if (in.bbox_rows == 0) {
        out.tiling_key = CV_KEY_EMPTY;
        out.block_dim = 1;
        out.td.N = 0;
        out.td.img_width_x = img_width_x;
        out.td.img_height_y = img_height_y;
    } else {
        out.tiling_key = CV_KEY_NORMAL;
        out.block_dim = 0;
        out.td.N = in.bbox_rows;
        out.td.img_width_x = img_width_x;
        out.td.img_height_y = img_height_y;
    }

    return CV_STATUS_SUCCESS;
}

uint32_t ComputeBranch0Tiling(const CheckValidBranch0Inputs& in, CheckValidBranch0Result& out)
{
    if (in.N <= 0) {
        return CV_STATUS_FAILED;
    }
    int64_t sizeofT;
    if (in.dtype == CV_DTYPE_FLOAT16) {
        sizeofT = 2;
    } else if (in.dtype == CV_DTYPE_FLOAT) {
        sizeofT = 4;
    } else {
        return CV_STATUS_FAILED;
    }

    constexpr int64_t P = 3;
    constexpr int64_t COLS = 4;

    int64_t per_buf_bytes = (in.ub_available / P) & ~31;

    int64_t per_buf_elems = per_buf_bytes / sizeofT;
    int64_t raw_tile_n = per_buf_elems / COLS;

    int64_t rowAlignMask = (sizeofT == 2) ? ~15 : ~7;
    int64_t tile_n = raw_tile_n & rowAlignMask;

    if (tile_n <= 0) {
        tile_n = (sizeofT == 2) ? 16 : 8;
    }

    int64_t num_tiles = (in.N + tile_n - 1) / tile_n;
    int64_t tile_n_tail = in.N - (num_tiles - 1) * tile_n;
    if (tile_n_tail == 0 && num_tiles > 0) {
        tile_n_tail = tile_n;
    }

    int64_t num_cores = (in.available_cores < num_tiles) ? in.available_cores : num_tiles;
    if (num_cores < 1) {
        num_cores = 1;
    }
    int64_t tiles_main = num_tiles / num_cores;
    int64_t cores_tail = num_tiles % num_cores;

    out.block_dim = static_cast<uint32_t>(num_cores);
    out.tile_n = tile_n;
    out.tile_n_tail = tile_n_tail;
    out.num_tiles = num_tiles;
    out.num_cores = num_cores;
    out.tiles_main = tiles_main;
    out.cores_tail = cores_tail;
    out.per_buf_bytes = per_buf_bytes;

    return CV_STATUS_SUCCESS;
}

uint32_t ComputeBranch1Tiling(const CheckValidBranch1Inputs& in, CheckValidBranch1Result& out)
{
    if (in.N != 0) {
        return CV_STATUS_FAILED;
    }
    if (in.dtype != CV_DTYPE_FLOAT16 && in.dtype != CV_DTYPE_FLOAT) {
        return CV_STATUS_FAILED;
    }

    out.tile_n = 0;
    out.tile_n_tail = 0;
    out.num_tiles = 0;
    out.num_cores = 0;
    out.tiles_main = 0;
    out.cores_tail = 0;
    out.per_buf_bytes = 0;

    float img_width_x = in.W * in.r - 1.0f;
    float img_height_y = in.H * in.r - 1.0f;

    out.tiling_key = CV_KEY_EMPTY;
    out.block_dim = 1;
    out.N = 0;
    out.img_width_x = img_width_x;
    out.img_height_y = img_height_y;

    return CV_STATUS_SUCCESS;
}

} // namespace optiling
