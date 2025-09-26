/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_linear1d_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_LINEAR1D_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_LINEAR1D_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct UpsampleLinear1dCompileInfo {
    uint32_t coreNum;
};

BEGIN_TILING_DATA_DEF(UpsampleLinear1dTilingData)
TILING_DATA_FIELD_DEF(int64_t, mode);
TILING_DATA_FIELD_DEF(bool, align_corners);
TILING_DATA_FIELD_DEF(int64_t, slide_size_w);
TILING_DATA_FIELD_DEF(int64_t, slide_size_h);
TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(float, scale_w);
TILING_DATA_FIELD_DEF(float, scale_h);

TILING_DATA_FIELD_DEF(uint64_t, intermediate_matrix_size);
TILING_DATA_FIELD_DEF(uint32_t, radio_matrix_size_w);
TILING_DATA_FIELD_DEF(uint32_t, radio_matrix_size_h);
TILING_DATA_FIELD_DEF(uint32_t, need_core_num_w);
TILING_DATA_FIELD_DEF(uint32_t, need_core_num_h);

TILING_DATA_FIELD_DEF(int64_t, eachCoreSlideNumW);
TILING_DATA_FIELD_DEF(int64_t, tailStartSlideNumW);
TILING_DATA_FIELD_DEF(int64_t, slideNumW);
TILING_DATA_FIELD_DEF(int64_t, groupCoreNumW);
TILING_DATA_FIELD_DEF(int64_t, tailAvergingRowsW);
TILING_DATA_FIELD_DEF(int64_t, remainderW);

TILING_DATA_FIELD_DEF(int64_t, eachCoreSlideNumH);
TILING_DATA_FIELD_DEF(int64_t, tailStartSlideNumH);
TILING_DATA_FIELD_DEF(int64_t, slideNumH);
TILING_DATA_FIELD_DEF(int64_t, groupCoreNumH);
TILING_DATA_FIELD_DEF(int64_t, tailAvergingRowsH);
TILING_DATA_FIELD_DEF(int64_t, remainderH);

TILING_DATA_FIELD_DEF_ARR(int64_t, 4, input_shapes);
TILING_DATA_FIELD_DEF_ARR(int64_t, 4, output_shapes);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling_w);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling_h);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleLinear1d, UpsampleLinear1dTilingData)
REGISTER_TILING_DATA_CLASS(UpsampleBilinear2d, UpsampleLinear1dTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_LINEAR_1D_TILING_H
