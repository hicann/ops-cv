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
 * \file resize_nearest_neighbor_v2_grad_tiling.h
 * \brief resize_nearest_neighbor_v2_grad_tiling
 */
 
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TILING_H_
 
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
 
namespace optiling {
constexpr int64_t RESIZE_NEAREST_NEIGHBOR_V2_GRAD_INPUT_DEPENDENCY_IDX = 1;
constexpr int64_t MAX_DIM_NUM = 4;

struct ResizeNearestNeighborV2GradCompileInfo {
    int32_t core_num = 0;
    int32_t ubSize = 0;
};

ge::graphStatus ResizeNearestNeighborV2GradTilingForAscendC(gert::TilingContext* context, const ResizeNearestNeighborV2GradCompileInfo* compileInfo);

BEGIN_TILING_DATA_DEF(ResizeNearestNeighborV2GradTilingData)
    TILING_DATA_FIELD_DEF(int64_t, ubSize);
    TILING_DATA_FIELD_DEF(int64_t, lenN);
    TILING_DATA_FIELD_DEF(int64_t, lenC);
    TILING_DATA_FIELD_DEF(int64_t, lenSrcH);
    TILING_DATA_FIELD_DEF(int64_t, lenSrcW);
    TILING_DATA_FIELD_DEF(int64_t, lenDstH);
    TILING_DATA_FIELD_DEF(int64_t, lenDstW);
    TILING_DATA_FIELD_DEF(int64_t, ubCFactor);
    TILING_DATA_FIELD_DEF(float, scaleH);
    TILING_DATA_FIELD_DEF(float, scaleW);
    TILING_DATA_FIELD_DEF(float, inverseScaleH); // for simt determine
    TILING_DATA_FIELD_DEF(float, inverseScaleW); // for simt determine
    TILING_DATA_FIELD_DEF(int64_t, initYRealCoreNum); // for init y
    TILING_DATA_FIELD_DEF(int64_t, initYSplitBlockFactor); // for init y
    TILING_DATA_FIELD_DEF(int64_t, initYSplitBlockTailFactor); // for init y
    TILING_DATA_FIELD_DEF(int64_t, realCoreNum); // for simt
    TILING_DATA_FIELD_DEF(int64_t, splitBlockFactor); // for simt
    TILING_DATA_FIELD_DEF(int64_t, splitBlockTailFactor); // for simt
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResizeNearestNeighborV2Grad, ResizeNearestNeighborV2GradTilingData)

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TILING_H_