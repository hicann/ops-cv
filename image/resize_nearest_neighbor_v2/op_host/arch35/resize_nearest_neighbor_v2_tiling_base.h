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
 * \file resize_nearest_neighbor_v2_tiling.h
 * \brief resize_nearest_neighbor_v2_tiling
 */
#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_RESIZE_NEAREST_NEIGHBOR_V2_ASCENDC_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_RESIZE_NEAREST_NEIGHBOR_V2_ASCENDC_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ResizeNearestNeighborV2TilingData)
 TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
 TILING_DATA_FIELD_DEF(int64_t, ubSize);
 TILING_DATA_FIELD_DEF(int64_t, alignCorners);
 TILING_DATA_FIELD_DEF(int64_t, halfPixelCenters);
 TILING_DATA_FIELD_DEF(int64_t, lenN);
 TILING_DATA_FIELD_DEF(int64_t, lenC);
 TILING_DATA_FIELD_DEF(int64_t, lenSrcH);
 TILING_DATA_FIELD_DEF(int64_t, lenSrcW);
 TILING_DATA_FIELD_DEF(int64_t, lenDesH);
 TILING_DATA_FIELD_DEF(int64_t, lenDesW);
 TILING_DATA_FIELD_DEF(int64_t, condition);
 TILING_DATA_FIELD_DEF(int64_t, switchParams);
 TILING_DATA_FIELD_DEF(int64_t, splitBlockFactor);
 TILING_DATA_FIELD_DEF(int64_t, splitBlockTailFactor);
 TILING_DATA_FIELD_DEF(int64_t, lenCAlign);
 TILING_DATA_FIELD_DEF(int64_t, hwcNum);
 TILING_DATA_FIELD_DEF(int64_t, dstHwcNum);
 TILING_DATA_FIELD_DEF(int64_t, wcNum);
 TILING_DATA_FIELD_DEF(int64_t, dstWcNum);
 TILING_DATA_FIELD_DEF(int64_t, nLoop);
 TILING_DATA_FIELD_DEF(int64_t, nLoopTimesBefore);
 TILING_DATA_FIELD_DEF(int64_t, nLoopTimesLast);
 TILING_DATA_FIELD_DEF(int64_t, nLoopTailLast);
 TILING_DATA_FIELD_DEF(int64_t, wcLoop);
 TILING_DATA_FIELD_DEF(int64_t, wcLoopTimesBefore);
 TILING_DATA_FIELD_DEF(int64_t, wcLoopTailBefore);
 TILING_DATA_FIELD_DEF(int64_t, wcLoopTimesLast);
 TILING_DATA_FIELD_DEF(int64_t, wcLoopTailLast);
 TILING_DATA_FIELD_DEF(int64_t, splitBlockFullCount);
 TILING_DATA_FIELD_DEF(int64_t, splitFactorDesH);
 TILING_DATA_FIELD_DEF(int64_t, splitFactorTailDesH);
 TILING_DATA_FIELD_DEF(int64_t, splitCountDesH);
 TILING_DATA_FIELD_DEF(int64_t, splitFactorDesW);
 TILING_DATA_FIELD_DEF(int64_t, splitFactorTailDesW);
 TILING_DATA_FIELD_DEF(int64_t, splitCountDesW);
 TILING_DATA_FIELD_DEF(float, scaleW);
 TILING_DATA_FIELD_DEF(float, scaleH);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResizeNearestNeighborV2, ResizeNearestNeighborV2TilingData)


struct ResizeNearestNeighborV2CompileInfo {
    int32_t core_num = 0;
    int32_t ubSize = 0;
};

ge::graphStatus Tiling4ResizeNearestNeighborV2ForAscendC(gert::TilingContext* context,
                                                         const ResizeNearestNeighborV2CompileInfo* compileInfo);
} // namespace optiling

#endif  // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_RESIZE_NEAREST_NEIGHBOR_V2_ASCENDC_H_