/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_bilinear_v2_tiling.h
 * \brief resize_bilinear_v2_tiling
 */
#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_RESIZE_BILINEAR_V2_ASCENDC_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_RESIZE_BILINEAR_V2_ASCENDC_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ResizeBilinearV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
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
TILING_DATA_FIELD_DEF(int64_t, nFactor);
TILING_DATA_FIELD_DEF(int64_t, hFactor);
TILING_DATA_FIELD_DEF(int64_t, wFactor);
TILING_DATA_FIELD_DEF(int64_t, cFactor);
TILING_DATA_FIELD_DEF(int64_t, hwFactor);
TILING_DATA_FIELD_DEF(int64_t, ubNFactor);
TILING_DATA_FIELD_DEF(int64_t, ubHFactor);
TILING_DATA_FIELD_DEF(int64_t, ubWFactor);
TILING_DATA_FIELD_DEF(int64_t, ubCFactor);
TILING_DATA_FIELD_DEF(int64_t, splitBlockFactor);     // for simt
TILING_DATA_FIELD_DEF(int64_t, splitBlockTailFactor); // for simt
TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(int64_t, ubHWFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResizeBilinearV2, ResizeBilinearV2TilingData)

struct ResizeBilinearV2CompileInfo {
    int32_t core_num = 0;
    int32_t ubSize = 0;
};
} // namespace optiling

#endif // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_RESIZE_BILINEAR_V2_ASCENDC_H_