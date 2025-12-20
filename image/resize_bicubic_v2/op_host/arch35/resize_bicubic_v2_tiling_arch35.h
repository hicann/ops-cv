/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file resize_bicubic_v2_tiling_arch35.h
 * \brief resize_bicubic_v2_tiling_arch35
 */
#ifndef OPS_IMAGE_RESIZE_BICUBIC_V2_OP_HOST_RESIZE_BICUBIC_V2_TILING_ARCH35_H_
#define OPS_IMAGE_RESIZE_BICUBIC_V2_OP_HOST_RESIZE_BICUBIC_V2_TILING_ARCH35_H_

#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ResizeBicubicV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blkProcessNum);
TILING_DATA_FIELD_DEF(int64_t, splitBlockTailFactor);
TILING_DATA_FIELD_DEF(int64_t, lenSrcH);
TILING_DATA_FIELD_DEF(int64_t, lenSrcW);
TILING_DATA_FIELD_DEF(int64_t, lenDesH);
TILING_DATA_FIELD_DEF(int64_t, lenDesW);
TILING_DATA_FIELD_DEF(int64_t, lenC);
TILING_DATA_FIELD_DEF(int64_t, lenN);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(int64_t, nFactor);
TILING_DATA_FIELD_DEF(int64_t, hFactor);
TILING_DATA_FIELD_DEF(int64_t, wFactor);
TILING_DATA_FIELD_DEF(int64_t, cFactor);
TILING_DATA_FIELD_DEF(int64_t, ubNFactor);
TILING_DATA_FIELD_DEF(int64_t, ubHFactor);
TILING_DATA_FIELD_DEF(int64_t, ubWFactor);
TILING_DATA_FIELD_DEF(int64_t, ubCFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResizeBicubicV2, ResizeBicubicV2TilingData)
struct ResizeBicubicV2CompileInfo {
    int32_t totalCoreNum = 0;
    int32_t totalUbSize = 0;
};
} // namespace optiling

#endif // OPS_IMAGE_RESIZE_BICUBIC_V2_OP_HOST_RESIZE_BICUBIC_V2_TILING_ARCH35_H_
