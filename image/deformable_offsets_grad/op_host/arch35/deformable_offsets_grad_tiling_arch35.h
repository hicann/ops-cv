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
 * \file deformable_offsets_grad_tiling_arch35.h
 * \brief deformable_offsets_grad_tiling_arch35 info
 */
#ifndef DEFORMABLE_OFFSETS_GRAD_TILING_ARCH35_H
#define DEFORMABLE_OFFSETS_GRAD_TILING_ARCH35_H

#include <cstdint>
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeformableOffsetsGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, realCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, clearGradXCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, clearGradOffsetsCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, strideHeight);
TILING_DATA_FIELD_DEF(uint32_t, strideWidth);
TILING_DATA_FIELD_DEF(uint32_t, dilationHeight);
TILING_DATA_FIELD_DEF(uint32_t, dilationWidth);
TILING_DATA_FIELD_DEF(uint32_t, padsHeight);
TILING_DATA_FIELD_DEF(uint32_t, padsWidth);
TILING_DATA_FIELD_DEF(uint32_t, dimKHeight);
TILING_DATA_FIELD_DEF(uint32_t, dimKWidth);
TILING_DATA_FIELD_DEF(uint32_t, imgBatchNum);
TILING_DATA_FIELD_DEF(uint32_t, imgChannel);
TILING_DATA_FIELD_DEF(uint32_t, imgWidth);
TILING_DATA_FIELD_DEF(uint32_t, imgHeight);
TILING_DATA_FIELD_DEF(uint32_t, imgOutHeight);
TILING_DATA_FIELD_DEF(uint32_t, imgOutWidth);
TILING_DATA_FIELD_DEF(uint32_t, deformableGroups);
TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
TILING_DATA_FIELD_DEF(uint32_t, blockFactorTail);
TILING_DATA_FIELD_DEF(uint32_t, gradXFactor);
TILING_DATA_FIELD_DEF(uint32_t, gradXFactorTail);
TILING_DATA_FIELD_DEF(uint32_t, gradOffsetsFactor);
TILING_DATA_FIELD_DEF(uint32_t, gradOffsetsFactorTail);
TILING_DATA_FIELD_DEF(uint32_t, outputGradXSize);
TILING_DATA_FIELD_DEF(uint32_t, outputGradOffsetsSize);
END_TILING_DATA_DEF;

struct DeformableOffsetsGradOffset {
    uint32_t gradOffsetsBatchNum;
    uint32_t gradOffsetsImgOutHeight;
    uint32_t gradOffsetsImgOutWidth;
};

struct TilingPrepareForDeformableOffsetsGradCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

REGISTER_TILING_DATA_CLASS(DeformableOffsetsGrad, DeformableOffsetsGradTilingData)

ge::graphStatus DeformableOffsetsGradTilingForAscendC(gert::TilingContext* context_, const int64_t coreNum);
} // namespace optiling
#endif