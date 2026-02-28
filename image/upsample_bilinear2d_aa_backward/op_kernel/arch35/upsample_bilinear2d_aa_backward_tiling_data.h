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
 * \file upsample_bilinear2d_aa_backward_tiling_data.h
 * \brief
 */

#ifndef UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_DATA_H
#define UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_DATA_H

struct UpsampleBilinear2dAABackwardRegBaseTilingData {
    int64_t blkProcessNum = 0;
    int64_t realCoreNum = 0;
    int64_t tailBlockNum = 0;
    int64_t initBlkProcessNum = 0;
    int64_t initRealCoreNum = 0;
    int64_t initTailBlockNum = 0;
    int64_t lenN;
    int64_t lenC;
    int64_t inH;
    int64_t inW;
    int64_t outH;
    int64_t outW;
    int64_t maxInterpSizeH;
    int64_t maxInterpSizeW;
    int32_t ubFactor;
    float scaleH;
    float scaleW;
    float invScaleH;
    float invScaleW;
    float supportH;
    float supportW;
};

#endif // UPSAMPLE_BILINEAR2D_AA_BACKWARD_TILING_DATA_H
