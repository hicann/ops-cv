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
 * \file upsample_nearest3d_tiling_data.h
 * \brief upsample_nearest3d_tiling_data.h
 */

#ifndef _UPSAMPLE_NEAREST3D_REGBASE_TILING_DATA_H_
#define _UPSAMPLE_NEAREST3D_REGBASE_TILING_DATA_H_

struct UpsampleNearest3dRegBaseTilingData {
    int64_t blkProcessNum;
    int64_t lenN;
    int64_t lenC;
    int64_t inD;
    int64_t inH;
    int64_t inW;
    int64_t outD;
    int64_t outH;
    int64_t outW;
    int32_t ubFactor;
    int32_t tailBlockNum;
    float scaleD;
    float scaleH;
    float scaleW;
};

struct UpsampleNearest3dRegBaseSimdTilingData {
    int64_t lenN;
    int64_t lenC;
    int64_t inD;
    int64_t inH;
    int64_t inW;
    int64_t outD;
    int64_t outH;
    int64_t outW;
    float scaleD;
    float scaleH;
    float scaleW;
    int64_t slideSizeW;
    int64_t tensorSizeW;
    int64_t tensorSizeH;
    int64_t tensorSizeD;
    int64_t slideNumH;
    int64_t slideNumD;
    int64_t eachCoreSlideNum;
    int64_t remainder;
    int64_t tailStartSlideNum;
    int64_t groupCoreNum;
    int64_t inputRow;
    int64_t tailAvergingRow;
    int32_t realCoreNum;
    bool isView1DAndSmallW;
};
#endif
