/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_nearest3d_struct.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST3D_STRUCT_H
#define UPSAMPLE_NEAREST3D_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace UpsampleNearest3d {

#define UPSAMPLE_NEAREST3D_TPL_FP16 10
#define UPSAMPLE_NEAREST3D_TPL_FP32 20
#define UPSAMPLE_NEAREST3D_TPL_BF16 30
#define UPSAMPLE_NEAREST3D_TPL_UINT8 40

#ifndef DT_UINT8
#define DT_UINT8 4
#define UPSAMPLE_NEAREST3D_UNDEF_DT_UINT8
#endif

#if defined(ORIG_DTYPE_X) && defined(ORIG_DTYPE_Y) && ORIG_DTYPE_X == DT_UINT8 && ORIG_DTYPE_Y == DT_UINT8
#define UPSAMPLE_NEAREST3D_UINT8_DEVICE
#endif

// Keep the original dtype compilation unit byte-for-byte isolated from UINT8.
// Host compilation and the UINT8 device unit still expose the additional dtype.
#if defined(ORIG_DTYPE_X) && !defined(UPSAMPLE_NEAREST3D_UINT8_DEVICE)
ASCENDC_TPL_ARGS_DECL(UpsampleNearest3d,
                      ASCENDC_TPL_DTYPE_DECL(D_T_X, UPSAMPLE_NEAREST3D_TPL_FP16, UPSAMPLE_NEAREST3D_TPL_BF16,
                                             UPSAMPLE_NEAREST3D_TPL_FP32),
                      ASCENDC_TPL_DTYPE_DECL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_FP16, UPSAMPLE_NEAREST3D_TPL_BF16,
                                             UPSAMPLE_NEAREST3D_TPL_FP32), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_FP16),
                                     ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_FP16), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_BF16),
                                     ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_BF16), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_FP32),
                                     ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_FP32), ));
#else
ASCENDC_TPL_ARGS_DECL(UpsampleNearest3d,
                      ASCENDC_TPL_DTYPE_DECL(D_T_X, UPSAMPLE_NEAREST3D_TPL_FP16, UPSAMPLE_NEAREST3D_TPL_BF16,
                                             UPSAMPLE_NEAREST3D_TPL_FP32, UPSAMPLE_NEAREST3D_TPL_UINT8),
                      ASCENDC_TPL_DTYPE_DECL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_FP16, UPSAMPLE_NEAREST3D_TPL_BF16,
                                             UPSAMPLE_NEAREST3D_TPL_FP32, UPSAMPLE_NEAREST3D_TPL_UINT8), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_FP16),
                                     ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_FP16), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_BF16),
                                     ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_BF16), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_FP32),
                                     ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_FP32), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_UINT8),
                                     ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_UINT8), ));
#endif

#ifdef UPSAMPLE_NEAREST3D_UNDEF_DT_UINT8
#undef DT_UINT8
#undef UPSAMPLE_NEAREST3D_UNDEF_DT_UINT8
#endif

struct UpsampleNearest3dTilingData {
    uint8_t dataType;
    int64_t batches;
    int64_t inputShapes[3];
    int64_t outputShapes[3];
    float scaleW;
    float scaleH;
    float scaleD;
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
    int64_t needCoreNum;
    bool isView1DAndSmallW;
};

} // namespace UpsampleNearest3d
#endif // UPSAMPLE_NEAREST3D_STRUCT_H
