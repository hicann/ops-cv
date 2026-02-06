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
 * \file resize_nearest_neighbor_v2_grad_tiling_key.h
 * \brief ResizeNearestNeighborV2Grad tiling key declare
 */

 #ifndef CANN_RESIZE_NEIGHBOR_V2_GRAD_TILING_KEY_H
 #define CANN_RESIZE_NEIGHBOR_V2_GRAD_TILING_KEY_H
 #include "ascendc/host_api/tiling/template_argument.h"

#define TPL_SCH_ID_ALL_COPY 0      // all copy mode
#define TPL_SCH_ID_NOT_DETERMINE 1  // simt not determine 
#define TPL_SCH_ID_DETERMINE 2      // simt determine of NCHW
#define TPL_SCH_ID_DETERMINE_HW 3    // simt determine of HW
#define TPL_SCH_ID_DETERMINE_1D 4    // NCHW 1D determine 
#define TPL_SCH_ID_NOT_DETERMINE_HW 5  // simt not determine of HW 

#define TPL_FORMAT_NCHW 0
#define TPL_FORMAT_NHWC 1

#define TPL_ALIGN_CORNERS_FALSE 0
#define TPL_ALIGN_CORNERS_TRUE 1

#define TPL_HALF_PIXEL_CENTERS_FALSE 0
#define TPL_HALF_PIXEL_CENTERS_TRUE 1

#define TPL_IDX_INT32 0
#define TPL_IDX_INT64 1

#define RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TPL_KEY_DECL()                                                                \
    ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_SCH_ID_ALL_COPY,          \
                          TPL_SCH_ID_NOT_DETERMINE, TPL_SCH_ID_DETERMINE, TPL_SCH_ID_DETERMINE_HW, \
                          TPL_SCH_ID_DETERMINE_1D,TPL_SCH_ID_NOT_DETERMINE_HW),                    \
        ASCENDC_TPL_UINT_DECL(format, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_FORMAT_NCHW, TPL_FORMAT_NHWC),  \
        ASCENDC_TPL_UINT_DECL(alignCorners, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_ALIGN_CORNERS_FALSE,          \
                              TPL_ALIGN_CORNERS_TRUE),                                                              \
        ASCENDC_TPL_UINT_DECL(halfPixelCenters, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_HALF_PIXEL_CENTERS_FALSE, \
                              TPL_HALF_PIXEL_CENTERS_TRUE),                                                         \
        ASCENDC_TPL_UINT_DECL(idxType, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_IDX_INT32, TPL_IDX_INT64)


#define RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TPL_KEY_SEL_GROUP1()                                                           \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, TPL_SCH_ID_ALL_COPY, TPL_SCH_ID_NOT_DETERMINE, TPL_SCH_ID_NOT_DETERMINE_HW),               \
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_FORMAT_NCHW, TPL_FORMAT_NHWC),               \
        ASCENDC_TPL_UINT_SEL(alignCorners, ASCENDC_TPL_UI_LIST, TPL_ALIGN_CORNERS_TRUE), \
        ASCENDC_TPL_UINT_SEL(halfPixelCenters, ASCENDC_TPL_UI_LIST, TPL_HALF_PIXEL_CENTERS_FALSE),          \
        ASCENDC_TPL_UINT_SEL(idxType, ASCENDC_TPL_UI_LIST, TPL_IDX_INT32, TPL_IDX_INT64)


#define RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TPL_KEY_SEL_GROUP2()                                                           \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, TPL_SCH_ID_DETERMINE_1D, TPL_SCH_ID_ALL_COPY, TPL_SCH_ID_DETERMINE),               \
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_FORMAT_NCHW, TPL_FORMAT_NHWC),               \
        ASCENDC_TPL_UINT_SEL(alignCorners, ASCENDC_TPL_UI_LIST, TPL_ALIGN_CORNERS_FALSE), \
        ASCENDC_TPL_UINT_SEL(halfPixelCenters, ASCENDC_TPL_UI_LIST, TPL_HALF_PIXEL_CENTERS_FALSE,              \
                             TPL_HALF_PIXEL_CENTERS_TRUE),                                                    \
        ASCENDC_TPL_UINT_SEL(idxType, ASCENDC_TPL_UI_LIST, TPL_IDX_INT32, TPL_IDX_INT64)


#define RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TPL_KEY_SEL_GROUP3()                                                           \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, TPL_SCH_ID_DETERMINE_HW),               \
        ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_FORMAT_NCHW),               \
        ASCENDC_TPL_UINT_SEL(alignCorners, ASCENDC_TPL_UI_LIST, TPL_ALIGN_CORNERS_FALSE), \
        ASCENDC_TPL_UINT_SEL(halfPixelCenters, ASCENDC_TPL_UI_LIST, TPL_HALF_PIXEL_CENTERS_FALSE,              \
                             TPL_HALF_PIXEL_CENTERS_TRUE),                                                    \
        ASCENDC_TPL_UINT_SEL(idxType, ASCENDC_TPL_UI_LIST, TPL_IDX_INT32, TPL_IDX_INT64)


ASCENDC_TPL_ARGS_DECL(ResizeNearestNeighborV2Grad, RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TPL_KEY_DECL());
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TPL_KEY_SEL_GROUP1()),
                ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TPL_KEY_SEL_GROUP2()),
                ASCENDC_TPL_ARGS_SEL(RESIZE_NEAREST_NEIGHBOR_V2_GRAD_TPL_KEY_SEL_GROUP3()));
    
 #endif // CANN_RESIZE_NEIGHBOR_V2_GRAD_TILING_KEY_H