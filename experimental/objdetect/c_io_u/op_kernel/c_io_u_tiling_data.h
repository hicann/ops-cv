/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file c_io_u_tiling_data.h
 * \brief tiling data struct for CIoU
 */

#ifndef __C_IO_U_TILING_DATA_H__
#define __C_IO_U_TILING_DATA_H__

#include <cstdint>

struct CIoUTilingData {
    uint32_t totalN;      // total N (cols of (4,N))
    uint32_t basePerCore; // base cols per core (multiple of alignElem)
    uint32_t pivot;       // first 'pivot' cores get +alignElem extra cols
    uint32_t tileN;       // cols per UB tile
    uint32_t usedCoreNum; // actual launched cores
    uint32_t alignElem;   // 32 / sizeof(dtype) cache-line element count
    uint32_t tailN;       // unaligned tail (< alignElem); handled by last core
    int32_t trans;        // 0 = xyxy, 1 = cxcywh
    int32_t modeId;       // 0 = "iou", 1 = "iof"
    int32_t atanSubFlag;  // 0 = output zeros, 1 = compute atan_sub
    float eps;            // small denominator constant
};

#endif // __C_IO_U_TILING_DATA_H__
