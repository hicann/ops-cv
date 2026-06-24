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
 * \file blend_face_bg_part_two_tiling_data.h
 * \brief tiling data struct
 */
#ifndef __BLEND_FACE_BG_PART_TWO_TILING_DATA_H__
#define __BLEND_FACE_BG_PART_TWO_TILING_DATA_H__

#include <cstdint>

struct BlendFaceBgPartTwoTilingData {
    uint32_t totalElems; // total number of elements (broadcast across 4 inputs)
    uint32_t baseElems;  // base elements per core (floor division)
    uint32_t pivot;      // first 'pivot' cores get baseElems+1
    uint32_t tileSize;   // UB tile size in elements
    float epsilon;       // attribute epsilon for numerical stability
};

#endif
