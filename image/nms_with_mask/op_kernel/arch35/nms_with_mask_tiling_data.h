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
 * \file nms_with_mask_tiling_data.h
 * \brief nms_with_mask_tiling_data info
 */

#ifndef NMS_WITH_MASK_ARCH35_TILING_DATA_H_
#define NMS_WITH_MASK_ARCH35_TILING_DATA_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct NMSWithMaskTilingData {
    int64_t boxesNum{0};
    int64_t usedCoreNum{0};
    int64_t groupSize{0};
    int64_t groupNum{0};
    int64_t blockNum{0};
    int64_t headCoreNum{0};
    int64_t blockPerHead{0};
    float iouThreshold{0};
};
#endif