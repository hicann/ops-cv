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
 * \file yolo_tiling_data.h
 * \brief Tiling data struct for yolo operator
 */

#ifndef YOLO_TILING_DATA_H_
#define YOLO_TILING_DATA_H_

struct YoloTilingData {
    int32_t N = 0;           // batch size
    int32_t boxes = 0;       // number of anchor boxes B
    int32_t classes = 0;     // number of classes K
    int64_t HW = 0;          // H * W (spatial dimension flattened)
    int64_t ceilHW = 0;      // HW aligned to 32B/2 for output coord stride
    int64_t ceilBoxesHw = 0; // boxes*HW aligned to 32B/2 for output obj/cls stride
};

#endif // YOLO_TILING_DATA_H_
