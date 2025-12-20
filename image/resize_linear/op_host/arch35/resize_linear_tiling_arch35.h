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
 * \file resize_linear_tiling_arc35.h
 * \brief resize_linear_tiling_arc35
 */
#ifndef OPS_IMAGE_RESIZE_LINEAR_OP_HOST_RESIZE_LINEAR_TILING_ARCH35_H_
#define OPS_IMAGE_RESIZE_LINEAR_OP_HOST_RESIZE_LINEAR_TILING_ARCH35_H_

#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/const_util.h"
#include "tiling_base/tiling_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ResizeLinearTilingData)
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blkProcessNum);
TILING_DATA_FIELD_DEF(int64_t, splitBlockTailFactor);
TILING_DATA_FIELD_DEF(int64_t, lenSrcL);
TILING_DATA_FIELD_DEF(int64_t, lenDesL);
TILING_DATA_FIELD_DEF(float, scaleL);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResizeLinear, ResizeLinearTilingData)
struct ResizeLinearCompileInfo {
    int32_t totalCoreNum = 0;
    int32_t totalUbSize = 0;
};
} // namespace optiling

#endif // OPS_IMAGE_RESIZE_LINEAR_OP_HOST_RESIZE_LINEAR_TILING_ARCH35_H_
