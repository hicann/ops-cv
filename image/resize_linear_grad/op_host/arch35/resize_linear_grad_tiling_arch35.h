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
 * \file resize_linear_grad_tiling_arch35.h
 * \brief resize_linear_grad_tiling_arch35
 */
#ifndef OPS_IMAGE_RESIZE_LINEAR_GRAD_OP_HOST_RESIZE_LINEAR_GRAD_TILING_ARCH35_H_
#define OPS_IMAGE_RESIZE_LINEAR_GRAD_OP_HOST_RESIZE_LINEAR_GRAD_TILING_ARCH35_H_

#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ResizeLinearGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, initCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blkProcessNum);
TILING_DATA_FIELD_DEF(int64_t, ubLoopSizeB);
TILING_DATA_FIELD_DEF(int64_t, ubLoopSizeT);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactorTailB);
TILING_DATA_FIELD_DEF(int64_t, ubFactorTailT);
TILING_DATA_FIELD_DEF(int64_t, lenSrcLOrUb);
TILING_DATA_FIELD_DEF(int64_t, lenDesL);
TILING_DATA_FIELD_DEF(float, scaleL);
TILING_DATA_FIELD_DEF(float, inverseScaleL);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResizeLinearGrad, ResizeLinearGradTilingData)
struct ResizeLinearGradCompileInfo {
    int32_t totalCoreNum = 0;
    int32_t totalUbSize = 0;
    int32_t blockSize = 0;
};
} // namespace optiling

#endif // OPS_IMAGE_RESIZE_LINEAR_GRAD_OP_HOST_RESIZE_LINEAR_GRAD_TILING_ARCH35_H_
