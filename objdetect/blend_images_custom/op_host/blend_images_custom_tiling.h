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
 * \file blend_images_custom_tiling.h
 * \brief BlendImagesCustom 算子 TilingData 结构定义.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_BLEND_IMAGES_CUSTOM_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_BLEND_IMAGES_CUSTOM_TILING_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingDataBlendImages)
  TILING_DATA_FIELD_DEF(uint32_t, totalAlphaLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BlendImagesCustom, TilingDataBlendImages)
}

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_BLEND_IMAGES_CUSTOM_TILING_H_
