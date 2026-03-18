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
 * \file rasterizer_tiling.h
 * \brief
 */

#ifndef __RASTERIZER_TILING_H__
#define __RASTERIZER_TILING_H__

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct RasterizerCompileInfo {
};

BEGIN_TILING_DATA_DEF(RasterizerTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numFaces);
  TILING_DATA_FIELD_DEF(uint32_t, numVertices);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(float, occlusionTruncation);
  TILING_DATA_FIELD_DEF(uint32_t, useDepthPrior);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Rasterizer, RasterizerTilingData)
} // namespace optiling

#endif