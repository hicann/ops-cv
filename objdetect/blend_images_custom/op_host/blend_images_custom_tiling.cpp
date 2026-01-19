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
 * \file blend_images_custom_tiling.cpp
 * \brief BlendImagesCustom 算子 Tiling 入口.
 */
#include "blend_images_custom_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "log/log.h"

using namespace optiling;

static constexpr uint32_t ASCEND_310P_BLOCK_DIM = 8;

static ge::graphStatus Tiling4BlendImagesCustom(gert::TilingContext* context) {
    if (context == nullptr) {
      return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Tiling4BlendImagesCustom running begin");
    auto tensorAlpha = context->GetInputTensor(1);
    uint32_t totalAlphaLength = tensorAlpha->GetShapeSize();
    TilingDataBlendImages tiling_host;
    tiling_host.set_totalAlphaLength(totalAlphaLength);
    context->SetBlockDim(ASCEND_310P_BLOCK_DIM);
    tiling_host.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling_host.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForBlendImagesCustom(gert::TilingParseContext* context) {
  OP_LOGD("BlendImagesCustom", "TilingPrepareForBlendImagesCustom start.");
  if (context == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_SUCCESS;
}
struct BlendImagesCustomCompileInfo {};

namespace optiling {
IMPL_OP_OPTILING(BlendImagesCustom)
    .Tiling(Tiling4BlendImagesCustom)
    .TilingParse<BlendImagesCustomCompileInfo>(TilingPrepareForBlendImagesCustom);
}