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
 * \file background_replace.cc
 * \brief BackgroundReplace算子Tiling入口
 */
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "background_replace_tiling.h"
#include <limits>

namespace optiling {

static constexpr uint64_t TILING_KEY_HALF_C1 = 1;
static constexpr uint64_t TILING_KEY_UINT8_C1 = 2;
static constexpr uint64_t TILING_KEY_HALF_C3 = 3;
static constexpr uint64_t TILING_KEY_UINT8_C3 = 4;

static constexpr uint32_t ASCEND_310P_BLOCK_DIM = 8;

static ge::graphStatus TilingBackgroundReplace(gert::TilingContext* context)
{
    TilingDataBackgroundReplace tiling;
    auto tensorBkg = context->GetInputTensor(0);
    auto tensorSrc = context->GetInputTensor(1);
    auto tensorMask = context->GetInputTensor(2);
    auto outDesc = context->GetOutputDesc(0);
    if (tensorBkg == nullptr || tensorSrc == nullptr || tensorMask == nullptr || outDesc == nullptr) {
        OP_LOGE("BackgroundReplace", "bkg, src, mask or out tensor is null.");
        return ge::GRAPH_FAILED;
    }
    int64_t maskSize = tensorMask->GetShapeSize();
    int64_t bkgSize = tensorBkg->GetShapeSize();
    if (maskSize < 0 || bkgSize < 0 || maskSize > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE("BackgroundReplace", "mask or background shape size is invalid.");
        return ge::GRAPH_FAILED;
    }
    auto bkgDataType = tensorBkg->GetDataType();
    if (tensorSrc->GetDataType() != bkgDataType || outDesc->GetDataType() != bkgDataType) {
        OP_LOGE("BackgroundReplace", "bkg, src and out dtype must be the same.");
        return ge::GRAPH_FAILED;
    }
    uint64_t tiling_key = 0;
    if (maskSize == bkgSize && bkgDataType == ge::DT_FLOAT16) {
        tiling_key = TILING_KEY_HALF_C1;
    } else if (maskSize == bkgSize && bkgDataType == ge::DT_UINT8) {
        tiling_key = TILING_KEY_UINT8_C1;
    } else if (maskSize != bkgSize && bkgDataType == ge::DT_FLOAT16) {
        tiling_key = TILING_KEY_HALF_C3;
    } else if (maskSize != bkgSize && bkgDataType == ge::DT_UINT8) {
        tiling_key = TILING_KEY_UINT8_C3;
    }
    tiling.set_size(static_cast<uint32_t>(maskSize));
    context->SetTilingKey(tiling_key);
    context->SetBlockDim(ASCEND_310P_BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForBackgroundReplace(gert::TilingParseContext* context)
{
    OP_LOGD("BackgroundReplace", "TilingPrepareForBackgroundReplace start.");
    return ge::GRAPH_SUCCESS;
}
struct BackgroundReplaceCompileInfo {};
IMPL_OP_OPTILING(BackgroundReplace)
    .Tiling(TilingBackgroundReplace)
    .TilingParse<BackgroundReplaceCompileInfo>(TilingPrepareForBackgroundReplace);
} // namespace optiling
