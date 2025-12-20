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
 * \file resize_bicubic_v2_grad_simt_determine_tiling_arch35.cpp
 * \brief resize_bicubic_v2_grad_simt_determine_tiling_arch35
 */
#include "resize_bicubic_v2_grad_tiling_arch35.h"

namespace optiling {

constexpr uint64_t TILING_KEY_SIMT_DETERMINE = 20000;
constexpr uint64_t TILING_KEY_SIMT_DETERMINE_IDX64 = 20001;
constexpr uint64_t TILING_PRIORITY_SIMT_DETERMINE = 2000;

bool ResizeBicubicV2GradSimtDetermineTiling::IsCapable()
{
    if (compileInfo_.isDetermine) {
        calcInfo_.isMatchDetermine = 1;
        return true;
    } else {
        calcInfo_.isMatchDetermine = 0;
        if (inputInfo_.lenSrcH < inputInfo_.lenDstH || inputInfo_.lenSrcW < inputInfo_.lenDstW) {
            calcInfo_.isMatchDetermine = 1;
            return true;
        }
        this->SetScales();
        if ((calcInfo_.scaleH > 0.0f && calcInfo_.scaleH < 1.0f) ||
            (calcInfo_.scaleW > 0.0f && calcInfo_.scaleW < 1.0f)) {
            calcInfo_.isMatchDetermine = 1;
            return true;
        }
    }
    return false;
}

void ResizeBicubicV2GradSimtDetermineTiling::SetTilingData()
{
    tilingData_.set_lenC(inputInfo_.lenC);
    tilingData_.set_lenSrcH(inputInfo_.lenSrcH);
    tilingData_.set_lenSrcW(inputInfo_.lenSrcW);
    tilingData_.set_lenDstH(inputInfo_.lenDstH);
    tilingData_.set_lenDstW(inputInfo_.lenDstW);
    tilingData_.set_format(inputInfo_.format);
    tilingData_.set_alignCorners(inputInfo_.alignCorners);
    tilingData_.set_useCoreNum(calcInfo_.useCoreNum);
    tilingData_.set_coreFactor(calcInfo_.coreFactor);
    tilingData_.set_coreTailFactor(calcInfo_.coreTailFactor);
    tilingData_.set_scaleH(calcInfo_.scaleH);
    tilingData_.set_scaleW(calcInfo_.scaleW);
    tilingData_.set_inverseScaleH(calcInfo_.inverseScaleH);
    tilingData_.set_inverseScaleW(calcInfo_.inverseScaleW);
}

void ResizeBicubicV2GradSimtDetermineTiling::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "ResizeBicubicV2Grad tilingData: lenC is %ld, lenSrcH is %ld, lenSrcW is %ld, lenDstH is %ld, lenDstW is %ld, \
format is %ld, alignCorners is %ld, useCoreNum is %ld, coreFactor is %ld, coreTailFactor is %ld, scaleH is %f, \
scaleW is %f, inverseScaleH is %f, inverseScaleW is %f",
        tilingData_.get_lenC(), tilingData_.get_lenSrcH(), tilingData_.get_lenSrcW(), tilingData_.get_lenDstH(),
        tilingData_.get_lenDstW(), tilingData_.get_format(), tilingData_.get_alignCorners(),
        tilingData_.get_useCoreNum(), tilingData_.get_coreFactor(), tilingData_.get_coreTailFactor(),
        tilingData_.get_scaleH(), tilingData_.get_scaleW(), tilingData_.get_inverseScaleH(),
        tilingData_.get_inverseScaleW());
    return;
}

ge::graphStatus ResizeBicubicV2GradSimtDetermineTiling::DoOpTiling()
{
    this->SetScales();

    calcInfo_.useCoreNum = calcInfo_.yShapeSize < compileInfo_.coreNum ? calcInfo_.yShapeSize : compileInfo_.coreNum;
    calcInfo_.coreFactor = Ops::Base::FloorDiv(calcInfo_.yShapeSize, calcInfo_.useCoreNum);
    calcInfo_.coreTailFactor = calcInfo_.yShapeSize - calcInfo_.coreFactor * calcInfo_.useCoreNum;

    SetTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

uint64_t ResizeBicubicV2GradSimtDetermineTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILING_KEY_SIMT_DETERMINE_IDX64;
    if (this->IsUseIdx32()) {
        tilingKey = TILING_KEY_SIMT_DETERMINE;
    }
    return tilingKey;
}

ge::graphStatus ResizeBicubicV2GradSimtDetermineTiling::PostTiling()
{
    context_->SetBlockDim(calcInfo_.useCoreNum);

    OP_CHECK_IF(
        tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual simt determine tiling data size %zu > context tiling data size %zu",
            tilingData_.GetDataSize(), context_->GetRawTilingData()->GetCapacity()),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(
    ResizeBicubicV2Grad, ResizeBicubicV2GradSimtDetermineTiling, TILING_PRIORITY_SIMT_DETERMINE);
} // namespace optiling
