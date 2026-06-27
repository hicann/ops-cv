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
 * \file resize_upsample_trilinear_tiling_arch35.cpp
 * \brief ResizeUpsampleTrilinear A950 SIMT tiling.
 */

#include <cmath>
#include <limits>
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "resize_upsample_trilinear_tiling.h"
#include "image/resize_upsample_trilinear/op_kernel/arch35/resize_upsample_trilinear_tiling_data.h"
#include "image/resize_upsample_trilinear/op_kernel/arch35/resize_upsample_trilinear_tiling_key.h"

namespace optiling {
using namespace Ops::Cv::OpTiling;

constexpr int32_t CONST_0 = 0;
constexpr int32_t CONST_1 = 1;
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int32_t CONST_4 = 4;
constexpr int64_t INPUT_DIMS = 5;
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);

constexpr int32_t OUTPUT_SIZE_ATTR = 0;
constexpr int32_t ALIGN_CORNERS_ATTR = 1;
constexpr int32_t SCALE_D_ATTR = 2;
constexpr int32_t SCALE_H_ATTR = 3;
constexpr int32_t SCALE_W_ATTR = 4;

constexpr float MAX_SUPPORT_SCALE = 50.0f;

struct ResizeUpsampleTrilinearBaseTiling {
    int64_t inD = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t outSize = 0;
    int64_t blkProcessNum = 0;
    int32_t tailBlockNum = 0;
    int32_t realCoreNum = 0;
    int32_t coreNum = 0;
    int32_t alignCorners = 0;
    uint64_t isInt32 = 1;
    float scaleD = 0.0f;
    float scaleH = 0.0f;
    float scaleW = 0.0f;
    float checkScaleD = 0.0f;
    float checkScaleH = 0.0f;
    float checkScaleW = 0.0f;
};

static bool IsInputDtypeSupported(ge::DataType inputDtype)
{
    return inputDtype == ge::DT_FLOAT || inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16;
}

static bool IsShapeSizeWithinLimit(int64_t shapeSize, uint64_t limit)
{
    return shapeSize > 0 && static_cast<uint64_t>(shapeSize) <= limit;
}

static bool IsDimProductWithinLimit(int64_t dimD, int64_t dimH, int64_t dimW, uint64_t limit)
{
    if (dimD <= 0 || dimH <= 0 || dimW <= 0) {
        return false;
    }
    uint64_t product = static_cast<uint64_t>(dimD);
    uint64_t dimHValue = static_cast<uint64_t>(dimH);
    if (product > limit / dimHValue) {
        return false;
    }
    product *= dimHValue;

    uint64_t dimWValue = static_cast<uint64_t>(dimW);
    return product <= limit / dimWValue;
}

class ResizeUpsampleTrilinearRegbaseTiling {
public:
    explicit ResizeUpsampleTrilinearRegbaseTiling(gert::TilingContext* context) : context_(context){};
    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputShapeAndAttr();
    void ComputeScales(float originalScaleD, float originalScaleH, float originalScaleW);
    void CalTilingData();
    void FillTilingData();

private:
    ResizeUpsampleTrilinearBaseTiling baseTiling_;
    gert::TilingContext* context_ = nullptr;
    ResizeUpsampleTrilinearRegBaseTilingData* tilingData_ = nullptr;
};

static float ComputeScaleValue(bool alignCorners, int64_t inputSize, int64_t outputSize, float scale)
{
    if (outputSize == inputSize) {
        return 1.0f;
    }
    if (alignCorners) {
        return outputSize > 1 ? static_cast<float>(inputSize - 1) / static_cast<float>(outputSize - 1) : 0.0f;
    }
    return scale > 0.0f ? scale : static_cast<float>(inputSize) / static_cast<float>(outputSize);
}

static float ComputeCheckScaleValue(bool alignCorners, int64_t inputSize, int64_t outputSize, float scale)
{
    if (outputSize == inputSize) {
        return 1.0f;
    }
    if (alignCorners) {
        if (inputSize > 1 && outputSize > 1) {
            return static_cast<float>(outputSize - 1) / static_cast<float>(inputSize - 1);
        }
        return static_cast<float>(outputSize) / static_cast<float>(inputSize);
    }
    return scale > 0.0f ? 1.0f / scale : static_cast<float>(outputSize) / static_cast<float>(inputSize);
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::Init()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int32_t coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNumAiv <= 0, OP_LOGE(context_, "coreNum must greater than zero, but is %d", coreNumAiv),
        return ge::GRAPH_FAILED);
    baseTiling_.coreNum = coreNumAiv;

    tilingData_ = context_->GetTilingData<ResizeUpsampleTrilinearRegBaseTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_, "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(tilingData_, sizeof(ResizeUpsampleTrilinearRegBaseTilingData), 0,
                     sizeof(ResizeUpsampleTrilinearRegBaseTilingData)) != EOK),
        OP_LOGE(context_, "memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::CheckInputParams()
{
    auto inputDesc = context_->GetInputDesc(CONST_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(!IsInputDtypeSupported(inputDtype),
        OP_LOGE(context_, "input dtype is not support, but input dtype is %d", inputDtype), return ge::GRAPH_FAILED);

    auto outputDesc = context_->GetOutputDesc(CONST_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OP_CHECK_IF(outputDesc->GetDataType() != inputDtype, OP_LOGE(context_, "input and output dtype must be same"),
        return ge::GRAPH_FAILED);

    auto inputX = context_->GetInputShape(CONST_0);
    auto outY = context_->GetOutputShape(CONST_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outY);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    auto outputShape = EnsureNotScalar(outY->GetStorageShape());
    OP_CHECK_IF((inputShape.GetDimNum() != INPUT_DIMS) || (outputShape.GetDimNum() != INPUT_DIMS),
        OP_LOGE(context_, "The dim of input or output should be equal to five."), return ge::GRAPH_FAILED);

    int64_t inputSize = inputShape.GetShapeSize();
    int64_t outputSize = outputShape.GetShapeSize();
    OP_CHECK_IF(inputSize <= 0 || outputSize <= 0, OP_LOGE(context_, "not support empty input or output"),
        return ge::GRAPH_FAILED);
    int64_t inputN = inputShape.GetDim(CONST_0);
    int64_t inputC = inputShape.GetDim(CONST_1);
    baseTiling_.inD = inputShape.GetDim(CONST_2);
    baseTiling_.inH = inputShape.GetDim(CONST_3);
    baseTiling_.inW = inputShape.GetDim(CONST_4);
    baseTiling_.outD = outputShape.GetDim(CONST_2);
    baseTiling_.outH = outputShape.GetDim(CONST_3);
    baseTiling_.outW = outputShape.GetDim(CONST_4);
    baseTiling_.outSize = outputSize;

    OP_CHECK_IF((inputN != outputShape.GetDim(CONST_0)) || (inputC != outputShape.GetDim(CONST_1)),
        OP_LOGE(context_, "The N and C dimensions of input and output must be same"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(inputN > 0 && inputC > 0),
        OP_LOGE(context_, "N and C dimensions of input and output must be greater than zero"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(baseTiling_.inD > 0 && baseTiling_.inH > 0 && baseTiling_.inW > 0 && baseTiling_.outD > 0 &&
                    baseTiling_.outH > 0 && baseTiling_.outW > 0),
        OP_LOGE(context_, "D/H/W dimensions of input and output must be greater than zero"),
        return ge::GRAPH_FAILED);

    uint64_t uint32Max = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    baseTiling_.isInt32 = static_cast<uint64_t>(
        IsShapeSizeWithinLimit(inputSize, uint32Max) && IsShapeSizeWithinLimit(outputSize, uint32Max) &&
        IsDimProductWithinLimit(baseTiling_.inD, baseTiling_.inH, baseTiling_.inW, uint32Max) &&
        IsDimProductWithinLimit(baseTiling_.outD, baseTiling_.outH, baseTiling_.outW, uint32Max));
    return ge::GRAPH_SUCCESS;
}

void ResizeUpsampleTrilinearRegbaseTiling::ComputeScales(
    float originalScaleD, float originalScaleH, float originalScaleW)
{
    bool alignCorners = baseTiling_.alignCorners == 1;
    baseTiling_.scaleD = ComputeScaleValue(alignCorners, baseTiling_.inD, baseTiling_.outD, originalScaleD);
    baseTiling_.scaleH = ComputeScaleValue(alignCorners, baseTiling_.inH, baseTiling_.outH, originalScaleH);
    baseTiling_.scaleW = ComputeScaleValue(alignCorners, baseTiling_.inW, baseTiling_.outW, originalScaleW);
    baseTiling_.checkScaleD =
        ComputeCheckScaleValue(alignCorners, baseTiling_.inD, baseTiling_.outD, baseTiling_.scaleD);
    baseTiling_.checkScaleH =
        ComputeCheckScaleValue(alignCorners, baseTiling_.inH, baseTiling_.outH, baseTiling_.scaleH);
    baseTiling_.checkScaleW =
        ComputeCheckScaleValue(alignCorners, baseTiling_.inW, baseTiling_.outW, baseTiling_.scaleW);
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::CheckInputShapeAndAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_, "attrs is nullptr"), return ge::GRAPH_FAILED);

    auto outputSizeAttr = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(OUTPUT_SIZE_ATTR);
    int64_t outputSizeNum = outputSizeAttr == nullptr ? 0 : outputSizeAttr->GetSize();
    if (outputSizeAttr != nullptr) {
        OP_CHECK_IF(outputSizeNum > 0 && outputSizeNum != CONST_3,
            OP_LOGE(context_, "output_size must be empty or have 3 values, but got %ld", outputSizeNum),
            return ge::GRAPH_FAILED);
    }
    if (outputSizeAttr != nullptr && outputSizeNum == CONST_3) {
        const int64_t* outputSizeData = outputSizeAttr->GetData();
        OP_CHECK_IF((baseTiling_.outD != outputSizeData[CONST_0]) || (baseTiling_.outH != outputSizeData[CONST_1]) ||
                        (baseTiling_.outW != outputSizeData[CONST_2]),
            OP_LOGE(context_, "output D/H/W dimensions must be same as output_size attr"),
            return ge::GRAPH_FAILED);
    }

    const bool* alignCornersPtr = attrs->GetAttrPointer<bool>(ALIGN_CORNERS_ATTR);
    baseTiling_.alignCorners = (alignCornersPtr != nullptr && *alignCornersPtr) ? 1 : 0;

    const float* scaleDPtr = attrs->GetAttrPointer<float>(SCALE_D_ATTR);
    const float* scaleHPtr = attrs->GetAttrPointer<float>(SCALE_H_ATTR);
    const float* scaleWPtr = attrs->GetAttrPointer<float>(SCALE_W_ATTR);
    float originalScaleD = scaleDPtr == nullptr ? 0.0f : *scaleDPtr;
    float originalScaleH = scaleHPtr == nullptr ? 0.0f : *scaleHPtr;
    float originalScaleW = scaleWPtr == nullptr ? 0.0f : *scaleWPtr;
    ComputeScales(originalScaleD, originalScaleH, originalScaleW);
    OP_CHECK_IF((baseTiling_.scaleD > MAX_SUPPORT_SCALE) || (baseTiling_.scaleH > MAX_SUPPORT_SCALE) ||
                    (baseTiling_.scaleW > MAX_SUPPORT_SCALE),
        OP_LOGE(context_, "Scales should not exceed 50, but got D/H/W: %f/%f/%f", baseTiling_.scaleD,
            baseTiling_.scaleH, baseTiling_.scaleW),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((baseTiling_.checkScaleD > MAX_SUPPORT_SCALE) || (baseTiling_.checkScaleH > MAX_SUPPORT_SCALE) ||
                    (baseTiling_.checkScaleW > MAX_SUPPORT_SCALE),
        OP_LOGE(context_, "Check scales should not exceed 50, but got D/H/W: %f/%f/%f", baseTiling_.checkScaleD,
            baseTiling_.checkScaleH, baseTiling_.checkScaleW),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ResizeUpsampleTrilinearRegbaseTiling::CalTilingData()
{
    baseTiling_.realCoreNum =
        baseTiling_.outSize < static_cast<int64_t>(baseTiling_.coreNum) ? baseTiling_.outSize : baseTiling_.coreNum;
    if (baseTiling_.realCoreNum <= 0) {
        baseTiling_.realCoreNum = 1;
    }
    baseTiling_.blkProcessNum = baseTiling_.outSize / static_cast<int64_t>(baseTiling_.realCoreNum);
    baseTiling_.tailBlockNum =
        static_cast<int32_t>(baseTiling_.outSize % static_cast<int64_t>(baseTiling_.realCoreNum));
}

void ResizeUpsampleTrilinearRegbaseTiling::FillTilingData()
{
    tilingData_->blkProcessNum = baseTiling_.blkProcessNum;
    tilingData_->inD = baseTiling_.inD;
    tilingData_->inH = baseTiling_.inH;
    tilingData_->inW = baseTiling_.inW;
    tilingData_->outD = baseTiling_.outD;
    tilingData_->outH = baseTiling_.outH;
    tilingData_->outW = baseTiling_.outW;
    tilingData_->tailBlockNum = baseTiling_.tailBlockNum;
    tilingData_->alignCorners = baseTiling_.alignCorners;
    tilingData_->scaleD = baseTiling_.scaleD;
    tilingData_->scaleH = baseTiling_.scaleH;
    tilingData_->scaleW = baseTiling_.scaleW;
}

ge::graphStatus ResizeUpsampleTrilinearRegbaseTiling::DoTiling()
{
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CheckInputParams failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeAndAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CheckInputShapeAndAttr failed"),
        return ge::GRAPH_FAILED);
    CalTilingData();
    FillTilingData();

    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(RESIZE_UPSAMPLE_TRILINEAR_SCH_MODE_NCDHW, baseTiling_.isInt32);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(baseTiling_.realCoreNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ResizeUpsampleTrilinearRegbase(gert::TilingContext* context)
{
    ResizeUpsampleTrilinearRegbaseTiling tilingImpl(context);
    OP_CHECK_IF(tilingImpl.Init() != ge::GRAPH_SUCCESS, OP_LOGE(context, "arch35 tiling init failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingImpl.DoTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context, "arch35 tiling do tiling failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
