/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_upsample_trilinear_tiling_arch35.cpp
 * \brief resize_upsample_trilinear tiling implementation for A5 architecture (SIMT paradigm)
 */
#include "resize_upsample_trilinear_tiling_arch35.h"
#include "image/resize_upsample_trilinear/op_kernel/arch35/resize_upsample_trilinear_tiling_key.h"
#include "image/resize_upsample_trilinear/op_kernel/arch35/resize_upsample_trilinear_tiling_data.h"
#include "log/log.h"
#include "securec.h"
#include <cmath>
#include <algorithm>

namespace optiling {
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr size_t DIM_4 = 4;
static constexpr size_t DIM_5 = 5;
static constexpr uint32_t MIN_THREADS_PER_BLOCK = 32;
static constexpr uint32_t MAX_THREADS_PER_BLOCK = 512;
static constexpr uint32_t MIN_ELEMENTS_PER_THREAD = 1;
static constexpr uint32_t MAX_BLOCKS = 1024;
static constexpr uint32_t MAX_ELEMENTS_PER_THREAD = 1024;
static constexpr float MAX_SUPPORT_SCALE = 50.0f;

static float ComputeSourceCoordScale(bool alignCorners, int64_t inputSize, int64_t outputSize, float scale)
{
    if (outputSize == inputSize) {
        return 1.0f;
    }
    if (alignCorners) {
        if (outputSize > 1) {
            return static_cast<float>(inputSize - 1) / static_cast<float>(outputSize - 1);
        } else {
            return 0.0f;
        }
    } else {
        if (scale > 0.0f) {
            return 1.0f / scale;
        } else {
            return static_cast<float>(inputSize) / static_cast<float>(outputSize);
        }
    }
}

bool ResizeUpsampleTrilinearArch35Tiling::IsCapable()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfo is nullptr, cannot get platform info for arch35 tiling.");
        return false;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        platformInfoCached_ = true;
        return true;
    }
    return false;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::GetPlatformInfo()
{
    if (platformInfoCached_) {
        OP_CHECK_IF(
            coreNum_ <= 0,
            OP_LOGE(context_->GetNodeName(), "coreNum is error: %d", coreNum_),
            return ge::GRAPH_FAILED);
        OP_LOGI(context_->GetNodeName(), "A5 coreNum(AIV)=%d", coreNum_);
        return ge::GRAPH_SUCCESS;
    }
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        coreNum_ <= 0,
        OP_LOGE(context_->GetNodeName(), "coreNum is error: %d", coreNum_),
        return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "A5 coreNum(AIV)=%d", coreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::ValidateAndGetInputShape()
{
    auto inputShape = context_->GetInputShape(DIM_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    gert::Shape inShape = inputShape->GetStorageShape();
    int32_t inDims = inShape.GetDimNum();
    OP_CHECK_IF(
        inDims != DIM_5,
        OP_LOGE(context_->GetNodeName(), "input dims must be 5, but got %d", inDims),
        return ge::GRAPH_FAILED);
    inN_ = inShape.GetDim(DIM_0);
    inC_ = inShape.GetDim(DIM_1);
    inputD_ = inShape.GetDim(DIM_2);
    inputH_ = inShape.GetDim(DIM_3);
    inputW_ = inShape.GetDim(DIM_4);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::ValidateAndGetOutputShape()
{
    auto outputShape = context_->GetOutputShape(DIM_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    gert::Shape outShape = outputShape->GetStorageShape();
    int32_t outDims = outShape.GetDimNum();
    OP_CHECK_IF(
        outDims != DIM_5,
        OP_LOGE(context_->GetNodeName(), "output dims must be 5, but got %d", outDims),
        return ge::GRAPH_FAILED);
    outN_ = outShape.GetDim(DIM_0);
    outC_ = outShape.GetDim(DIM_1);
    outputD_ = outShape.GetDim(DIM_2);
    outputH_ = outShape.GetDim(DIM_3);
    outputW_ = outShape.GetDim(DIM_4);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::ValidateShapeValues()
{
    OP_CHECK_IF(
        inputD_ <= 0 || inputH_ <= 0 || inputW_ <= 0,
        OP_LOGE(context_->GetNodeName(), "input D/H/W must be positive, got D=%ld H=%ld W=%ld",
                inputD_, inputH_, inputW_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputD_ <= 0 || outputH_ <= 0 || outputW_ <= 0,
        OP_LOGE(context_->GetNodeName(), "output D/H/W must be positive, got D=%ld H=%ld W=%ld",
                outputD_, outputH_, outputW_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inN_ <= 0 || inC_ <= 0,
        OP_LOGE(context_->GetNodeName(), "input N/C must be positive, got N=%ld C=%ld", inN_, inC_),
        return ge::GRAPH_FAILED);
    batchCount_ = inN_ * inC_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::ExtractAttrsAndComputeScales()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const bool* alignCornersPtr = attrs->GetAttrPointer<bool>(DIM_1);
    alignCorners_ = (alignCornersPtr != nullptr && *alignCornersPtr) ? 1 : 0;
    const float* scalesDPtr = attrs->GetAttrPointer<float>(DIM_2);
    const float* scalesHPtr = attrs->GetAttrPointer<float>(DIM_3);
    const float* scalesWPtr = attrs->GetAttrPointer<float>(DIM_4);
    float attrScaleD = (scalesDPtr != nullptr) ? *scalesDPtr : 0.0f;
    float attrScaleH = (scalesHPtr != nullptr) ? *scalesHPtr : 0.0f;
    float attrScaleW = (scalesWPtr != nullptr) ? *scalesWPtr : 0.0f;
    scaleD_ = ComputeSourceCoordScale(alignCorners_ != 0, inputD_, outputD_, attrScaleD);
    scaleH_ = ComputeSourceCoordScale(alignCorners_ != 0, inputH_, outputH_, attrScaleH);
    scaleW_ = ComputeSourceCoordScale(alignCorners_ != 0, inputW_, outputW_, attrScaleW);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::DetermineDtypeKey()
{
    auto dataType = context_->GetInputDesc(DIM_0)->GetDataType();
    if (dataType == ge::DT_FLOAT) {
        dtypeKey_ = TPL_DTYPE_FP32;
    } else if (dataType == ge::DT_FLOAT16) {
        dtypeKey_ = TPL_DTYPE_FP16;
    } else if (dataType == ge::DT_BF16) {
        dtypeKey_ = TPL_DTYPE_BF16;
    } else {
        OP_LOGE(context_->GetNodeName(), "unsupported data type: %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(ValidateAndGetInputShape() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateAndGetOutputShape() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateShapeValues() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ExtractAttrsAndComputeScales() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(DetermineDtypeKey() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::ValidateDimensionsAndComputeTotal()
{
    OP_CHECK_IF(
        inN_ != outN_ || inC_ != outC_,
        OP_LOGE(context_->GetNodeName(), "input and output N/C dimensions must match"),
        return ge::GRAPH_FAILED);
    int64_t dhw = outputD_ * outputH_;
    OP_CHECK_IF(
        outputH_ > 0 && dhw > INT64_MAX / outputW_,
        OP_LOGE(context_->GetNodeName(), "output D*H*W overflow, outputD=%ld outputH=%ld outputW=%ld",
                outputD_, outputH_, outputW_),
        return ge::GRAPH_FAILED);
    dhw *= outputW_;
    OP_CHECK_IF(
        batchCount_ > INT64_MAX / dhw,
        OP_LOGE(context_->GetNodeName(), "totalElements overflow detected, batchCount=%ld outputD=%ld outputH=%ld outputW=%ld",
                batchCount_, outputD_, outputH_, outputW_),
        return ge::GRAPH_FAILED);
    totalElements_ = static_cast<uint64_t>(batchCount_) * static_cast<uint64_t>(dhw);
    float checkScaleD = (scaleD_ > 0.0f) ? 1.0f / scaleD_ : 0.0f;
    float checkScaleH = (scaleH_ > 0.0f) ? 1.0f / scaleH_ : 0.0f;
    float checkScaleW = (scaleW_ > 0.0f) ? 1.0f / scaleW_ : 0.0f;
    OP_CHECK_IF(
        checkScaleD > MAX_SUPPORT_SCALE || checkScaleH > MAX_SUPPORT_SCALE || checkScaleW > MAX_SUPPORT_SCALE,
        OP_LOGE(context_->GetNodeName(),
                "scales exceed max support scale %f, got scaleD=%f scaleH=%f scaleW=%f",
                MAX_SUPPORT_SCALE, checkScaleD, checkScaleH, checkScaleW),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::HandleZeroElements()
{
    if (totalElements_ == 0) {
        usedCoreNum_ = 1;
        blockCount_ = 1;
        baseElementsPerBlock_ = 0;
        tailElements_ = 0;
        threadsPerBlock_ = MIN_THREADS_PER_BLOCK;
        elementsPerThread_ = MIN_ELEMENTS_PER_THREAD;
        useInt32_ = 1;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::ComputeThreadBlockConfig()
{
    OP_CHECK_IF(
        totalElements_ > static_cast<uint64_t>(MAX_ELEMENTS_PER_THREAD) * MAX_THREADS_PER_BLOCK * MAX_BLOCKS,
        OP_LOGE(context_->GetNodeName(), "totalElements too large for SIMT processing: %lu", totalElements_),
        return ge::GRAPH_FAILED);
    threadsPerBlock_ = MAX_THREADS_PER_BLOCK;
    elementsPerThread_ = MIN_ELEMENTS_PER_THREAD;
    uint64_t totalThreads = (totalElements_ + elementsPerThread_ - 1) / elementsPerThread_;
    blockCount_ = static_cast<uint32_t>((totalThreads + threadsPerBlock_ - 1) / threadsPerBlock_);
    if (blockCount_ == 0) {
        blockCount_ = 1;
    }
    if (blockCount_ > MAX_BLOCKS) {
        uint32_t minEPT = static_cast<uint32_t>(
            (totalElements_ + static_cast<uint64_t>(MAX_BLOCKS) * threadsPerBlock_ - 1) /
            (static_cast<uint64_t>(MAX_BLOCKS) * threadsPerBlock_));
        elementsPerThread_ = std::max(minEPT, elementsPerThread_);
        OP_CHECK_IF(
            elementsPerThread_ > MAX_ELEMENTS_PER_THREAD,
            OP_LOGE(context_->GetNodeName(), "cannot reduce blockCount below MAX_BLOCKS after max elementsPerThread"),
            return ge::GRAPH_FAILED);
        if(elementsPerThread_ == 0){
            return ge::GRAPH_FAILED;}
        totalThreads = (totalElements_ + elementsPerThread_ - 1) / elementsPerThread_;
        blockCount_ = static_cast<uint32_t>((totalThreads + threadsPerBlock_ - 1) / threadsPerBlock_);
        if (blockCount_ == 0) {
            blockCount_ = 1;
        }
    } else if (totalThreads < threadsPerBlock_) {
        threadsPerBlock_ = static_cast<uint32_t>(totalThreads);
        if (threadsPerBlock_ < MIN_THREADS_PER_BLOCK) {
            threadsPerBlock_ = MIN_THREADS_PER_BLOCK;
        }
        blockCount_ = 1;
    }
    threadsPerBlock_ = ((threadsPerBlock_ + 31) / 32) * 32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::ComputeFinalTilingConfig()
{
    usedCoreNum_ = (blockCount_ < static_cast<uint32_t>(coreNum_)) ? blockCount_ : static_cast<uint32_t>(coreNum_);
    baseElementsPerBlock_ = threadsPerBlock_ * elementsPerThread_;
    if (blockCount_ == 1) {
        tailElements_ = static_cast<uint32_t>(totalElements_);
    } else {
        tailElements_ = static_cast<uint32_t>(
            totalElements_ - static_cast<uint64_t>(blockCount_ - 1) * baseElementsPerBlock_);
    }

    static constexpr int64_t INT32_MAX_VAL = 2147483647LL;
    int64_t strideBcInput = inputD_ * inputH_ * inputW_;
    int64_t strideBcOutput = outputD_ * outputH_ * outputW_;
    int64_t strideDInput = inputH_ * inputW_;
    int64_t strideDOutput = outputH_ * outputW_;
    useInt32_ = (totalElements_ <= static_cast<uint64_t>(INT32_MAX_VAL) &&
                 strideBcInput <= INT32_MAX_VAL &&
                 strideBcOutput <= INT32_MAX_VAL &&
                 strideDInput <= INT32_MAX_VAL &&
                 strideDOutput <= INT32_MAX_VAL &&
                 inputW_ <= INT32_MAX_VAL &&
                 outputW_ <= INT32_MAX_VAL &&
                 inputD_ <= INT32_MAX_VAL &&
                 inputH_ <= INT32_MAX_VAL &&
                 outputD_ <= INT32_MAX_VAL &&
                 outputH_ <= INT32_MAX_VAL) ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::DoOpTiling()
{
    OP_CHECK_IF(ValidateDimensionsAndComputeTotal() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    if (HandleZeroElements() == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(ComputeThreadBlockConfig() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeFinalTilingConfig() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ResizeUpsampleTrilinearArch35Tiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(dtypeKey_);
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeUpsampleTrilinearArch35Tiling::PostTiling()
{
    ResizeUpsampleTrilinearArch35TilingData tilingData;
    tilingData.elements_per_thread = elementsPerThread_;
    tilingData.block_count = blockCount_;
    tilingData.used_core_num = usedCoreNum_;
    tilingData.base_elements_per_block = baseElementsPerBlock_;
    tilingData.tail_elements = tailElements_;
    tilingData.total_elements = totalElements_;
    tilingData.batch_count = batchCount_;
    tilingData.input_d = inputD_;
    tilingData.input_h = inputH_;
    tilingData.input_w = inputW_;
    tilingData.output_d = outputD_;
    tilingData.output_h = outputH_;
    tilingData.output_w = outputW_;
    tilingData.scale_d = scaleD_;
    tilingData.scale_h = scaleH_;
    tilingData.scale_w = scaleW_;
    tilingData.align_corners = alignCorners_;
    tilingData.use_int32 = useInt32_;

    auto* rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    uint32_t tilingSize = sizeof(tilingData);
    OP_CHECK_IF(
        rawTilingData->GetCapacity() < tilingSize,
        OP_LOGE(context_->GetNodeName(), "tiling data capacity %zu is less than required %u",
                rawTilingData->GetCapacity(), tilingSize),
        return ge::GRAPH_FAILED);
    errno_t cpyRet = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &tilingData, tilingSize);
    if (cpyRet != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s tiling data failed, ret=%d.", cpyRet);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingSize);

    context_->SetBlockDim(usedCoreNum_);

    OP_LOGI(context_->GetNodeName(),
            "ResizeUpsampleTrilinear A5 tiling: dtypeKey=%lu, elementsPerThread=%u, "
            "blockCount=%u, usedCoreNum=%u, baseElementsPerBlock=%u, tailElements=%u, totalElements=%lu",
            dtypeKey_, elementsPerThread_, blockCount_, usedCoreNum_,
            baseElementsPerBlock_, tailElements_, totalElements_);
    OP_LOGI(context_->GetNodeName(),
            "ResizeUpsampleTrilinear A5 tiling: batchCount=%ld, inputD=%ld, inputH=%ld, inputW=%ld, "
            "outputD=%ld, outputH=%ld, outputW=%ld, scaleD=%f, scaleH=%f, scaleW=%f, alignCorners=%d, useInt32=%d",
            batchCount_, inputD_, inputH_, inputW_, outputD_, outputH_, outputW_,
            scaleD_, scaleH_, scaleW_, alignCorners_, useInt32_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ResizeUpsampleTrilinear, ResizeUpsampleTrilinearArch35Tiling, 2000);
} // namespace optiling