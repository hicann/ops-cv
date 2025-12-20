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
 * \file upsample_nearest3d_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "upsample_nearest3d_tiling.h"
#include "image/upsample_nearest3d/op_kernel/arch35/upsample_nearest3d_tiling_data.h"
#include "image/upsample_nearest3d/op_kernel/arch35/upsample_nearest3d_tiling_key.h"

namespace optiling {
using namespace Ops::Cv::OpTiling;

constexpr int32_t CONST_0 = 0;
constexpr int32_t CONST_1 = 1;
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int32_t CONST_4 = 4;
constexpr int64_t INPUT_DIMS = 5;
constexpr int64_t THREAD_NUM = 2048; // simt开的线程数
constexpr int32_t CACHE_LINE = 128;
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr const float EPSILON = 1e-8f;

struct BaseTilingData {
    int64_t dimN = 0;
    int64_t dimC = 0;
    int64_t inD = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t outSize = 0;
    int64_t blkProcessNum = 0;
    float scaleD = 0.0;
    float scaleH = 0.0;
    float scaleW = 0.0;
    int32_t isUint32 = 1;
    int32_t schId = 0;
    int32_t coreNum = 0;
    int32_t ubSize = 0;
    int32_t realCoreNum = 0;
    int32_t tailBlockNum = 0;
    int32_t ubFactor = 0;
    int32_t dtypeSize = 0;
    int32_t cacheLineNum = 0;
    int32_t oneBlockNum = 0;
};

static const std::map<ge::DataType, int32_t> inputDtypeList = { { ge::DT_DOUBLE, 8 },
    { ge::DT_UINT8, 1 },
    { ge::DT_FLOAT, 4 },
    { ge::DT_FLOAT16, 2 },
    { ge::DT_BF16, 2 } };
class UpsampleNearest3dRegbaseTiling {
public:
    explicit UpsampleNearest3dRegbaseTiling(gert::TilingContext *context) : context_(context){};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputShapeAndAttr();
    void ComputeScales(float scaleD, float scaleH, float scaleW);
    void CalTilingData();
    void ComputeDataCopy();
    void FillTilingData();
    void PrintTilingData();

private:
    BaseTilingData baseTiling_;
    gert::TilingContext *context_ = nullptr;
    UpsampleNearest3dRegBaseTilingData *tilingData_{ nullptr };
};
void UpsampleNearest3dRegbaseTiling::ComputeDataCopy()
{
    int64_t coreNum = static_cast<int64_t>(baseTiling_.coreNum);
    if (baseTiling_.outSize <= static_cast<int64_t>(baseTiling_.cacheLineNum * baseTiling_.coreNum)) {
        baseTiling_.realCoreNum = static_cast<int32_t>(baseTiling_.outSize) / baseTiling_.cacheLineNum;
    }
    if (baseTiling_.realCoreNum == 0) {
        baseTiling_.realCoreNum = 1;
    }
    baseTiling_.blkProcessNum = baseTiling_.outSize / static_cast<int64_t>(baseTiling_.realCoreNum);
    baseTiling_.tailBlockNum =
        static_cast<int32_t>(baseTiling_.outSize % static_cast<int64_t>(baseTiling_.realCoreNum));
    baseTiling_.ubFactor =
        (baseTiling_.ubSize - baseTiling_.oneBlockNum * baseTiling_.dtypeSize) / (CONST_2 * baseTiling_.dtypeSize);
    baseTiling_.ubFactor = (baseTiling_.ubFactor / baseTiling_.oneBlockNum) * baseTiling_.oneBlockNum;
    return;
}

void UpsampleNearest3dRegbaseTiling::CalTilingData()
{
    int64_t outDHW = baseTiling_.outD * baseTiling_.outH * baseTiling_.outW;
    int64_t outCDHW = baseTiling_.dimC * outDHW;
    int64_t outNCDHW = baseTiling_.dimN * outCDHW;
    int64_t maxNum = static_cast<int64_t>(baseTiling_.coreNum) * THREAD_NUM;
    bool isDataCopy = baseTiling_.outD == baseTiling_.inD && baseTiling_.outH == baseTiling_.inH &&
        baseTiling_.outW == baseTiling_.inW && std::abs(baseTiling_.scaleD - 1.0f) <= EPSILON && std::abs(baseTiling_.scaleH - 1.0f) <= EPSILON &&
        std::abs(baseTiling_.scaleW - 1.0f) <= EPSILON;
    baseTiling_.realCoreNum = baseTiling_.coreNum;
    baseTiling_.cacheLineNum = CACHE_LINE / baseTiling_.dtypeSize;

    if (isDataCopy) {
        OP_LOGI(context_, "enter datacopy");
        baseTiling_.schId = 0; // 纯copy模板
        baseTiling_.isUint32 = 0;
        ComputeDataCopy();
    } else {
        int64_t allNum = 0;
        if (outDHW >= maxNum) {
            OP_LOGI(context_, "enter simt dhw");
            baseTiling_.schId = 1; // dhw 分线程， nc拉成一维
            baseTiling_.dimN = baseTiling_.dimN * baseTiling_.dimC;
            allNum = outDHW;
        } else if (outCDHW >= maxNum) {
            OP_LOGI(context_, "enter simt cdhw");
            baseTiling_.schId = CONST_2; // cdhw 分线程, 和竞品一致
            allNum = outCDHW;
        } else {
            OP_LOGI(context_, "enter simt ncdhw");
            int64_t coreNum = static_cast<int64_t>(baseTiling_.coreNum);
            if (outNCDHW < coreNum) {
                baseTiling_.realCoreNum = static_cast<int32_t>(outNCDHW);
            }
            allNum = outNCDHW;
            baseTiling_.schId = CONST_3; // ncdhw 分线程
        }
        baseTiling_.blkProcessNum = allNum / static_cast<int64_t>(baseTiling_.realCoreNum);
        baseTiling_.tailBlockNum = static_cast<int32_t>(allNum % static_cast<int64_t>(baseTiling_.realCoreNum));
    }
    return;
}

ge::graphStatus UpsampleNearest3dRegbaseTiling::CheckInputParams()
{
    auto input = context_->GetInputDesc(CONST_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input);
    auto inputDtype = input->GetDataType();
    OP_CHECK_IF(inputDtypeList.count(inputDtype) == 0,
        OP_LOGE(context_, "input dtype is not support, but input dtype is %d", inputDtype), return ge::GRAPH_FAILED);
    baseTiling_.dtypeSize = inputDtypeList.find(inputDtype)->second;
    int32_t ubBlockSize = static_cast<int32_t>(Ops::Base::GetUbBlockSize(context_));
    baseTiling_.oneBlockNum = ubBlockSize / baseTiling_.dtypeSize;
    auto outDescPtr0 = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDescPtr0);
    auto outDtype = outDescPtr0->GetDataType();
    OP_CHECK_IF(outDtype != inputDtype, OP_LOGE(context_, "input and output dtype must be same"),
        return ge::GRAPH_FAILED);
    auto inputX = context_->GetInputShape(0);
    auto outY = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outY);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    auto outShape = EnsureNotScalar(outY->GetStorageShape());
    OP_CHECK_IF((inputShape.GetDimNum() != INPUT_DIMS) || (outShape.GetDimNum() != INPUT_DIMS),
        OP_LOGE(context_, "The dim of input0 or output0 should be equal to 5."), return ge::GRAPH_FAILED);
    int64_t inputSize = inputShape.GetShapeSize();
    int64_t outputSize = outShape.GetShapeSize();
    baseTiling_.dimN = inputShape.GetDim(CONST_0);
    baseTiling_.dimC = inputShape.GetDim(CONST_1);
    baseTiling_.inD = inputShape.GetDim(CONST_2);
    baseTiling_.inH = inputShape.GetDim(CONST_3);
    baseTiling_.inW = inputShape.GetDim(CONST_4);
    baseTiling_.outD = outShape.GetDim(CONST_2);
    baseTiling_.outH = outShape.GetDim(CONST_3);
    baseTiling_.outW = outShape.GetDim(CONST_4);
    baseTiling_.outSize = outputSize;
    OP_CHECK_IF(inputSize == 0 || outputSize == 0, OP_LOGE(context_, "not support empty input or output"),
        ge::GRAPH_FAILED);
    int64_t uint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    int32_t isUint32 = static_cast<int32_t>((inputSize <= uint32Max) && (outputSize <= uint32Max));
    baseTiling_.isUint32 = isUint32;
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearest3dRegbaseTiling::ComputeScales(float scaleD, float scaleH, float scaleW)
{
    baseTiling_.scaleD = static_cast<float>(baseTiling_.inD) / static_cast<float>(baseTiling_.outD);
    baseTiling_.scaleH = static_cast<float>(baseTiling_.inH) / static_cast<float>(baseTiling_.outH);
    baseTiling_.scaleW = static_cast<float>(baseTiling_.inW) / static_cast<float>(baseTiling_.outW);
    if (scaleD > 0.0f) {
        baseTiling_.scaleD = 1.0f / scaleD;
    }
    if (scaleH > 0.0f) {
        baseTiling_.scaleH = 1.0f / scaleH;
    }
    if (scaleW > 0.0f) {
        baseTiling_.scaleW = 1.0f / scaleW;
    }
    return;
}

ge::graphStatus UpsampleNearest3dRegbaseTiling::CheckInputShapeAndAttr()
{
    auto outDescPtr0 = context_->GetOutputShape(0);
    auto outShape = outDescPtr0->GetStorageShape();
    int64_t outN = outShape.GetDim(CONST_0);
    int64_t outC = outShape.GetDim(CONST_1);
    OP_CHECK_IF((outN != baseTiling_.dimN) || (outC != baseTiling_.dimC),
        OP_LOGE(context_, "The N and C dimensions of the input and output need to be the same."),
        return ge::GRAPH_FAILED);
    auto *attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_, "attrs is nullptr"), return ge::GRAPH_FAILED);
    auto outputSize = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_0);
    OP_CHECK_IF(outputSize == nullptr, OP_LOGE(context_, "outputSize is nullptr"), return ge::GRAPH_FAILED);
    const float *scaleDPtr = attrs->GetAttrPointer<float>(1);
    OP_CHECK_IF(scaleDPtr == nullptr, OP_LOGE(context_, "scaleDPtr is nullptr"), return ge::GRAPH_FAILED);
    float scaleD = *scaleDPtr;
    const float *scaleHPtr = attrs->GetAttrPointer<float>(CONST_2);
    OP_CHECK_IF(scaleHPtr == nullptr, OP_LOGE(context_, "scaleHPtr is nullptr"), return ge::GRAPH_FAILED);
    float scaleH = *scaleHPtr;
    const float *scaleWPtr = attrs->GetAttrPointer<float>(CONST_3);
    OP_CHECK_IF(scaleWPtr == nullptr, OP_LOGE(context_, "scaleWPtr is nullptr"), return ge::GRAPH_FAILED);
    float scaleW = *scaleWPtr;
    OP_LOGI(context_, "scaleD %f, scaleH %f, scaleW %f", scaleD, scaleH, scaleW);
    int64_t outSizeNum = outputSize->GetSize();
    OP_CHECK_IF(outSizeNum > 0 && (outSizeNum != CONST_3),
        OP_LOGE(context_, "the num of outputSize is %ld, invalid, must be 3", outSizeNum), return ge::GRAPH_FAILED);
    const int64_t *outData = reinterpret_cast<const int64_t *>(outputSize->GetData());
    int64_t outD = baseTiling_.outD;
    int64_t outH = baseTiling_.outH;
    int64_t outW = baseTiling_.outW;
    if (outSizeNum == CONST_3) {
        outD = outData[0];
        outH = outData[CONST_1];
        outW = outData[CONST_2];
    }
    OP_CHECK_IF((baseTiling_.outD != outD) || (baseTiling_.outH != outH) || (baseTiling_.outW != outW),
        OP_LOGE(context_, "The output D H W dimensions must be the same as the attribute outputSize."),
        return ge::GRAPH_FAILED);
    ComputeScales(scaleD, scaleH, scaleW);
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearest3dRegbaseTiling::FillTilingData()
{
    tilingData_->ubFactor = baseTiling_.ubFactor;
    tilingData_->tailBlockNum = baseTiling_.tailBlockNum;
    tilingData_->blkProcessNum = baseTiling_.blkProcessNum;
    tilingData_->lenN = baseTiling_.dimN;
    tilingData_->lenC = baseTiling_.dimC;
    tilingData_->inD = baseTiling_.inD;
    tilingData_->inH = baseTiling_.inH;
    tilingData_->inW = baseTiling_.inW;
    tilingData_->outD = baseTiling_.outD;
    tilingData_->outH = baseTiling_.outH;
    tilingData_->outW = baseTiling_.outW;
    tilingData_->scaleD = baseTiling_.scaleD;
    tilingData_->scaleH = baseTiling_.scaleH;
    tilingData_->scaleW = baseTiling_.scaleW;
}

void UpsampleNearest3dRegbaseTiling::PrintTilingData()
{
    OP_LOGD(context_,
        "ubFactor %d, tailBlockNum %d, blkProcessNum %ld, lenN %ld, lenC %ld, inD %ld, inH %ld, inW %ld, outD %ld, "
        "outH %ld, outW %ld, scaleD %f, scaleH %f, scaleW %f",
        tilingData_->ubFactor, tilingData_->tailBlockNum, tilingData_->blkProcessNum, tilingData_->lenN,
        tilingData_->lenC, tilingData_->inD, tilingData_->inH, tilingData_->inW, tilingData_->outD, tilingData_->outH,
        tilingData_->outW, tilingData_->scaleD, tilingData_->scaleH, tilingData_->scaleW);
}

ge::graphStatus UpsampleNearest3dRegbaseTiling::Init()
{
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context_, "coreNum must greater than zero, but is %d", coreNum),
        return ge::GRAPH_FAILED);
    baseTiling_.coreNum = coreNum;
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0UL, OP_LOGE(context_, "ubSize must greater than zero, but is %lu", ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGI(context_, "coreNum is %ld, ubSize is %lu", coreNum, ubSize);
    baseTiling_.ubSize = static_cast<int32_t>(ubSize);
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<UpsampleNearest3dRegBaseTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_, "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData_, sizeof(UpsampleNearest3dRegBaseTilingData), 0,
        sizeof(UpsampleNearest3dRegBaseTilingData)) != EOK),
        OP_LOGE(context_, "memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearest3dRegbaseTiling::DoTiling()
{
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CheckInputParams is failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeAndAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CheckInputShapes is failed"),
        return ge::GRAPH_FAILED);
    CalTilingData();
    FillTilingData();
    PrintTilingData();
    uint64_t schId = static_cast<uint64_t>(baseTiling_.schId);
    uint64_t isUint32 = static_cast<uint64_t>(baseTiling_.isUint32);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, isUint32);
    OP_LOGI(context_, "tilingKey %lu, schId %lu, isUint32 %lu, realCoreNum %d", tilingKey, schId, isUint32,
        baseTiling_.realCoreNum);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(baseTiling_.realCoreNum);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4UpsampleNearest3dRegbase(gert::TilingContext *context)
{
    UpsampleNearest3dRegbaseTiling tilingImpl = UpsampleNearest3dRegbaseTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4UpsampleNearest3dRegbase init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4UpsampleNearest3dRegbase do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context, "end Tiling4UpsampleNearest3dRegbase");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling