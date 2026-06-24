/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_nearest_exact2d_grad_tiling_arch35.cpp
 * \brief
 */

#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "upsample_nearest_exact2d_grad_tiling.h"
#include "image/upsample_nearest_exact2d_grad/op_kernel/arch35/upsample_nearest_exact2d_grad_tiling_data.h"
#include "image/upsample_nearest_exact2d_grad/op_kernel/arch35/upsample_nearest_exact2d_grad_tiling_key.h"

namespace {
constexpr int32_t CONST_0 = 0;
constexpr int32_t CONST_1 = 1;
constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int32_t CONST_4 = 4;
constexpr int64_t INPUT_DIMS = 4;
constexpr int32_t CACHE_LINE = 128;
constexpr int64_t THREAD_NUM = 2048;
constexpr size_t WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr const float EPSILON = 1e-8f;

struct BaseTilingData {
    int64_t dimN = 0;
    int64_t dimC = 0;
    int64_t inH = 0;
    int64_t outH = 0;
    int64_t inW = 0;
    int64_t outW = 0;
    int64_t outSize = 0;
    int64_t blkProcessNum = 0;
    float scaleH = 0.0;
    float scaleW = 0.0;
    int32_t isUint32 = 1;
    int32_t ubSize = 0;
    int32_t coreNum = 0;
    int32_t schId = 0;
    int32_t realCoreNum = 0;
    int32_t tailBlockNum = 0;
    int32_t dtypeSize = 0;
    int32_t ubFactor = 0;
    int32_t cacheLineNum = 0;
    int32_t oneBlockNum = 0;
};

static const std::map<ge::DataType, int32_t> inputDtypeList = {
    {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};
} // namespace

namespace optiling {
using namespace Ops::Cv::OpTiling;

class UpsampleNearestExact2dGradRegbaseTiling {
public:
    explicit UpsampleNearestExact2dGradRegbaseTiling(gert::TilingContext* context) : tilingContext(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputParams();
    ge::graphStatus CheckInputShapeAndAttr();
    void ComputeScales(float scaleH, float scaleW);
    void ComputeDataCopy();
    void CalTilingData();
    void FillTilingData();
    void PrintTilingData();

private:
    BaseTilingData baseTiling;
    gert::TilingContext* tilingContext = nullptr;
    UpsampleNearestExact2dGradRegBaseTilingData* tilingData{nullptr};
};
void UpsampleNearestExact2dGradRegbaseTiling::ComputeDataCopy()
{
    int64_t coreNum = static_cast<int64_t>(baseTiling.coreNum);
    if (baseTiling.outSize <= static_cast<int64_t>(baseTiling.cacheLineNum * baseTiling.coreNum)) {
        baseTiling.realCoreNum = static_cast<int32_t>(baseTiling.outSize) / baseTiling.cacheLineNum;
    }
    if (baseTiling.realCoreNum == 0) {
        baseTiling.realCoreNum = 1;
    }
    baseTiling.blkProcessNum = baseTiling.outSize / static_cast<int64_t>(baseTiling.realCoreNum);
    baseTiling.tailBlockNum = static_cast<int32_t>(baseTiling.outSize % static_cast<int64_t>(baseTiling.realCoreNum));
    baseTiling.ubFactor = (baseTiling.ubSize - baseTiling.oneBlockNum * baseTiling.dtypeSize) /
                          (CONST_2 * baseTiling.dtypeSize);
    baseTiling.ubFactor = (baseTiling.ubFactor / baseTiling.oneBlockNum) * baseTiling.oneBlockNum;
    return;
}

void UpsampleNearestExact2dGradRegbaseTiling::CalTilingData()
{
    int64_t outHW = baseTiling.outH * baseTiling.outW;
    int64_t outCHW = baseTiling.dimC * outHW;
    int64_t outNCHW = baseTiling.dimN * outCHW;
    int64_t maxNum = static_cast<int64_t>(baseTiling.coreNum) * THREAD_NUM;
    bool isDataCopy = baseTiling.outH == baseTiling.inH && baseTiling.outW == baseTiling.inW &&
                      std::abs(baseTiling.scaleH - 1.0f) <= EPSILON && std::abs(baseTiling.scaleW - 1.0f) <= EPSILON;
    baseTiling.realCoreNum = baseTiling.coreNum;
    baseTiling.cacheLineNum = CACHE_LINE / baseTiling.dtypeSize;

    if (isDataCopy) {
        OP_LOGI(tilingContext, "Enter datacopy");
        baseTiling.schId = 0;
        baseTiling.isUint32 = 0;
        ComputeDataCopy();
    } else {
        int64_t allNum = 0;
        if (outHW >= maxNum) {
            OP_LOGI(tilingContext, "Enter simt hw");
            baseTiling.schId = 1;
            baseTiling.dimN = baseTiling.dimN * baseTiling.dimC;
            allNum = outHW;
        } else if (outCHW >= maxNum) {
            OP_LOGI(tilingContext, "Enter simt chw");
            baseTiling.schId = CONST_2;
            allNum = outCHW;
        } else {
            OP_LOGI(tilingContext, "Enter simt nchw");
            int64_t coreNum = static_cast<int64_t>(baseTiling.coreNum);
            if (outNCHW < coreNum) {
                baseTiling.realCoreNum = static_cast<int32_t>(outNCHW);
            }
            allNum = outNCHW;
            baseTiling.schId = CONST_3;
        }
        baseTiling.blkProcessNum = allNum / static_cast<int64_t>(baseTiling.realCoreNum);
        baseTiling.tailBlockNum = static_cast<int32_t>(allNum % static_cast<int64_t>(baseTiling.realCoreNum));
    }
    return;
}

ge::graphStatus UpsampleNearestExact2dGradRegbaseTiling::CheckInputParams()
{
    auto input = tilingContext->GetInputDesc(CONST_0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, input);
    auto inputDtype = input->GetDataType();
    OP_CHECK_IF(inputDtypeList.count(inputDtype) == 0,
                OP_LOGE(tilingContext, "Input dtype is not support, but input dtype is %d", inputDtype),
                return ge::GRAPH_FAILED);
    baseTiling.dtypeSize = inputDtypeList.find(inputDtype)->second;
    int32_t ubBlockSize = static_cast<int32_t>(Ops::Base::GetUbBlockSize(tilingContext));
    baseTiling.oneBlockNum = ubBlockSize / baseTiling.dtypeSize;
    auto outDescPtr0 = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outDescPtr0);
    auto outDtype = outDescPtr0->GetDataType();
    OP_CHECK_IF(outDtype != inputDtype, OP_LOGE(tilingContext, "Input and output dtype must be same"),
                return ge::GRAPH_FAILED);
    auto inputX = tilingContext->GetInputShape(0);
    auto outY = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outY);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    auto outShape = EnsureNotScalar(outY->GetStorageShape());
    OP_CHECK_IF((inputShape.GetDimNum() != INPUT_DIMS) || (outShape.GetDimNum() != INPUT_DIMS),
                OP_LOGE(tilingContext, "The dim of input0 or output0 should be equal to 4."), return ge::GRAPH_FAILED);
    int64_t inputSize = inputShape.GetShapeSize();
    int64_t outputSize = outShape.GetShapeSize();
    baseTiling.dimN = inputShape.GetDim(CONST_0);
    baseTiling.dimC = inputShape.GetDim(CONST_1);
    baseTiling.inH = inputShape.GetDim(CONST_2);
    baseTiling.inW = inputShape.GetDim(CONST_3);
    baseTiling.outH = outShape.GetDim(CONST_2);
    baseTiling.outW = outShape.GetDim(CONST_3);
    baseTiling.outSize = outputSize;
    OP_CHECK_IF(inputSize == 0 || outputSize == 0, OP_LOGE(tilingContext, "Not support empty input or output"),
                ge::GRAPH_FAILED);
    int64_t uint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    int32_t isUint32 = static_cast<int32_t>((inputSize <= uint32Max) && (outputSize <= uint32Max));
    baseTiling.isUint32 = isUint32;
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearestExact2dGradRegbaseTiling::ComputeScales(float scaleH, float scaleW)
{
    baseTiling.scaleH = static_cast<float>(baseTiling.inH) / static_cast<float>(baseTiling.outH);
    baseTiling.scaleW = static_cast<float>(baseTiling.inW) / static_cast<float>(baseTiling.outW);
    if (scaleH > 0.0f) {
        baseTiling.scaleH = scaleH;
    }
    if (scaleW > 0.0f) {
        baseTiling.scaleW = scaleW;
    }
    return;
}

ge::graphStatus UpsampleNearestExact2dGradRegbaseTiling::CheckInputShapeAndAttr()
{
    auto outDescPtr0 = tilingContext->GetOutputShape(0);
    auto outShape = outDescPtr0->GetStorageShape();
    int64_t outN = outShape.GetDim(CONST_0);
    int64_t outC = outShape.GetDim(CONST_1);
    OP_CHECK_IF((outN != baseTiling.dimN) || (outC != baseTiling.dimC),
                OP_LOGE(tilingContext, "The N and C dimensions of the input and output need to be the same."),
                return ge::GRAPH_FAILED);
    auto* attrs = tilingContext->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(tilingContext, "Attrs is nullptr"), return ge::GRAPH_FAILED);

    auto gradOutputSizeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_0);
    OP_CHECK_IF(gradOutputSizeAttr == nullptr, OP_LOGE(tilingContext, "GradOutputSize is nullptr"),
                return ge::GRAPH_FAILED);

    auto gradInputSize = attrs->GetAttrPointer<gert::ContinuousVector>(CONST_1);
    OP_CHECK_IF(gradInputSize == nullptr, OP_LOGE(tilingContext, "GradInputSize is nullptr"), return ge::GRAPH_FAILED);

    auto scaleHAttr = attrs->GetAttrPointer<float>(CONST_2);
    auto scaleWAttr = attrs->GetAttrPointer<float>(CONST_3);

    float scaleH = 0.0f;
    float scaleW = 0.0f;
    if (scaleHAttr != nullptr) {
        scaleH = *scaleHAttr;
    }
    if (scaleWAttr != nullptr) {
        scaleW = *scaleWAttr;
    }

    OP_LOGI(tilingContext, "ScaleH %f, scaleW %f", scaleH, scaleW);
    int64_t outSizeNum = gradInputSize->GetSize();
    OP_CHECK_IF(outSizeNum > 0 && (outSizeNum != INPUT_DIMS),
                OP_LOGE(tilingContext, "The num of gradInputSize is %ld, invalid, must be 4", outSizeNum),
                return ge::GRAPH_FAILED);
    const int64_t* outData = static_cast<const int64_t*>(gradInputSize->GetData());
    int64_t outH = baseTiling.outH;
    int64_t outW = baseTiling.outW;
    if (outSizeNum == INPUT_DIMS) {
        outH = outData[CONST_2];
        outW = outData[CONST_3];
    }
    OP_CHECK_IF((baseTiling.outH != outH) || (baseTiling.outW != outW),
                OP_LOGE(tilingContext, "The output H W dimensions must be the same as the attribute gradInputSize."),
                return ge::GRAPH_FAILED);
    ComputeScales(scaleH, scaleW);
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearestExact2dGradRegbaseTiling::FillTilingData()
{
    tilingData->ubFactor = baseTiling.ubFactor;
    tilingData->tailBlockNum = baseTiling.tailBlockNum;
    tilingData->blkProcessNum = baseTiling.blkProcessNum;
    tilingData->lenN = baseTiling.dimN;
    tilingData->lenC = baseTiling.dimC;
    tilingData->inH = baseTiling.inH;
    tilingData->inW = baseTiling.inW;
    tilingData->outH = baseTiling.outH;
    tilingData->outW = baseTiling.outW;
    tilingData->scaleH = baseTiling.scaleH;
    tilingData->scaleW = baseTiling.scaleW;
}

void UpsampleNearestExact2dGradRegbaseTiling::PrintTilingData()
{
    OP_LOGD(tilingContext,
            "ubFactor %d, tailBlockNum %d, blkProcessNum %ld, lenN %ld, lenC %ld, inH %ld, inW %ld, outH %ld, "
            "outW %ld, scaleH %f, scaleW %f",
            tilingData->ubFactor, tilingData->tailBlockNum, tilingData->blkProcessNum, tilingData->lenN,
            tilingData->lenC, tilingData->inH, tilingData->inW, tilingData->outH, tilingData->outW, tilingData->scaleH,
            tilingData->scaleW);
}

ge::graphStatus UpsampleNearestExact2dGradRegbaseTiling::Init()
{
    fe::PlatFormInfos* platformInfoPtr = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(tilingContext, "coreNum must greater than zero, but is %d", coreNum),
                return ge::GRAPH_FAILED);
    baseTiling.coreNum = coreNum;
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0UL, OP_LOGE(tilingContext, "UbSize must greater than zero, but is %lu", ubSize),
                return ge::GRAPH_FAILED);
    OP_LOGI(tilingContext, "coreNum is %ld, ubSize is %lu", coreNum, ubSize);
    baseTiling.ubSize = static_cast<int32_t>(ubSize);
    if (tilingData == nullptr) {
        tilingData = tilingContext->GetTilingData<UpsampleNearestExact2dGradRegBaseTilingData>();
        OP_CHECK_IF(tilingData == nullptr, OP_LOGE(tilingContext, "Get tilingdata ptr failed"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData, sizeof(UpsampleNearestExact2dGradRegBaseTilingData), 0,
                          sizeof(UpsampleNearestExact2dGradRegBaseTilingData)) != EOK),
                OP_LOGE(tilingContext, "Memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearestExact2dGradRegbaseTiling::DoTiling()
{
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "CheckInputParams is failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShapeAndAttr() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "CheckInputShapes is failed"),
                return ge::GRAPH_FAILED);
    CalTilingData();
    FillTilingData();
    PrintTilingData();
    uint64_t schId = static_cast<uint64_t>(baseTiling.schId);
    uint64_t isUint32 = static_cast<uint64_t>(baseTiling.isUint32);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, isUint32);
    OP_LOGI(tilingContext, "tilingKey %lu, schId %lu, isUint32 %lu, realCoreNum %d", tilingKey, schId, isUint32,
            baseTiling.realCoreNum);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(baseTiling.realCoreNum);

    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4UpsampleNearestExact2dGradRegbase(gert::TilingContext* context)
{
    UpsampleNearestExact2dGradRegbaseTiling tilingImpl = UpsampleNearestExact2dGradRegbaseTiling(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4UpsampleNearestExact2dGradRegbase init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling4UpsampleNearestExact2dGradRegbase do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context, "End Tiling4UpsampleNearestExact2dGradRegbase");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
