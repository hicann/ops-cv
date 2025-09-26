/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_nearest3d_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "upsample_nearest3d_tiling.h"

namespace optiling {
constexpr int64_t BEST_PERFORMANCE_SIZE_1 = 192;
constexpr int64_t BEST_PERFORMANCE_SIZE_2 = 768;
constexpr int64_t BEST_PERFORMANCE_SIZE_3 = 1536;
constexpr int64_t BEST_PERFORMANCE_SIZE_4 = 2048;

constexpr float BEST_PERFORMANCE_SCALE_1 = 100.0f;
constexpr float BEST_PERFORMANCE_SCALE_2 = 24.0f;
constexpr float BEST_PERFORMANCE_SCALE_3 = 10.0f;
constexpr float BEST_PERFORMANCE_SCALE_4 = 6.0f;

constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

constexpr uint8_t RESERVED_LENGTH = 4;

constexpr uint8_t HALF_TYPE = 1;
constexpr uint8_t FLOAT_TYPE = 2;
constexpr uint8_t BFLOAT_TYPE = 3;

constexpr uint8_t BATCH_DIM = 2;
constexpr uint8_t DIM = 3;
constexpr uint8_t D_INDEX = 0;
constexpr uint8_t H_INDEX = 1;
constexpr uint8_t W_INDEX = 2;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;

class UpsampleNearest3dTiling {
public:
    explicit UpsampleNearest3dTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus Init() const;
    ge::graphStatus RunBigKernelTiling();

private:
    void SetScale();
    inline float ComputeScaleValue(int64_t inSize, int64_t outSize, const float *scale) const;
    void GetShapes();
    void GetSlideSize();
    uint8_t GetDataTypeVal() const;
    void GetNeedCoreNum(int64_t coreNumPlatform);
    void FillTilingData();
    inline bool CheckMaxSizes(const gert::TilingContext *context);

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    ge::DataType dataType = ge::DT_UNDEFINED;

    gert::TilingContext *tilingContext = nullptr;
    gert::Shape inputShape;
    const gert::ContinuousVector *outputSize = nullptr;
    const float *scaleD = nullptr;
    const float *scaleH = nullptr;
    const float *scaleW = nullptr;

    float realScaleW = ONE_FLOAT;
    float realScaleH = ONE_FLOAT;
    float realScaleD = ONE_FLOAT;

    int64_t batches = 0;
    int64_t outputShapes[3] = {0};
    int64_t inputShapes[3] = {0};

    int64_t needCoreNum = 0;

    int64_t slideSizeW = BEST_PERFORMANCE_SIZE_1;
    UpsampleNearest3dTilingData tilingData;
};

inline bool FloatEqual(float a, float b)
{
    float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

ge::graphStatus UpsampleNearest3dTiling::Init() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearest3dTiling::RunBigKernelTiling()
{
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入的参数
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    size_t idx = 0;
    outputSize = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    scaleD = attrs->GetAttrPointer<float>(idx++);
    scaleH = attrs->GetAttrPointer<float>(idx++);
    scaleW = attrs->GetAttrPointer<float>(idx++);

    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dataType = tilingContext->GetInputDesc(0)->GetDataType();

    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();

    GetShapes();
    if (!CheckMaxSizes(tilingContext)) {
        return ge::GRAPH_FAILED;
    }
    SetScale();
    GetSlideSize();

    // 数据分核
    auto compileInfo = reinterpret_cast<const UpsampleNearest3dCompileInfo *>(tilingContext->GetCompileInfo());
    int64_t coreNumPlatform = compileInfo->coreNum;
    GetNeedCoreNum(coreNumPlatform);
    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(1);
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearest3dTiling::GetShapes()
{
    const int64_t *outputSizeArray = reinterpret_cast<const int64_t *>(outputSize->GetData());
    batches = inputShape.GetDim(0) * inputShape.GetDim(1);
    for (int8_t i = 0; i < DIM; i++) {
        inputShapes[i] = inputShape.GetDim(i + BATCH_DIM);
        outputShapes[i] = outputSizeArray[i];
    }
    tilingData.set_batches(batches);
    tilingData.set_inputShapes(inputShapes);
    tilingData.set_outputShapes(outputShapes);
}

inline bool UpsampleNearest3dTiling::CheckMaxSizes(const gert::TilingContext *context)
{
    OP_CHECK_IF(inputShape.GetDim(0) > INT32_MAX,
        OP_LOGE(context->GetNodeName(),
            "The size of N should not exceed %d, but got size of N (%ld) ",
            INT32_MAX,
            inputShape.GetDim(0)),
        return false);
    OP_CHECK_IF(inputShape.GetDim(1) > INT32_MAX,
        OP_LOGE(context->GetNodeName(),
            "The size of C should not exceed %d, but got size of C (%ld) ",
            INT32_MAX,
            inputShape.GetDim(1)),
        return false);
    OP_CHECK_IF(inputShapes[0] > INT32_MAX,
        OP_LOGE(context->GetNodeName(),
            "The size of input_d should not exceed %d, but got size of input_d (%ld) ",
            INT32_MAX,
            inputShapes[0]),
        return false);
    OP_CHECK_IF(inputShapes[1] > INT32_MAX,
        OP_LOGE(context->GetNodeName(),
            "The size of input_h should not exceed %d, but got size of input_h (%ld) ",
            INT32_MAX,
            inputShapes[1]),
        return false);
    OP_CHECK_IF(inputShapes[2] > INT32_MAX,
        OP_LOGE(context->GetNodeName(),
            "The size of input_w should not exceed %d, but got size of input_w (%ld) ",
            INT32_MAX,
            inputShapes[2]),
        return false);
    OP_CHECK_IF(outputShapes[0] > INT32_MAX,
        OP_LOGE(context->GetNodeName(),
            "The size of output_d should not exceed %d, but got size of output_d (%ld) ",
            INT32_MAX,
            outputShapes[0]),
        return false);
    OP_CHECK_IF(outputShapes[1] > INT32_MAX,
        OP_LOGE(context->GetNodeName(),
            "The size of output_h should not exceed %d, but got size of output_h (%ld) ",
            INT32_MAX,
            outputShapes[1]),
        return false);
    OP_CHECK_IF(outputShapes[2] > INT32_MAX,
        OP_LOGE(context->GetNodeName(),
            "The size of output_w should not exceed %d, but got size of output_w (%ld) ",
            INT32_MAX,
            outputShapes[2]),
        return false);
    return true;
}

void UpsampleNearest3dTiling::SetScale()
{
    realScaleD = ComputeScaleValue(inputShapes[D_INDEX], outputShapes[D_INDEX], scaleD);
    realScaleH = ComputeScaleValue(inputShapes[H_INDEX], outputShapes[H_INDEX], scaleH);
    realScaleW = ComputeScaleValue(inputShapes[W_INDEX], outputShapes[W_INDEX], scaleW);
    tilingData.set_scaleD(realScaleD);
    tilingData.set_scaleH(realScaleH);
    tilingData.set_scaleW(realScaleW);
}

inline float UpsampleNearest3dTiling::ComputeScaleValue(int64_t inSize, int64_t outSize, const float *scale) const
{
    if (*scale > ZERO_FLOAT) {
        return *scale;
    } else {
        return outSize != 0 ? (static_cast<float>(inSize) / outSize) : ZERO_FLOAT;
    }
}

void UpsampleNearest3dTiling::GetSlideSize()
{
    if (realScaleW <= BEST_PERFORMANCE_SCALE_4) {
        slideSizeW = BEST_PERFORMANCE_SIZE_4;
    } else if (realScaleW <= BEST_PERFORMANCE_SCALE_3) {
        slideSizeW = BEST_PERFORMANCE_SIZE_3;
    } else if (realScaleW <= BEST_PERFORMANCE_SCALE_2) {
        slideSizeW = BEST_PERFORMANCE_SIZE_2;
    } else {
        slideSizeW = BEST_PERFORMANCE_SIZE_1;
    }
    tilingData.set_slideSizeW(slideSizeW);
}

void UpsampleNearest3dTiling::GetNeedCoreNum(int64_t coreNumPlatform)
{
    int64_t slideNumW = CeilA2B(outputShapes[W_INDEX], slideSizeW);
    int64_t tensorSizeW = Ceil(slideSizeW * std::min(realScaleW, BEST_PERFORMANCE_SCALE_1)) + RESERVED_LENGTH;

    int64_t slideNumH = outputShapes[H_INDEX];
    int64_t tensorSizeH = 1;
    if (realScaleH > ZERO_FLOAT && realScaleH < ONE_FLOAT) {
        slideNumH = inputShapes[H_INDEX];
        tensorSizeH = RESERVED_LENGTH;
    }

    int64_t slideNumD = outputShapes[D_INDEX];
    int64_t tensorSizeD = 1;
    if (realScaleD > ZERO_FLOAT && realScaleD < ONE_FLOAT) {
        slideNumD = inputShapes[D_INDEX];
        tensorSizeD = RESERVED_LENGTH;
    }

    int64_t slideNum = slideNumW * slideNumH * slideNumD;
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
    int64_t inputRow = batches;
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRow = 1;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRow = CeilA2B(inputRow, groupCoreNum);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRow, tailAvergingRow));
    }

    needCoreNum = coreNumPlatform;
    if (eachCoreSlideNum == 0 && remainder > 0) {
        needCoreNum = remainder * groupCoreNum;
    }

    tilingData.set_tensorSizeW(tensorSizeW);
    tilingData.set_tensorSizeH(tensorSizeH);
    tilingData.set_tensorSizeD(tensorSizeD);
    tilingData.set_slideNumH(slideNumH);
    tilingData.set_slideNumD(slideNumD);
    tilingData.set_eachCoreSlideNum(eachCoreSlideNum);
    tilingData.set_remainder(remainder);
    tilingData.set_tailStartSlideNum(eachCoreSlideNum * coreNumPlatform);
    tilingData.set_groupCoreNum(groupCoreNum);
    tilingData.set_inputRow(inputRow);
    tilingData.set_tailAvergingRow(tailAvergingRow);
    tilingData.set_needCoreNum(needCoreNum);
}

void UpsampleNearest3dTiling::FillTilingData()
{
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

uint8_t UpsampleNearest3dTiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return HALF_TYPE;
        case ge::DT_FLOAT:
            return FLOAT_TYPE;
        case ge::DT_BF16:
            return BFLOAT_TYPE;
        default:
            return 0;
    }
}

template <typename T1, typename T2>
inline auto UpsampleNearest3dTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleNearest3dTiling::Ceil(T1 x) const
{
    int32_t floorX = int32_t(x);
    if (FloatEqual(x, floorX)) {
        return floorX;
    }
    return floorX + 1;
}

static ge::graphStatus Tiling4UpsampleNearest3dTiling(gert::TilingContext *context)
{
    UpsampleNearest3dTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<UpsampleNearest3dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(compileInfo->coreNum <= 0,
        OP_LOGE(context->GetNodeName(), "UpsampleNearest3d GetHardwareInfo Failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleNearest3d)
    .Tiling(Tiling4UpsampleNearest3dTiling)
    .TilingParse<UpsampleNearest3dCompileInfo>(TilingPrepareTiling);

IMPL_OP_OPTILING(UpsampleNearestExact3d)
    .Tiling(Tiling4UpsampleNearest3dTiling)
    .TilingParse<UpsampleNearest3dCompileInfo>(TilingPrepareTiling);

}  // namespace optiling
