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
 * \file upsample_nearest3d_tiling.cpp
 * \brief
 */
#include "register/tilingdata_base.h"
#include "upsample_nearest3d_tiling_common.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "upsample_nearest3d_tiling.h"

using namespace ge;
using namespace UpsampleNearest3d;

namespace optiling {
constexpr uint8_t BATCH_DIM = 2;
constexpr uint8_t DIM = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;

class UpsampleNearest3dTiling {
public:
    explicit UpsampleNearest3dTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus Init() const;
    ge::graphStatus RunBigKernelTiling();

private:
    inline bool CheckMaxSizes(const gert::TilingContext* context);
    void GetTilingKey();

private:
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    const gert::ContinuousVector* outputSize = nullptr;
    const float* scaleD = nullptr;
    const float* scaleH = nullptr;
    const float* scaleW = nullptr;

    int64_t outputShapes[3] = {0};
    int64_t inputShapes[3] = {0};
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
    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
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

    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();

    const int64_t* outputSizeArray = reinterpret_cast<const int64_t*>(outputSize->GetData());
    for (int8_t i = 0; i < DIM; i++) {
        inputShapes[i] = inputShape.GetDim(i + BATCH_DIM);
        outputShapes[i] = outputSizeArray[i];
    }
    if (!CheckMaxSizes(tilingContext)) {
        return ge::GRAPH_FAILED;
    }

    float scales[3] = {*scaleD, *scaleH, *scaleW};
    auto compileInfo = reinterpret_cast<const UpsampleNearest3dCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    int64_t coreNumPlatform = compileInfo->coreNum;
    UpsampleNearest3d::UpsampleNearest3dTilingData* tilingData =
        tilingContext->GetTilingData<UpsampleNearest3d::UpsampleNearest3dTilingData>();
    UpsampleNearest3d::UpsampleNearest3dTiling::UpsampleNearest3dCommonTiling<gert::Shape>(
        srcShape->GetStorageShape(), scales, outputShapes, *tilingData, compileInfo->coreNum);

    GetTilingKey();
    tilingContext->SetBlockDim(tilingData->needCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

void UpsampleNearest3dTiling::GetTilingKey()
{
    uint32_t D_T_X = UPSAMPLE_NEAREST3D_TPL_FP32, D_T_Y = UPSAMPLE_NEAREST3D_TPL_FP32;
    ge::DataType dtype_x = tilingContext->GetInputDesc(0)->GetDataType();
    if (dtype_x == ge::DataType::DT_FLOAT) {
        D_T_X = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_FP32);
        D_T_Y = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_FP32);
    } else if (dtype_x == ge::DataType::DT_FLOAT16) {
        D_T_X = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_FP16);
        D_T_Y = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_FP16);
    } else if (dtype_x == ge::DataType::DT_BF16) {
        D_T_X = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_BF16);
        D_T_Y = static_cast<uint32_t>(UPSAMPLE_NEAREST3D_TPL_BF16);
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y);
    tilingContext->SetTilingKey(tilingKey);
}

inline bool UpsampleNearest3dTiling::CheckMaxSizes(const gert::TilingContext* context)
{
    OP_CHECK_IF(inputShape.GetDim(0) > INT32_MAX,
        OP_LOGE(
            context->GetNodeName(), "The size of N should not exceed %d, but got size of N (%ld) ", INT32_MAX,
            inputShape.GetDim(0)),
        return false);
    OP_CHECK_IF(inputShape.GetDim(1) > INT32_MAX,
        OP_LOGE(
            context->GetNodeName(), "The size of C should not exceed %d, but got size of C (%ld) ", INT32_MAX,
            inputShape.GetDim(1)),
        return false);
    OP_CHECK_IF(inputShapes[0] > INT32_MAX,
        OP_LOGE(
            context->GetNodeName(), "The size of input_d should not exceed %d, but got size of input_d (%ld) ",
            INT32_MAX, inputShapes[0]),
        return false);
    OP_CHECK_IF(inputShapes[1] > INT32_MAX,
        OP_LOGE(
            context->GetNodeName(), "The size of input_h should not exceed %d, but got size of input_h (%ld) ",
            INT32_MAX, inputShapes[1]),
        return false);
    OP_CHECK_IF(inputShapes[2] > INT32_MAX,
        OP_LOGE(
            context->GetNodeName(), "The size of input_w should not exceed %d, but got size of input_w (%ld) ",
            INT32_MAX, inputShapes[2]),
        return false);
    OP_CHECK_IF(outputShapes[0] > INT32_MAX,
        OP_LOGE(
            context->GetNodeName(), "The size of output_d should not exceed %d, but got size of output_d (%ld) ",
            INT32_MAX, outputShapes[0]),
        return false);
    OP_CHECK_IF(outputShapes[1] > INT32_MAX,
        OP_LOGE(
            context->GetNodeName(), "The size of output_h should not exceed %d, but got size of output_h (%ld) ",
            INT32_MAX, outputShapes[1]),
        return false);
    OP_CHECK_IF(outputShapes[2] > INT32_MAX,
        OP_LOGE(
            context->GetNodeName(), "The size of output_w should not exceed %d, but got size of output_w (%ld) ",
            INT32_MAX, outputShapes[2]),
        return false);
    return true;
}

static ge::graphStatus Tiling4UpsampleNearest3dTiling(gert::TilingContext* context)
{
    UpsampleNearest3dTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<UpsampleNearest3dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        compileInfo->coreNum <= 0, OP_LOGE(context->GetNodeName(), "UpsampleNearest3d GetHardwareInfo Failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleNearest3d)
    .Tiling(Tiling4UpsampleNearest3dTiling)
    .TilingParse<UpsampleNearest3dCompileInfo>(TilingPrepareTiling);

IMPL_OP_OPTILING(UpsampleNearestExact3d)
    .Tiling(Tiling4UpsampleNearest3dTiling)
    .TilingParse<UpsampleNearest3dCompileInfo>(TilingPrepareTiling);

} // namespace optiling
