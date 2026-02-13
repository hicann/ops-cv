/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file col2im_tiling_arch35.cpp
 * \brief
 */

#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "col2im_tiling_arch35.h"
#include "../../op_kernel/arch35/col2im_tiling_key.h"
#include "util/math_util.h"

namespace optiling {
constexpr uint32_t MAX_DIM_NUM = 8;
constexpr uint32_t INPUT_GRADOUT_IDX = 0;
constexpr uint32_t INPUT_OUTPUTSIZE_IDX = 1;
constexpr uint32_t DIM_NUM_2D = 2;
constexpr uint32_t DIM_NUM_3D = 3;
constexpr uint32_t DIM_NUM_4D = 4;
constexpr uint32_t N_DIM = 0;
constexpr uint32_t C_DIM = 1;
constexpr uint64_t ATTR_0 = 0;
constexpr uint64_t ATTR_1 = 1;
constexpr uint64_t ATTR_2 = 2;
constexpr uint64_t ATTR_3 = 3;
constexpr uint64_t ATTR_4 = 4;
constexpr uint64_t TYPE_MODE1 = 0; // 为fp32
constexpr uint64_t TYPE_MODE2 = 1; // 为fp16
constexpr uint64_t TYPE_MODE3 = 2; // 为bf16
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t FIRST_IDX_IN_VECTOR = 0;
constexpr int64_t SECOND_IDX_IN_VECTOR = 1;

template <typename T>
ge::graphStatus CopyData2Array(gert::TilingContext* context, const gert::Tensor* listTensor, int64_t listSize, int64_t dataList[])
{
    const T* listDataPtr = listTensor->GetData<T>();
    if (listDataPtr == nullptr) {
        OP_LOGE(context, "listTensor->GetData<T>() is nullptr.");
        return ge::GRAPH_FAILED;
    }

    for (int64_t i = 0; i < listSize; i++) {
        dataList[i] = static_cast<int64_t>(listDataPtr[i]);
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus GetConstInputData(
    gert::TilingContext* context, const size_t idxInput, T& dataList, int64_t& dataListLength)
{
    auto listTensor = context->GetInputTensor(idxInput);
    if (listTensor == nullptr) {
        OP_LOGE(context, "listTensor is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto inputSizeDesc = context->GetInputDesc(INPUT_OUTPUTSIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeDesc);
    auto listDataType = inputSizeDesc->GetDataType();
    auto inputSizeShapePtr = context->GetInputShape(INPUT_OUTPUTSIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSizeShapePtr);
    auto inputSizeShape = inputSizeShapePtr->GetStorageShape();
    int64_t listSize = static_cast<int64_t>(inputSizeShape.GetDim(0));
    dataListLength = listSize;

    if (listDataType == ge::DT_INT32) {
        return CopyData2Array<int32_t>(context, listTensor, listSize, dataList);
    }
    if (listDataType == ge::DT_INT64) {
        return CopyData2Array<int64_t>(context, listTensor, listSize, dataList);
    }

    return ge::GRAPH_FAILED;
}

void Col2imTiling::SetTilingData()
{
    useCoreNum_ = static_cast<int64_t>(Ops::Base::CeilDiv(totalLength_, Ops::Base::CeilDiv(totalLength_, totalCoreNum_)));
    context_->SetBlockDim(useCoreNum_);
    tilingData_->totalLength = totalLength_;
    tilingData_->outputSizeH = outputSizeH_;
    tilingData_->outputSizeW = outputSizeW_;
    tilingData_->kernelSizeH = kernelSizeH_;
    tilingData_->kernelSizeW = kernelSizeW_;
    tilingData_->dilationH = dilationH_;
    tilingData_->dilationW = dilationW_;
    tilingData_->paddingH = paddingH_;
    tilingData_->paddingW = paddingW_;
    tilingData_->strideH = strideH_;
    tilingData_->strideW = strideW_;
    tilingData_->colH = colH_;
    tilingData_->colW = colW_;
}

ge::graphStatus Col2imTiling::SetAttrParams()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const auto kernelSizePtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSizePtr);
    OP_CHECK_IF(
        kernelSizePtr->GetSize() != DIM_NUM_2D, 
        OP_LOGE(context_, "The dim of kernelSize shoule be 2."),
        return ge::GRAPH_FAILED);
    kernelSizeH_ = static_cast<int64_t>((reinterpret_cast<const int64_t *>(kernelSizePtr->GetData()))[FIRST_IDX_IN_VECTOR]);
    kernelSizeW_ = static_cast<int64_t>((reinterpret_cast<const int64_t *>(kernelSizePtr->GetData()))[SECOND_IDX_IN_VECTOR]);
    OP_CHECK_IF(
        kernelSizeH_ <= 0,
        OP_LOGE(context_, "kernelSizeH must be greater than 0, but this is %ld.", kernelSizeH_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        kernelSizeW_ <= 0,
        OP_LOGE(context_, "kernelSizeW must be greater than 0, but this is %ld.", kernelSizeW_),
        return ge::GRAPH_FAILED);

    const auto dilationPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dilationPtr);
    OP_CHECK_IF(
        dilationPtr->GetSize() != DIM_NUM_2D, 
        OP_LOGE(context_, "The dim of dilation shoule be 2."),
        return ge::GRAPH_FAILED);
    dilationH_ = static_cast<int64_t>((reinterpret_cast<const int64_t *>(dilationPtr->GetData()))[FIRST_IDX_IN_VECTOR]);
    dilationW_ = static_cast<int64_t>((reinterpret_cast<const int64_t *>(dilationPtr->GetData()))[SECOND_IDX_IN_VECTOR]);
    OP_CHECK_IF(
        dilationH_ <= 0,
        OP_LOGE(context_, "dilationH must be greater than 0, but this is %ld.", dilationH_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dilationW_ <= 0,
        OP_LOGE(context_, "dilationW must be greater than 0, but this is %ld.", dilationW_),
        return ge::GRAPH_FAILED);
    const auto paddingPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingPtr);
    OP_CHECK_IF(
        paddingPtr->GetSize() != DIM_NUM_2D, 
        OP_LOGE(context_, "The dim of padding shoule be 2."),
        return ge::GRAPH_FAILED);
    paddingH_ = static_cast<int64_t>((reinterpret_cast<const int64_t *>(paddingPtr->GetData()))[FIRST_IDX_IN_VECTOR]);
    paddingW_ = static_cast<int64_t>((reinterpret_cast<const int64_t *>(paddingPtr->GetData()))[SECOND_IDX_IN_VECTOR]);
    OP_CHECK_IF(
        paddingH_ < 0,
        OP_LOGE(context_, "paddingH must be greater or equal than 0, but this is %ld.", paddingH_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        paddingW_ < 0,
        OP_LOGE(context_, "paddingW must be greater or equal than 0, but this is %ld.", paddingW_),
        return ge::GRAPH_FAILED);
    const auto stridePtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stridePtr);
    OP_CHECK_IF(
        stridePtr->GetSize() != DIM_NUM_2D, 
        OP_LOGE(context_, "The dim of stride shoule be 2."),
        return ge::GRAPH_FAILED);
    strideH_ = static_cast<int64_t>((reinterpret_cast<const int64_t *>(stridePtr->GetData()))[FIRST_IDX_IN_VECTOR]);
    strideW_ = static_cast<int64_t>((reinterpret_cast<const int64_t *>(stridePtr->GetData()))[SECOND_IDX_IN_VECTOR]);
    OP_CHECK_IF(
        strideH_ <= 0,
        OP_LOGE(context_, "strideH must be greater than 0, but this is %ld.", strideH_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        strideW_ <= 0,
        OP_LOGE(context_, "strideW must be greater than 0, but this is %ld.", strideW_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Col2imTiling::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        totalCoreNum_ <= 0,
        OP_LOGE(context_, "Failed to core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= 0UL,
        OP_LOGE(context_, "ubSize must greater than zero, but is %lu", ubSize),
        return ge::GRAPH_FAILED);
    auto localMemorySize = context_->SetLocalMemorySize(ubSize - DCACHE_SIZE);
    OP_LOGD(context_, "ubSize = %lu, localMemorySize = %d.", ubSize, localMemorySize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Col2imTiling::InitTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<Col2imRegBaseTilingData>();
        OP_CHECK_IF(
            tilingData_ == nullptr,
            OP_LOGE(context_, "get tilingdata ptr failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        (memset_s(
             tilingData_, sizeof(Col2imRegBaseTilingData), 0, sizeof(Col2imRegBaseTilingData)) !=
         EOK),
        OP_LOGE(context_, "memset tilingdata failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool Col2imTiling::GetDataTypeKey(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dataTypeTilingKey_ = TYPE_MODE2;
            break;
        case ge::DT_BF16:
            dataTypeTilingKey_ = TYPE_MODE3;
            break;
        case ge::DT_FLOAT:
            dataTypeTilingKey_ = TYPE_MODE1;
            break;
        default:
            return false;
    }

    return true;
}

void Col2imTiling::SetTilingKey()
{
    OP_LOGI(
        context_, "dtype is %lu", dataTypeTilingKey_);
    tilingKey_ = GET_TPL_TILING_KEY(dataTypeTilingKey_);
}

uint64_t Col2imTiling::GetTilingKey()
{
    return tilingKey_;
}

ge::graphStatus Col2imTiling::GetInputTensorInfo()
{
    auto gradOut = context_->GetDynamicInputTensor(INPUT_GRADOUT_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradOut);
    auto &gradOutShape = gradOut->GetOriginShape();
    auto gradOutDimNum = gradOutShape.GetDimNum();
    OP_CHECK_IF(
        gradOutDimNum != DIM_NUM_4D,
        OP_LOGE(context_, "The dim num of gradOut shoule be 4, but this is %ld.", gradOutDimNum),
        return ge::GRAPH_FAILED);

    // 获取outputSize
    int64_t outputSizeArray[MAX_DIM_NUM] = {0};
    int64_t outputSizeLength = 0;

    if (GetConstInputData(context_, INPUT_OUTPUTSIZE_IDX, outputSizeArray, outputSizeLength) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "GetConstInputData outputSize failed.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        outputSizeLength != 2,
        OP_LOGE(context_, "outputSizeLength must be 2, but this is %ld.", outputSizeLength),
        return ge::GRAPH_FAILED);
    outputSizeH_ = static_cast<int64_t>(outputSizeArray[FIRST_IDX_IN_VECTOR]);
    outputSizeW_ = static_cast<int64_t>(outputSizeArray[SECOND_IDX_IN_VECTOR]);
    OP_CHECK_IF(
        outputSizeH_ <= 0,
        OP_LOGE(context_, "outputSizeH must be greater than 0, but this is %ld.", outputSizeH_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputSizeW_ <= 0,
        OP_LOGE(context_, "outputSizeW must be greater than 0, but this is %ld.", outputSizeW_),
        return ge::GRAPH_FAILED);

    // 获取所需处理的数据量totalLength以及colH、colW
    totalLength_ = 1;
    totalLength_ *= outputSizeH_ * outputSizeW_;
    OP_CHECK_IF(
        totalLength_ > INT32_MAX,
        OP_LOGE(context_, "outputSizeH*outputSizeW must be less or equal than 2^31-1."),
        return ge::GRAPH_FAILED);
    totalLength_ *= gradOutShape[N_DIM];
    totalLength_ *= gradOutShape[C_DIM];
    OP_CHECK_IF(
        totalLength_ > INT32_MAX,
        OP_LOGE(context_, "N*C*outputSizeH*outputSizeW must be less or equal than 2^31-1."),
        return ge::GRAPH_FAILED);

    colH_ = (outputSizeH_ + 2 * paddingH_ - dilationH_ * (kernelSizeH_ - 1) - 1) / strideH_ + 1;
    colW_ = (outputSizeW_ + 2 * paddingW_ - dilationW_ * (kernelSizeW_ - 1) - 1) / strideW_ + 1;

    auto gradOutDesc = context_->GetInputDesc(INPUT_GRADOUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradOutDesc);
    gardInDType_ = gradOutDesc->GetDataType();
    OP_CHECK_IF(
        GetDataTypeKey(gardInDType_) == false,
        OP_LOGE(context_, "The dtype of input gradOut must be in [float32, float16, bfloat16]."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Col2imTiling::DoTiling()
{
    OP_LOGD(context_, "Enter Col2imRegBaseTilingData DoTiling");

    OP_CHECK_IF(
        SetAttrParams() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "SetAttrParams failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetInputTensorInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetInputTensorInfo failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetPlatformInfo failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        InitTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "InitTilingData failed."), return ge::GRAPH_FAILED);
    SetTilingData();

    SetTilingKey();
    context_->SetTilingKey(GetTilingKey());

    OP_LOGI(context_, "tiling data:%s.", tilingData_->toString().c_str());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] =
        ascendcPlatform.GetLibApiWorkSpaceSize(); // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。

    OP_LOGD(context_, "End dotiling");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForCol2im(gert::TilingContext *context)
{
    OP_LOGD(context, "TilingForCol2im running begin.");

    Col2imTiling tilingObj(context);

    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepareForCol2im([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Col2im)
    .Tiling(TilingForCol2im)
    .TilingParse<Col2imCompileInfo>(TilingPrepareForCol2im);
}  // namespace optiling