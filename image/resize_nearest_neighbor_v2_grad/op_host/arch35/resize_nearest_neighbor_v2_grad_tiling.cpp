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
 * \file resize_nearest_neighbor_v2_grad.cc
 * \brief resize_nearest_neighbor_v2_grad
 */
#include "resize_nearest_neighbor_v2_grad_tiling_base.h"
#include "tiling_base/tiling_util.h"

#include "log/log.h"

namespace optiling {
constexpr int64_t DEFAULT_TILING_MODE = 100000;
constexpr int64_t MAX_ALIGN_ENLARGE_THRESHOLD1 = 120;
constexpr int64_t MAX_ALIGN_ENLARGE_THRESHOLD2 = 100;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t VECTOR_NUM = 64;
constexpr size_t NC1HWC0_LEN = 5;
constexpr size_t NCHW_LEN = 4;
constexpr int64_t INDEX_0 = 0;
constexpr int64_t INDEX_1 = 1;
constexpr int64_t INDEX_2 = 2;
constexpr int64_t INDEX_3 = 3;
constexpr int64_t INDEX_4 = 4;

enum class ResizeTilingKey : int64_t {
  TILING_N_2_N = 1,
  TILING_ONE_2_N = 2,
  TILING_N_2_ONE = 3,
};

struct ResizeClassRunParams {
    // image data type
    ge::DataType input_dtype;
    ge::Format input_format;
    int64_t input_c0;
};

struct ResizeClassTilingParamsRT {
    int64_t tiling_key;
    int64_t input_batch;
    int64_t input_c1;
    int64_t input_height;
    int64_t input_width;
    int64_t output_height;
    int64_t output_width;
    // cut core num by batch * C1
    int64_t cut_batch_c1_num;
    // cut core num by height
    int64_t cut_height_num;
    // cut core num by width
    int64_t cut_width_num;
    // aicore num by GE
    int64_t core_num;
};

bool CheckShapeValid(const gert::TilingContext* context, const gert::Shape& xShape, const gert::Shape& yShape,
                     const ge::Format& xFormat) {
    size_t dimLen = NCHW_LEN;
    if (xFormat == ge::FORMAT_NC1HWC0 || xFormat == ge::FORMAT_NCDHW || xFormat == ge::FORMAT_NDHWC) {
        dimLen = NC1HWC0_LEN;
    }
    OP_CHECK_IF(
        xShape.GetDimNum() != dimLen,
        OP_LOGE(
          context->GetNodeName(), "the input shape must be 5(NC1HWC0)/4(NCHW) but shape is %s, format:%d.",
          Ops::Base::ToString(xShape).c_str(), static_cast<int32_t>(xFormat)),
        return false);
    OP_CHECK_IF(
        yShape.GetDimNum() != dimLen,
        OP_LOGE(
          context->GetNodeName(), "the output shape must be 5(NC1HWC0)/4(NCHW) but shape is %s, format:%d.",
          Ops::Base::ToString(yShape).c_str(), static_cast<int32_t>(xFormat)),
        return false);

    return true;
}

void InitTilingData(ResizeClassTilingParamsRT* tiling_params, const gert::Shape& input_shape,
                    const gert::Shape& output_shape) {
    tiling_params->tiling_key = DEFAULT_TILING_MODE;
    tiling_params->input_batch = input_shape.GetDim(INDEX_0);
    tiling_params->input_c1 = input_shape.GetDim(INDEX_1);
    tiling_params->input_height = input_shape.GetDim(INDEX_2);
    tiling_params->output_height = output_shape.GetDim(INDEX_2);
    tiling_params->input_width = input_shape.GetDim(INDEX_3);
    tiling_params->output_width = output_shape.GetDim(INDEX_3);
    tiling_params->cut_batch_c1_num = 1;
    tiling_params->cut_height_num = 1;
    tiling_params->cut_width_num = 1;
    tiling_params->core_num = 0;
}

bool ResizeTilingWithInitData(gert::TilingContext* context, const size_t input_idx, const size_t output_idx,
                              ResizeClassRunParams& run_info) {
    OP_LOGD(context->GetNodeName(), "begin do ResizeTilingWithInitData");
    auto xShapePtr = context->GetInputShape(input_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xShape = Ops::Cv::OpTiling::EnsureNotScalar(xShapePtr->GetStorageShape());
    auto yShapePtr = context->GetOutputShape(output_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    const gert::Shape& yShape = Ops::Cv::OpTiling::EnsureNotScalar(yShapePtr->GetStorageShape());
    auto xTensorPtr = context->GetInputDesc(input_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensorPtr);
    auto xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xTensorPtr->GetStorageFormat()));
    OP_CHECK_IF(
        !CheckShapeValid(context, xShape, yShape, xFormat),
        OP_LOGE(context->GetNodeName(), "Shape is Invalid."),
        return false);

    // set ResizeClassRunParams with runtime info
    run_info.input_dtype = xTensorPtr->GetDataType();
    run_info.input_format = xFormat;
    if (xFormat == ge::FORMAT_NC1HWC0) {
        run_info.input_c0 = xShape.GetDim(NC1HWC0_LEN - 1);
    } else {
        run_info.input_c0 = 1;
    }

    // get and init tilingdata
    ResizeClassTilingParamsRT* tilingDataPtr = context->GetTilingData<ResizeClassTilingParamsRT>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingDataPtr);
    InitTilingData(tilingDataPtr, xShape, yShape);
    OP_LOGD(context->GetNodeName(), "end do ResizeTilingWithInitData");

    return true;
}

ge::graphStatus Tiling4ResizeNearestNeighborV2Grad(gert::TilingContext* context) {
  // get ResizeClassRunParams with runtime info
  static constexpr size_t inputGradsIdx = 0;
  static constexpr size_t outputYIdx = 0;
  ResizeClassRunParams runInfo;
  OP_CHECK_IF(!ResizeTilingWithInitData(context, inputGradsIdx, outputYIdx, runInfo),
                  OP_LOGE(context->GetNodeName(), "do ResizeTilingWithInitData failed!"),
                  return ge::GRAPH_FAILED);

  // get tiling data ptr
  ResizeClassTilingParamsRT* tilingDataPtr = context->GetTilingData<ResizeClassTilingParamsRT>();
  OP_CHECK_NULL_WITH_CONTEXT(context, tilingDataPtr);

  // get compile info ptr
  const ResizeNearestNeighborV2GradCompileInfo* compileInfo =
      static_cast<const ResizeNearestNeighborV2GradCompileInfo*>(context->GetCompileInfo());
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
  return ResizeNearestNeighborV2GradTilingForAscendC(context, compileInfo);
}

static ge::graphStatus TilingPrepare4ResizeNearestNeighborV2Grad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepare4ResizeNearestNeighborV2Grad.");
    auto compileInfo = context->GetCompiledInfo<ResizeNearestNeighborV2GradCompileInfo>();
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->core_num <= 0), OP_LOGE(context->GetNodeName(), "core num invalid."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "ub size invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ResizeNearestNeighborV2Grad op.
IMPL_OP_OPTILING(ResizeNearestNeighborV2Grad)
    .Tiling(Tiling4ResizeNearestNeighborV2Grad)
    .TilingParse<ResizeNearestNeighborV2GradCompileInfo>(TilingPrepare4ResizeNearestNeighborV2Grad)
    .TilingInputsDataDependency({RESIZE_NEAREST_NEIGHBOR_V2_GRAD_INPUT_DEPENDENCY_IDX});
}  // namespace optiling
