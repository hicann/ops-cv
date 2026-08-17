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
 * \file crop_and_resize_tiling_arch35.cpp
 * \brief Tiling implementation for crop_and_resize operator
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../crop_and_resize_constraints.h"
#include "image/crop_and_resize/op_kernel/arch35/crop_and_resize_tiling_data.h"
#include "image/crop_and_resize/op_kernel/arch35/crop_and_resize_tiling_key.h"

#include <string>

namespace optiling {

// tiling 专属常量（非约束阈值，不与 def/infershape 共享）
constexpr int64_t PER_CORE_MIN = 1024;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

// 约束阈值常量来自 op_host/crop_and_resize_constraints.h（与 def.cpp/infershape.cpp 共享）

struct CropAndResizeCompileInfo {};

// 输入信息结构体，用于在约束检查和 tiling 计算间传递
struct CropAndResizeInputInfo {
    int32_t batch = 0;
    int32_t imageHeight = 0;
    int32_t imageWidth = 0;
    int32_t depth = 0;
    int32_t numBoxes = 0;
    int32_t cropHeight = 0;
    int32_t cropWidth = 0;
    float extrapolationValue = 0.0f;
    ge::DataType xDtype = ge::DT_UNDEFINED;
};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum,
                                       uint64_t& sysWorkspaceSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// 约束 1-9：与 TBE check_supported 对齐
static ge::graphStatus CheckTbeConstraints(gert::TilingContext* context, const CropAndResizeInputInfo& info)
{
    if (info.batch <= 0 || info.imageHeight <= 0 || info.imageWidth <= 0 || info.depth <= 0) {
        std::string valMsg = "[" + std::to_string(info.batch) + ", " + std::to_string(info.imageHeight) + ", " +
                             std::to_string(info.imageWidth) + ", " + std::to_string(info.depth) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", valMsg.c_str(), "all dims must be positive");
        return ge::GRAPH_FAILED;
    }
    // int64 计算避免 H*W int32 乘法溢出（大维度回绕为负数绕过约束检查）
    int64_t hw = static_cast<int64_t>(info.imageHeight) * info.imageWidth;
    // 约束4 前置：crop_height/crop_width 必须 > 0，防止后续乘法/除法异常（SE §1.4/§5.4 边界条件）
    if (info.cropHeight <= 0 || info.cropWidth <= 0) {
        std::string valMsg = "[" + std::to_string(info.cropHeight) + ", " + std::to_string(info.cropWidth) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "crop_size", valMsg.c_str(),
                                              "crop_height and crop_width must be positive");
        return ge::GRAPH_FAILED;
    }
    if (info.numBoxes <= NUM_BOXES_MIN || info.numBoxes > NUM_BOXES_MAX) {
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "boxes", std::to_string(info.numBoxes).c_str(),
            ("(" + std::to_string(NUM_BOXES_MIN) + ", " + std::to_string(NUM_BOXES_MAX) + "]").c_str());
        return ge::GRAPH_FAILED; // 约束2
    }
    if (info.depth < DEPTH_MIN || info.depth > DEPTH_MAX) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "x", std::to_string(info.depth).c_str(),
                                  ("[" + std::to_string(DEPTH_MIN) + ", " + std::to_string(DEPTH_MAX) + "]").c_str());
        return ge::GRAPH_FAILED; // 约束3
    }
    if (info.cropHeight > CROP_DIM_MAX || info.cropWidth > CROP_DIM_MAX) {
        std::string valMsg = "[" + std::to_string(info.cropHeight) + ", " + std::to_string(info.cropWidth) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "crop_size", valMsg.c_str(),
                                              "max(crop_h, crop_w) must be <= " + std::to_string(CROP_DIM_MAX));
        return ge::GRAPH_FAILED; // 约束4
    }
    if (hw > HW_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(hw).c_str(),
                                              "H*W must be <= " + std::to_string(HW_MAX));
        return ge::GRAPH_FAILED; // 约束5
    }
    int64_t cropArea = static_cast<int64_t>(info.cropHeight) * info.cropWidth;
    if (cropArea > CROP_AREA_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "crop_size", std::to_string(cropArea).c_str(),
                                              "crop_h*crop_w must be <= " + std::to_string(CROP_AREA_MAX));
        return ge::GRAPH_FAILED; // 约束6
    }
    if (info.xDtype != ge::DT_FLOAT && info.xDtype != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(info.xDtype).c_str(),
                                  "FLOAT16/FLOAT");
        return ge::GRAPH_FAILED; // 约束7
    }
    if (info.xDtype == ge::DT_FLOAT && hw > HW_FP32_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(hw).c_str(),
                                              "float32 requires H*W <= " + std::to_string(HW_FP32_MAX));
        return ge::GRAPH_FAILED; // 约束9
    }
    return ge::GRAPH_SUCCESS;
}

// 约束 10-14：安全增强（TBE 未实现）
static ge::graphStatus CheckSafetyConstraints(gert::TilingContext* context, const CropAndResizeInputInfo& info)
{
    auto boxesShapePtr = context->GetInputShape(IDX_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShapePtr);
    auto boxIndexShapePtr = context->GetInputShape(IDX_BOX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxIndexShapePtr);
    auto cropSizeShapePtr = context->GetInputShape(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeShapePtr);
    auto boxesShape = boxesShapePtr->GetStorageShape();
    auto boxIndexShape = boxIndexShapePtr->GetStorageShape();
    auto cropSizeShape = cropSizeShapePtr->GetStorageShape();
    if (boxesShape.GetDimNum() != BOXES_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "boxes",
                                     (std::to_string(boxesShape.GetDimNum()) + "D").c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    if (boxIndexShape.GetDimNum() < 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "box_index",
                                     (std::to_string(boxIndexShape.GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    if (boxesShape.GetDim(0) != boxIndexShape.GetDim(0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "boxes and box_index",
                                               std::to_string(boxesShape.GetDim(0)).c_str(),
                                               "boxes.shape[0] must equal box_index.shape[0]");
        return ge::GRAPH_FAILED; // 约束10
    }
    if (boxesShape.GetDim(1) != BOX_COORDS) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "boxes", std::to_string(boxesShape.GetDim(1)).c_str(),
                                  std::to_string(BOX_COORDS).c_str());
        return ge::GRAPH_FAILED; // 约束11
    }
    if (cropSizeShape.GetDimNum() != 1 || cropSizeShape.GetDim(0) != CROP_SIZE_LEN) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "crop_size",
                                      std::to_string(cropSizeShape.GetDim(0)).c_str(),
                                      std::to_string(CROP_SIZE_LEN).c_str());
        return ge::GRAPH_FAILED; // 约束12
    }

    // 约束13: NaN 检查已移至 kernel 运行时（逐 box 检查 boxes 坐标，含 NaN 则填 NaN）
    // 原因：tiling 阶段在二进制/动态/常量编译模式下 boxes 数据不可用（nullptr），导致 OPTILING_FAILURE

    // 约束14: 输出元素数溢出检查
    int64_t outputTotal = static_cast<int64_t>(info.numBoxes) * info.cropHeight * info.cropWidth * info.depth;
    if (outputTotal > INT32_MAX) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "y", std::to_string(outputTotal).c_str(),
                                      std::to_string(INT32_MAX).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCropAndResize([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// 提取输入 shape + crop_size 值 + dtype
static ge::graphStatus ExtractInputInfo(gert::TilingContext* context, CropAndResizeInputInfo& info)
{
    auto xShapePtr = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto boxesShapePtr = context->GetInputShape(IDX_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto boxesShape = boxesShapePtr->GetStorageShape();
    if (xShape.GetDimNum() != X_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", (std::to_string(xShape.GetDimNum()) + "D").c_str(),
                                     "4D");
        return ge::GRAPH_FAILED;
    }
    info.batch = static_cast<int32_t>(xShape.GetDim(0));
    info.imageHeight = static_cast<int32_t>(xShape.GetDim(1));
    info.imageWidth = static_cast<int32_t>(xShape.GetDim(2));
    info.depth = static_cast<int32_t>(xShape.GetDim(3));
    info.numBoxes = static_cast<int32_t>(boxesShape.GetDim(0));
    info.xDtype = context->GetInputDesc(IDX_X)->GetDataType();

    const gert::Tensor* cropSizeTensor = context->GetInputTensor(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeTensor);
    const int32_t* cropSizeData = cropSizeTensor->GetData<int32_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeData);
    info.cropHeight = cropSizeData[0];
    info.cropWidth = cropSizeData[1];
    return ge::GRAPH_SUCCESS;
}

// 提取属性 extrapolation_value + method 校验
static ge::graphStatus ExtractAttrs(gert::TilingContext* context, CropAndResizeInputInfo& info)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* extrapolationValuePtr = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, extrapolationValuePtr);
    info.extrapolationValue = *extrapolationValuePtr;
    const char* methodStr = attrs->GetStr(1);
    OP_CHECK_IF(methodStr == nullptr, OP_LOGE(context, "method attr is null"), return ge::GRAPH_FAILED);
    if (std::string(methodStr) != "bilinear") {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "method", methodStr, "bilinear");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 核数切分 + 设置 TilingData + 设置 BlockDim/Workspace
static ge::graphStatus ComputeAndSetTiling(gert::TilingContext* context, const CropAndResizeInputInfo& info,
                                           int64_t coreNum, uint64_t ubSize, uint64_t sysWorkspaceSize)
{
    int64_t totalPositions = static_cast<int64_t>(info.numBoxes) * info.cropHeight * info.cropWidth;
    int64_t perCorePositions = Ops::Base::CeilDiv(totalPositions, coreNum);
    if (perCorePositions < PER_CORE_MIN) {
        perCorePositions = PER_CORE_MIN;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalPositions, perCorePositions);

    CropAndResizeTilingData* tiling = context->GetTilingData<CropAndResizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CropAndResizeTilingData), 0, sizeof(CropAndResizeTilingData)) != EOK,
                OP_LOGE(context, "memset_s tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalPositions = totalPositions;
    tiling->batch = info.batch;
    tiling->imageHeight = info.imageHeight;
    tiling->imageWidth = info.imageWidth;
    tiling->depth = info.depth;
    tiling->cropHeight = info.cropHeight;
    tiling->cropWidth = info.cropWidth;
    tiling->numBoxes = info.numBoxes;
    tiling->extrapolationValue = info.extrapolationValue;

    context->SetBlockDim(needCoreNum);
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE, OP_LOGE(context, "ubSize too small"),
                return ge::GRAPH_FAILED);
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    context->SetTilingKey(GET_TPL_TILING_KEY(CROP_AND_RESIZE_MODE_BILINEAR));
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    int64_t userWorkspaceSize = 0;
    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<int64_t>(sysWorkspaceSize));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CropAndResizeTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum, sysWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    CropAndResizeInputInfo info;
    OP_CHECK_IF(ExtractInputInfo(context, info) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ExtractInputInfo failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ExtractAttrs(context, info) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ExtractAttrs failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTbeConstraints(context, info) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TBE constraints check failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckSafetyConstraints(context, info) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "safety constraints check failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAndSetTiling(context, info, coreNum, ubSize, sysWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeAndSetTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Tiling 注册：三段式 + 值依赖（crop_size index=3 shape推导；NaN 检查已移至 kernel 运行时）
IMPL_OP_OPTILING(CropAndResize)
    .Tiling(CropAndResizeTilingFunc)
    .TilingParse<CropAndResizeCompileInfo>(TilingParseForCropAndResize)
    .TilingInputsDataDependency({3});
} // namespace optiling
