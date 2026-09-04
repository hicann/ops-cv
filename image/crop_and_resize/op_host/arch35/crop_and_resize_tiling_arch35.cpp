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
    int64_t batch = 0;
    int64_t imageHeight = 0;
    int64_t imageWidth = 0;
    int64_t depth = 0;
    int64_t numBoxes = 0;
    int32_t cropHeight = 0;
    int32_t cropWidth = 0;
    float extrapolationValue = 0.0f;
    ge::DataType xDtype = ge::DT_UNDEFINED;
    bool isNchw = false; // x format 是否为 NCHW：决定 dims→H/W/C 解析来源与 TilingKey 分流
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

// 约束 7：dtype 检查（x/boxes/box_index）
static ge::graphStatus CheckInputDtypes(gert::TilingContext* context, const CropAndResizeInputInfo& info)
{
    if (info.xDtype != ge::DT_FLOAT && info.xDtype != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(info.xDtype).c_str(),
                                  "FLOAT16/FLOAT");
        return ge::GRAPH_FAILED;
    }
    ge::DataType boxesDtype = context->GetInputDesc(IDX_BOXES)->GetDataType();
    if (boxesDtype != ge::DT_FLOAT && boxesDtype != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "boxes", Ops::Base::ToString(boxesDtype).c_str(),
                                  "FLOAT16/FLOAT");
        return ge::GRAPH_FAILED;
    }
    ge::DataType boxIndexDtype = context->GetInputDesc(IDX_BOX_INDEX)->GetDataType();
    if (boxIndexDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "box_index", Ops::Base::ToString(boxIndexDtype).c_str(),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
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
    int64_t hw = info.imageHeight * info.imageWidth;
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
        // depth 来源 dim 随 format 变化：NCHW 为 x.shape[1]，ND/NHWC 为 x.shape[3]（与 def.cpp depthDesc 对齐）
        std::string depthDesc = info.isNchw ? "x.shape[1]" : "x.shape[3]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(info.depth).c_str(),
                                              ("depth (" + depthDesc + ") must be in [" + std::to_string(DEPTH_MIN) +
                                               ", " + std::to_string(DEPTH_MAX) + "]")
                                                  .c_str());
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
    OP_CHECK_IF(CheckInputDtypes(context, info) != ge::GRAPH_SUCCESS, OP_LOGE(context, "dtype check failed"),
                return ge::GRAPH_FAILED);
    if (info.xDtype == ge::DT_FLOAT && hw > HW_FP32_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(hw).c_str(),
                                              "float32 requires H*W <= " + std::to_string(HW_FP32_MAX));
        return ge::GRAPH_FAILED; // 约束9
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSafetyConstraints(gert::TilingContext* context)
{
    auto boxesShapePtr = context->GetInputShape(IDX_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShapePtr);
    auto boxIndexShapePtr = context->GetInputShape(IDX_BOX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxIndexShapePtr);
    auto boxesShape = boxesShapePtr->GetStorageShape();
    auto boxIndexShape = boxIndexShapePtr->GetStorageShape();
    if (boxIndexShape.GetDimNum() != BOX_INDEX_DIM) {
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
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "boxes", std::to_string(boxesShape.GetDim(1)).c_str(), "boxes.shape[1] must be 4");
        return ge::GRAPH_FAILED; // 约束11
    }

    // 约束13: NaN 检查已移至 kernel 运行时（逐 box 检查 boxes 坐标，含 NaN 则填 NaN）
    // 原因：tiling 阶段在二进制/动态/常量编译模式下 boxes 数据不可用（nullptr），导致 OPTILING_FAILURE

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCropAndResize([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// 提取输入 format + shape + crop_size 值 + dtype
static ge::graphStatus ExtractInputInfo(gert::TilingContext* context, CropAndResizeInputInfo& info)
{
    auto xShapePtr = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto boxesShapePtr = context->GetInputShape(IDX_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto boxesShape = boxesShapePtr->GetStorageShape();

    // 读 x format，非 ND/NHWC/NCHW 拒绝（防御性兜底，与 infershape 拦截集合一致）
    auto xDescPtr = context->GetInputDesc(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDescPtr);
    ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDescPtr->GetStorageFormat()));
    OP_CHECK_IF(xFormat != ge::FORMAT_NCHW && xFormat != ge::FORMAT_NHWC && xFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x", Ops::Base::ToString(xFormat).c_str(),
                                           "NCHW, NHWC and ND"),
                return ge::GRAPH_FAILED);
    info.isNchw = (xFormat == ge::FORMAT_NCHW);

    if (xShape.GetDimNum() != X_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", (std::to_string(xShape.GetDimNum()) + "D").c_str(),
                                     "4D");
        return ge::GRAPH_FAILED;
    }
    info.batch = xShape.GetDim(0);
    if (info.isNchw) {
        // NCHW: x = (N, C, H, W)（对齐 TBE check_supported 双分支解析）
        info.depth = xShape.GetDim(1);       // C
        info.imageHeight = xShape.GetDim(2); // H
        info.imageWidth = xShape.GetDim(3);  // W
    } else {
        // ND/NHWC: x = (N, H, W, C)（现状解析，零回归）
        info.imageHeight = xShape.GetDim(1); // H
        info.imageWidth = xShape.GetDim(2);  // W
        info.depth = xShape.GetDim(3);       // C
    }
    if (boxesShape.GetDimNum() != BOXES_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "boxes",
                                     (std::to_string(boxesShape.GetDimNum()) + "D").c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    info.numBoxes = boxesShape.GetDim(0);
    info.xDtype = context->GetInputDesc(IDX_X)->GetDataType();

    auto cropSizeShapePtr = context->GetInputShape(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeShapePtr);
    auto cropSizeShape = cropSizeShapePtr->GetStorageShape();
    if (cropSizeShape.GetDimNum() != CROP_SIZE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "crop_size",
                                     (std::to_string(cropSizeShape.GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    if (cropSizeShape.GetDim(0) != CROP_SIZE_LEN) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "crop_size",
                                                  std::to_string(cropSizeShape.GetDim(0)).c_str(),
                                                  "crop_size.shape[0] must be 2");
        return ge::GRAPH_FAILED;
    }

    ge::DataType cropSizeDtype = context->GetInputDesc(IDX_CROP_SIZE)->GetDataType();
    if (cropSizeDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "crop_size", Ops::Base::ToString(cropSizeDtype).c_str(),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }

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
    // info.numBoxes 为 int64_t，info.cropHeight/cropWidth 为 int32_t，
    // int64_t * int32_t 运算时 int32_t 隐式提升为 int64_t，结果为 int64_t，无溢出风险
    int64_t totalPositions = info.numBoxes * info.cropHeight * info.cropWidth;
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
    // CropAndResizeTilingData 字段为 int32_t，此处赋值时 info 字段（int64_t）已通过
    // CheckTbeConstraints 和 CheckSafetyConstraints 校验，值均在 int32_t 范围内：
    //   batch: 通过 all dims must be positive 检查，实际值由框架约束在合理范围
    //   imageHeight/imageWidth: 通过 H*W <= 65530 检查，单维值远小于 INT32_MAX
    //   depth: 通过 [256, 2048] 检查
    //   numBoxes: 通过 (50, 4000] 检查
    tiling->batch = static_cast<int32_t>(info.batch);
    tiling->imageHeight = static_cast<int32_t>(info.imageHeight);
    tiling->imageWidth = static_cast<int32_t>(info.imageWidth);
    tiling->depth = static_cast<int32_t>(info.depth);
    tiling->cropHeight = info.cropHeight;
    tiling->cropWidth = info.cropWidth;
    tiling->numBoxes = static_cast<int32_t>(info.numBoxes);
    tiling->extrapolationValue = info.extrapolationValue;

    context->SetBlockDim(needCoreNum);
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE, OP_LOGE(context, "ubSize too small"),
                return ge::GRAPH_FAILED);
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    // layout 经 TilingKey 编码为 kernel 模板参数（NHWC 值 0 与原单值模式二进制兼容，ND/NHWC 路径零回归）
    // 三目结果显式 uint64_t 承接，避免花括号初始化列表 narrowing 报错
    uint64_t schMode = info.isNchw ? static_cast<uint64_t>(CROP_AND_RESIZE_MODE_BILINEAR_NCHW) :
                                     static_cast<uint64_t>(CROP_AND_RESIZE_MODE_BILINEAR_NHWC);
    context->SetTilingKey(GET_TPL_TILING_KEY(schMode));
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
    OP_CHECK_IF(CheckSafetyConstraints(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "safety constraints check failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAndSetTiling(context, info, coreNum, ubSize, sysWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeAndSetTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Tiling 注册：三段式 + 值依赖（crop_size index=3 shape推导；NaN 检查已移至 kernel 运行时）
IMPL_OP_OPTILING(CropAndResize)
    .Tiling(CropAndResizeTilingFunc)
    .TilingParse<CropAndResizeCompileInfo>(TilingParseForCropAndResize)
    .TilingInputsDataDependency({IDX_CROP_SIZE});
} // namespace optiling
