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
 * \file grid_unnormal_tiling_arch35.cpp
 * \brief GridUnnormal Tiling —— arch35 实现（纯 elementwise，按总元素数扁平分核 + UB 切分）
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/grid_unnormal_tiling_data.h"

#include <algorithm>
#include <string>

namespace optiling {

constexpr int64_t kRegElemAlign = 64;    // UB tile 元素数对齐：必须是一个 fp32 vector loop 的整数倍
constexpr int64_t kUbReserve = 8 * 1024; // 预留（栈 / 对齐余量），字节
constexpr size_t kAttrAlignCornersIdx = 0;
constexpr size_t kGridRank = 4;
constexpr size_t kLastDimIdx = 3;
constexpr int64_t kCoordDim = 2;
constexpr int64_t kInputTensorNum = 2;
constexpr int64_t kOutputDiffTensorNum = 1;
constexpr int64_t kOutputPositionTensorNum = 1;
constexpr int64_t kBufNum = 2;
static_assert(kRegElemAlign % 64 == 0, "kRegElemAlign must preserve full 64-element vector reads");

struct GridUnnormalCompileInfo {};

struct GridUnnormalPlatformInfo {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
};

struct GridUnnormalInputInfo {
    int64_t total = 0;
    int64_t dtSize = 0;
};

// 单元素 UB 占用（字节），随 grid dtype 变化（RegBase：中间量在寄存器，无 UB scratch）：
//   入队输入 + 出队输出：按 tensor 数和双缓冲数计；position 使用 int32_t 字节宽度。
static int64_t BytesPerElem(int64_t dtSize)
{
    return (kInputTensorNum + kOutputDiffTensorNum) * dtSize * kBufNum +
           kOutputPositionTensorNum * static_cast<int64_t>(sizeof(int32_t)) * kBufNum;
}

static int64_t GetDataTypeSize(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT16) {
        return 2;
    }
    if (dtype == ge::DT_FLOAT) {
        return 4;
    }
    return 0;
}

static bool IsSameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus CheckGridShape(gert::TilingContext* context, const gert::Shape& gridShape,
                                      const gert::Shape& assistShape)
{
    OP_CHECK_IF(
        gridShape.GetDimNum() != kGridRank,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid", "rank is not 4", "grid rank must be 4"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gridShape.GetDim(kLastDimIdx) != kCoordDim,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid", "last dim is not 2",
                                                       "grid last dim must be 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsSameShape(gridShape, assistShape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grid and assist", "not equal",
                                                       "storage shapes must be equal"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool ReadAlignCorners(const gert::TilingContext* context)
{
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const bool* p = attrs->GetAttrPointer<bool>(kAttrAlignCornersIdx);
    return (p == nullptr) ? false : *p;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, GridUnnormalPlatformInfo& info)
{
    fe::PlatFormInfos* pinfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, pinfo);
    auto plat = platform_ascendc::PlatformAscendC(pinfo);
    info.coreNum = plat.GetCoreNumAiv();
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    OP_CHECK_IF(info.coreNum <= 0,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "AIV core count must be > 0, coreNum=%s",
                                       std::to_string(info.coreNum).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.ubSize == 0,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "UB size must be > 0, ubSize=%s",
                                       std::to_string(info.ubSize).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputInfo(gert::TilingContext* context, GridUnnormalInputInfo& info)
{
    auto gridShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridShape);
    auto assistShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, assistShape);
    const gert::Shape gridStorageShape = gridShape->GetStorageShape();
    const gert::Shape assistStorageShape = assistShape->GetStorageShape();
    if (CheckGridShape(context, gridStorageShape, assistStorageShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    info.total = gridStorageShape.GetShapeSize();
    OP_CHECK_IF(
        info.total < 0,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "grid", std::to_string(info.total).c_str(),
                                                  "shape size must be non-negative"),
        return ge::GRAPH_FAILED);

    auto gridDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridDesc);
    auto assistDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, assistDesc);
    OP_CHECK_IF(gridDesc->GetDataType() != assistDesc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "grid and assist", "not equal",
                                                       "input dtypes must be equal"),
                return ge::GRAPH_FAILED);

    info.dtSize = GetDataTypeSize(gridDesc->GetDataType());
    OP_CHECK_IF(info.dtSize == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "grid",
                                                      std::to_string(gridDesc->GetDataType()).c_str(),
                                                      "only float16 and float32 are supported"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcUbFactor(gert::TilingContext* context, uint64_t ubSize, int64_t dtSize, int64_t& ubFactor)
{
    const int64_t bytesPerElem = BytesPerElem(dtSize);
    const int64_t usable = static_cast<int64_t>(ubSize) - kUbReserve;
    OP_CHECK_IF(
        usable < kRegElemAlign * bytesPerElem,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "UB is too small for one aligned tile, ubSize=%s, usable=%s",
                               std::to_string(ubSize).c_str(), std::to_string(usable).c_str()),
        return ge::GRAPH_FAILED);
    ubFactor = (usable / bytesPerElem / kRegElemAlign) * kRegElemAlign;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillEmptyTiling(gert::TilingContext* context, GridUnnormalTilingData* td, int64_t ubFactor,
                                       int32_t alignCorners)
{
    td->totalNum = 0;
    td->perCoreNum = 0;
    td->ubFactor = ubFactor;
    td->alignCorners = alignCorners;
    OP_CHECK_IF(context->SetBlockDim(1) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "SetBlockDim failed for empty tensor, blockDim=1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillNormalTiling(gert::TilingContext* context, GridUnnormalTilingData* td, int64_t total,
                                        int64_t coreNum, int64_t ubFactor, int32_t alignCorners)
{
    int64_t perCore = total / coreNum + (total % coreNum != 0); // 无溢出向上取整均分
    int64_t usedCores = total / perCore + (total % perCore != 0);
    usedCores = std::max<int64_t>(1, std::min<int64_t>(usedCores, coreNum));
    td->totalNum = total;
    td->perCoreNum = perCore;
    td->ubFactor = ubFactor;
    td->alignCorners = alignCorners;
    OP_CHECK_IF(context->SetBlockDim(static_cast<uint32_t>(usedCores)) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "SetBlockDim failed, blockDim=%s",
                                       std::to_string(usedCores).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GridUnnormalTilingFunc(gert::TilingContext* context)
{
    GridUnnormalPlatformInfo platformInfo;
    if (GetPlatformInfo(context, platformInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GridUnnormalInputInfo inputInfo;
    if (GetInputInfo(context, inputInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t ubFactor = 0;
    if (CalcUbFactor(context, platformInfo.ubSize, inputInfo.dtSize, ubFactor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto* td = context->GetTilingData<GridUnnormalTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    const int32_t alignCorners = ReadAlignCorners(context) ? 1 : 0;
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = 0U;
    return (inputInfo.total <= 0) ?
               FillEmptyTiling(context, td, ubFactor, alignCorners) :
               FillNormalTiling(context, td, inputInfo.total, platformInfo.coreNum, ubFactor, alignCorners);
}

static ge::graphStatus TilingParseForGridUnnormal([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GridUnnormal)
    .Tiling(GridUnnormalTilingFunc)
    .TilingParse<GridUnnormalCompileInfo>(TilingParseForGridUnnormal);

} // namespace optiling
