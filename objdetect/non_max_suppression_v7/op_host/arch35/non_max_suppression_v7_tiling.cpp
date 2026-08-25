/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../op_kernel/arch35/non_max_suppression_v7_tiling_data.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t kBoxesIndex = 0;
constexpr size_t kScoresIndex = 1;
constexpr size_t kMaxOutputSizeIndex = 2;
constexpr size_t kIouThresholdIndex = 3;
constexpr size_t kScoreThresholdIndex = 4;
constexpr size_t kIndexIdIndex = 5;
constexpr size_t kSelectedIndicesIndex = 0;
constexpr size_t kCenterPointBoxAttrIndex = 0;
constexpr size_t kMaxBoxesSizeAttrIndex = 1;
constexpr uint64_t kWorkspaceAlignment = 32;
constexpr int64_t kTileAlignment = 64;
constexpr int64_t kMaxTileSize = 4096;
constexpr uint64_t kSimtDataCacheReserveBytes = 32 * 1024;
constexpr uint64_t kUbReserveBytes = 16 * 1024;
constexpr uint64_t kFloatFieldsInUb = 11;
constexpr uint64_t kScratchFloatFieldCount = 6;
constexpr uint32_t kScheduleMode = 1;
constexpr int64_t kSmallBoxesThreshold = 16;

bool UseSerialPath(int64_t boxes, int64_t outputCapacity)
{
    return outputCapacity == 0 || boxes <= kSmallBoxesThreshold || (boxes <= 24 && outputCapacity <= 16) ||
           (boxes <= 40 && outputCapacity <= 4) || (boxes <= 64 && outputCapacity <= 2);
}

bool AddAligned(uint64_t offset, uint64_t size, uint64_t& nextOffset)
{
    if (offset > std::numeric_limits<uint64_t>::max() - size) {
        return false;
    }
    const uint64_t end = offset + size;
    if (end > std::numeric_limits<uint64_t>::max() - (kWorkspaceAlignment - 1)) {
        return false;
    }
    nextOffset = (end + kWorkspaceAlignment - 1) / kWorkspaceAlignment * kWorkspaceAlignment;
    return true;
}

bool IsPositiveShape(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

bool IsScalarOrOneElement(const gert::Shape& shape)
{
    return shape.GetDimNum() == 0 || (shape.GetDimNum() == 1 && shape.GetDim(0) == 1);
}

bool IsKnownEqual(int64_t lhs, int64_t rhs) { return lhs == ge::UNKNOWN_DIM || rhs == ge::UNKNOWN_DIM || lhs == rhs; }

bool MulFitsInt64(int64_t lhs, int64_t rhs)
{
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    return lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs;
}

bool CheckOptionalScalar(gert::TilingContext* context, size_t index, ge::DataType expectedType, uint8_t& present)
{
    const auto* shape = context->GetOptionalInputShape(index);
    if (shape == nullptr) {
        present = 0;
        return true;
    }
    const auto* desc = context->GetOptionalInputDesc(index);
    if (desc == nullptr) {
        OP_LOGE(context, "Optional input %zu has no tensor description.", index);
        return false;
    }
    if (desc->GetDataType() != expectedType) {
        OP_LOGE(context, "Optional input %zu has an unexpected dtype.", index);
        return false;
    }
    const auto& inputShape = shape->GetStorageShape();
    if (!IsScalarOrOneElement(inputShape) || !IsPositiveShape(inputShape)) {
        OP_LOGE(context, "Optional input %zu must be a scalar or have one element.", index);
        return false;
    }
    present = 1;
    return true;
}

bool CheckIndexInput(gert::TilingContext* context, int64_t batch, int64_t classes, int64_t boxes, uint8_t& present,
                     uint8_t& width)
{
    const auto* shape = context->GetOptionalInputShape(kIndexIdIndex);
    if (shape == nullptr) {
        present = 0;
        width = 0;
        return true;
    }
    const auto* desc = context->GetOptionalInputDesc(kIndexIdIndex);
    if (desc == nullptr || desc->GetDataType() != ge::DT_FLOAT16) {
        OP_LOGE(context, "index_id must have dtype float16.");
        return false;
    }
    const auto& indexShape = shape->GetStorageShape();
    if (!IsPositiveShape(indexShape) || indexShape.GetDimNum() != 4) {
        OP_LOGE(context, "index_id must have shape [B, C, N, 3] or [B, C, N, 4].");
        return false;
    }
    if (!IsKnownEqual(indexShape.GetDim(0), batch) || !IsKnownEqual(indexShape.GetDim(1), classes) ||
        !IsKnownEqual(indexShape.GetDim(2), boxes)) {
        OP_LOGE(context, "index_id leading dimensions must match boxes and scores.");
        return false;
    }
    const int64_t indexWidth = indexShape.GetDim(3);
    if (indexWidth != 3 && indexWidth != 4) {
        OP_LOGE(context, "index_id last dimension must be 3 or 4.");
        return false;
    }
    present = 1;
    width = static_cast<uint8_t>(indexWidth);
    return true;
}

} // namespace

namespace optiling {
struct NonMaxSuppressionV7CompileInfo {
    uint32_t coreNum{0};
    uint64_t ubSize{0};
};

static ge::graphStatus TilingParseForNonMaxSuppressionV7(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto* compileInfo = context->GetCompiledInfo<NonMaxSuppressionV7CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->coreNum == 0 || compileInfo->ubSize == 0,
                OP_LOGE(context, "Ascend950 platform resource query failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const auto* boxesInput = context->GetInputShape(kBoxesIndex);
    const auto* scoresInput = context->GetInputShape(kScoresIndex);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto& boxesShape = boxesInput->GetStorageShape();
    const auto& scoresShape = scoresInput->GetStorageShape();
    OP_CHECK_IF(boxesShape.GetDimNum() != 3 || scoresShape.GetDimNum() != 3,
                OP_LOGE(context, "boxes must be [B, N, 4] and scores must be [B, C, N]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsPositiveShape(boxesShape) || !IsPositiveShape(scoresShape),
                OP_LOGE(context, "NonMaxSuppressionV7 does not support unknown or non-positive runtime dimensions."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxesShape.GetDim(2) != 4 || boxesShape.GetDim(0) != scoresShape.GetDim(0) ||
                    boxesShape.GetDim(1) != scoresShape.GetDim(2),
                OP_LOGE(context, "boxes and scores shapes are incompatible."), return ge::GRAPH_FAILED);

    const auto* boxesDesc = context->GetInputDesc(kBoxesIndex);
    const auto* scoresDesc = context->GetInputDesc(kScoresIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresDesc);
    OP_CHECK_IF((boxesDesc->GetDataType() != ge::DT_FLOAT16 && boxesDesc->GetDataType() != ge::DT_FLOAT) ||
                    (scoresDesc->GetDataType() != ge::DT_FLOAT16 && scoresDesc->GetDataType() != ge::DT_FLOAT),
                OP_LOGE(context, "boxes and scores must be float16 or float32."), return ge::GRAPH_FAILED);

    const int64_t batch = boxesShape.GetDim(0);
    const int64_t boxes = boxesShape.GetDim(1);
    const int64_t classes = scoresShape.GetDim(1);
    OP_CHECK_IF(batch > std::numeric_limits<int32_t>::max() || boxes > std::numeric_limits<int32_t>::max() ||
                    classes > std::numeric_limits<int32_t>::max(),
                OP_LOGE(context, "Input dimensions exceed the int32 index range."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!MulFitsInt64(batch, classes) || !MulFitsInt64(batch * classes, boxes),
                OP_LOGE(context, "Input shape product overflows int64."), return ge::GRAPH_FAILED);

    const int64_t* centerPointBox = attrs->GetAttrPointer<int64_t>(kCenterPointBoxAttrIndex);
    const int64_t* maxBoxesSize = attrs->GetAttrPointer<int64_t>(kMaxBoxesSizeAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, centerPointBox);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxBoxesSize);
    OP_CHECK_IF((*centerPointBox != 0 && *centerPointBox != 1), OP_LOGE(context, "center_point_box must be 0 or 1."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*maxBoxesSize < 0 || *maxBoxesSize > std::numeric_limits<int32_t>::max(),
                OP_LOGE(context, "max_boxes_size must be in [0, INT32_MAX]."), return ge::GRAPH_FAILED);

    const auto* outputShape = context->GetOutputShape(kSelectedIndicesIndex);
    const auto* outputDesc = context->GetOutputDesc(kSelectedIndicesIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    const auto& selectedShape = outputShape->GetStorageShape();
    OP_CHECK_IF(selectedShape.GetDimNum() != 2 || selectedShape.GetDim(0) != *maxBoxesSize ||
                    selectedShape.GetDim(1) != 3 || outputDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(context, "selected_indices must have shape [max_boxes_size, 3] and dtype int32."),
                return ge::GRAPH_FAILED);

    uint8_t hasMax = 0;
    uint8_t hasIou = 0;
    uint8_t hasScore = 0;
    uint8_t hasIndex = 0;
    uint8_t indexWidth = 0;
    OP_CHECK_IF(!CheckOptionalScalar(context, kMaxOutputSizeIndex, ge::DT_INT32, hasMax) ||
                    !CheckOptionalScalar(context, kIouThresholdIndex, ge::DT_FLOAT, hasIou) ||
                    !CheckOptionalScalar(context, kScoreThresholdIndex, ge::DT_FLOAT, hasScore) ||
                    !CheckIndexInput(context, batch, classes, boxes, hasIndex, indexWidth),
                OP_LOGE(context, "Invalid NonMaxSuppressionV7 optional input."), return ge::GRAPH_FAILED);
    auto* tiling = context->GetTilingData<NonMaxSuppressionV7TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const int64_t taskCount = batch * classes;
    const auto* compileInfo = static_cast<const NonMaxSuppressionV7CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_CHECK_IF(compileInfo->coreNum == 0 || compileInfo->ubSize <= kSimtDataCacheReserveBytes + kUbReserveBytes,
                OP_LOGE(context, "Invalid Ascend950 AIV resources."), return ge::GRAPH_FAILED);
    const uint64_t localMemorySize = compileInfo->ubSize - kSimtDataCacheReserveBytes;
    OP_CHECK_IF(localMemorySize > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "Local memory size exceeds the uint32 range."), return ge::GRAPH_FAILED);
    const int64_t parallelCoreNum = taskCount < static_cast<int64_t>(compileInfo->coreNum) ?
                                        taskCount :
                                        static_cast<int64_t>(compileInfo->coreNum);
    const bool serialPath = UseSerialPath(boxes, *maxBoxesSize);
    const int64_t usedCoreNum = serialPath ? 1 : parallelCoreNum;
    const uint64_t ubBytesPerElement = kFloatFieldsInUb * sizeof(float) + sizeof(uint8_t);
    int64_t tileSize = boxes < kMaxTileSize ? boxes : kMaxTileSize;
    const int64_t ubTileLimit = static_cast<int64_t>((localMemorySize - kUbReserveBytes) / ubBytesPerElement);
    if (tileSize > ubTileLimit) {
        tileSize = ubTileLimit;
    }
    if (tileSize >= kTileAlignment) {
        tileSize = tileSize / kTileAlignment * kTileAlignment;
    }
    OP_CHECK_IF(tileSize <= 0, OP_LOGE(context, "Insufficient UB for NonMaxSuppressionV7 tile."),
                return ge::GRAPH_FAILED);

    uint64_t scratchFieldStride = 0;
    uint64_t scratchBytes = 0;
    uint64_t classIndicesOffset = 0;
    uint64_t classCountsOffset = 0;
    uint64_t userWorkspaceBytes = 0;
    const uint64_t taskCountU64 = static_cast<uint64_t>(taskCount);
    const int64_t outputCapacityPerClass = *maxBoxesSize == 0 ? 0 : ((*maxBoxesSize - 1) / taskCount + 1);
    const int64_t maxOutputPerClass = outputCapacityPerClass < boxes ? outputCapacityPerClass : boxes;
    if (serialPath) {
        const uint64_t selectedBoxCount = static_cast<uint64_t>(*maxBoxesSize);
        OP_CHECK_IF(selectedBoxCount > std::numeric_limits<uint64_t>::max() / sizeof(int32_t),
                    OP_LOGE(context, "Serial selected-box workspace size overflows uint64."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(!AddAligned(0, selectedBoxCount * sizeof(int32_t), userWorkspaceBytes),
                    OP_LOGE(context, "Serial workspace layout overflows uint64."), return ge::GRAPH_FAILED);
        classCountsOffset = userWorkspaceBytes;
    } else {
        const uint64_t boxesU64 = static_cast<uint64_t>(boxes);
        OP_CHECK_IF(boxesU64 > std::numeric_limits<uint64_t>::max() / sizeof(float),
                    OP_LOGE(context, "Scratch field size overflows uint64."), return ge::GRAPH_FAILED);
        const uint64_t scratchFieldBytes = boxesU64 * sizeof(float);
        OP_CHECK_IF(!AddAligned(0, scratchFieldBytes, scratchFieldStride),
                    OP_LOGE(context, "Scratch field alignment overflows uint64."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(scratchFieldStride > std::numeric_limits<uint64_t>::max() / kScratchFloatFieldCount,
                    OP_LOGE(context, "Scratch task size overflows uint64."), return ge::GRAPH_FAILED);
        const uint64_t scratchBytesPerTask = scratchFieldStride * kScratchFloatFieldCount;
        OP_CHECK_IF(taskCountU64 > std::numeric_limits<uint64_t>::max() / scratchBytesPerTask,
                    OP_LOGE(context, "Scratch workspace size overflows uint64."), return ge::GRAPH_FAILED);
        scratchBytes = taskCountU64 * scratchBytesPerTask;
        classIndicesOffset = scratchBytes;

        const uint64_t maxOutputPerClassU64 = static_cast<uint64_t>(maxOutputPerClass);
        OP_CHECK_IF(
            maxOutputPerClassU64 != 0 && taskCountU64 > std::numeric_limits<uint64_t>::max() / maxOutputPerClassU64,
            OP_LOGE(context, "Class result count overflows uint64."), return ge::GRAPH_FAILED);
        const uint64_t classResultCount = taskCountU64 * maxOutputPerClassU64;
        OP_CHECK_IF(classResultCount > std::numeric_limits<uint64_t>::max() / sizeof(int32_t) ||
                        taskCountU64 > std::numeric_limits<uint64_t>::max() / sizeof(int32_t),
                    OP_LOGE(context, "Class result workspace size overflows uint64."), return ge::GRAPH_FAILED);
        const uint64_t classIndicesBytes = classResultCount * sizeof(int32_t);
        const uint64_t classCountsBytes = taskCountU64 * sizeof(int32_t);
        OP_CHECK_IF(!AddAligned(classIndicesOffset, classIndicesBytes, classCountsOffset) ||
                        !AddAligned(classCountsOffset, classCountsBytes, userWorkspaceBytes),
                    OP_LOGE(context, "User workspace layout overflows uint64."), return ge::GRAPH_FAILED);
    }

    tiling->batch = batch;
    tiling->classes = classes;
    tiling->boxes = boxes;
    tiling->maxOutputSize = *maxBoxesSize;
    tiling->maxOutputPerClass = maxOutputPerClass;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tileSize = tileSize;
    tiling->reduceBufferSize = tileSize;
    tiling->scratchFieldStride = scratchFieldStride;
    tiling->classIndicesOffset = classIndicesOffset;
    tiling->classCountsOffset = classCountsOffset;
    tiling->iouThreshold = 0.0F;
    tiling->scoreThreshold = 0.0F;
    tiling->centerPointBox = static_cast<uint8_t>(*centerPointBox);
    tiling->hasMax = hasMax;
    tiling->hasIou = hasIou;
    tiling->hasScore = hasScore;
    tiling->hasIndex = hasIndex;
    tiling->indexWidth = indexWidth;

    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    const size_t systemWorkspace = platform_ascendc::PlatformAscendC(platformInfo).GetLibApiWorkSpaceSize();
    OP_CHECK_IF(static_cast<uint64_t>(*maxBoxesSize) > std::numeric_limits<size_t>::max() / sizeof(int32_t),
                OP_LOGE(context, "selected box workspace size overflows size_t."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(userWorkspaceBytes > std::numeric_limits<size_t>::max() ||
                    systemWorkspace > std::numeric_limits<size_t>::max() - static_cast<size_t>(userWorkspaceBytes),
                OP_LOGE(context, "workspace size overflows size_t."), return ge::GRAPH_FAILED);
    workspace[0] = systemWorkspace + static_cast<size_t>(userWorkspaceBytes);

    context->SetBlockDim(usedCoreNum);
    context->SetScheduleMode(kScheduleMode);
    OP_CHECK_IF(context->SetLocalMemorySize(static_cast<uint32_t>(localMemorySize)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to reserve SIMT data cache."), return ge::GRAPH_FAILED);
    context->SetTilingKey(0);
    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    rawTilingData->SetDataSize(sizeof(*tiling));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NonMaxSuppressionV7)
    .Tiling(Tiling)
    .TilingParse<NonMaxSuppressionV7CompileInfo>(TilingParseForNonMaxSuppressionV7);
} // namespace optiling
