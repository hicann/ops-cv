/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/combined_non_max_suppression_tiling_data.h"

namespace optiling {
namespace {
constexpr int32_t BOXES_INDEX = 0;
constexpr int32_t SCORES_INDEX = 1;
constexpr int32_t MAX_PER_CLASS_INDEX = 2;
constexpr int32_t MAX_TOTAL_INDEX = 3;
constexpr int32_t IOU_THRESHOLD_INDEX = 4;
constexpr int32_t SCORE_THRESHOLD_INDEX = 5;
constexpr int32_t ATTR_CLIP_BOXES_INDEX = 1;
constexpr int32_t MAX_OUTPUT_SIZE = 1000;
constexpr int32_t MAX_CLASSES = 200;
constexpr int32_t MAX_NUM_BOXES = 200000;
constexpr uint64_t ALIGN_BYTES = 32;
constexpr uint32_t WORKSPACE_COUNT = 1;
constexpr int32_t SCHEDULE_MODE = 1;
constexpr uint64_t SIMT_UB_RESERVE = 32 * 1024;
constexpr uint64_t FALLBACK_LOCAL_MEMORY_SIZE = 64 * 1024;
constexpr int32_t HOT_UB_MAX_BOXES = 4096;

struct CombinedNonMaxSuppressionCompileInfo {
    int32_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
};

uint64_t AlignUp(uint64_t value) { return (value + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES; }

template <typename T>
ge::graphStatus ReadScalar(gert::TilingContext* context, int32_t index, T& value)
{
    const gert::Tensor* tensor = context->GetInputTensor(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor);
    const T* data = tensor->GetData<T>();
    OP_CHECK_IF(data == nullptr, OP_LOGE(context, "input %d scalar data is null", index), return ge::GRAPH_FAILED);
    value = data[0];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateAndFill(gert::TilingContext* context, CombinedNonMaxSuppressionTilingData& tiling,
                                uint64_t& userWorkspaceSize)
{
    const gert::StorageShape* boxesStorage = context->GetInputShape(BOXES_INDEX);
    const gert::StorageShape* scoresStorage = context->GetInputShape(SCORES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresStorage);
    const gert::Shape& boxes = boxesStorage->GetStorageShape();
    const gert::Shape& scores = scoresStorage->GetStorageShape();
    OP_CHECK_IF(boxes.GetDimNum() != 4 || scores.GetDimNum() != 3,
                OP_LOGE(context, "boxes must be 4D and scores must be 3D"), return ge::GRAPH_FAILED);

    const int64_t batch = boxes.GetDim(0);
    const int64_t numBoxes = boxes.GetDim(1);
    const int64_t boxClasses = boxes.GetDim(2);
    const int64_t classes = scores.GetDim(2);
    OP_CHECK_IF(batch <= 0 || numBoxes <= 0 || classes <= 0, OP_LOGE(context, "input dimensions must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numBoxes > MAX_NUM_BOXES || classes > MAX_CLASSES,
                OP_LOGE(context, "num_boxes must be <= %d and num_classes must be <= %d", MAX_NUM_BOXES, MAX_CLASSES),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxes.GetDim(3) != 4 || scores.GetDim(0) != batch || scores.GetDim(1) != numBoxes,
                OP_LOGE(context, "invalid boxes/scores shape relationship"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(boxClasses != 1 && boxClasses != classes,
                OP_LOGE(context, "boxes q dimension must be 1 or num_classes"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(batch > std::numeric_limits<int32_t>::max(), OP_LOGE(context, "batch exceeds int32 range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(batch > std::numeric_limits<int32_t>::max() / classes,
                OP_LOGE(context, "batch * num_classes exceeds int32 range"), return ge::GRAPH_FAILED);

    int32_t maxPerClass = 0;
    int32_t maxTotal = 0;
    float iouThreshold = 0.0F;
    float scoreThreshold = 0.0F;
    OP_CHECK_IF(ReadScalar(context, MAX_PER_CLASS_INDEX, maxPerClass) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to read max_output_size_per_class"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadScalar(context, MAX_TOTAL_INDEX, maxTotal) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to read max_total_size"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadScalar(context, IOU_THRESHOLD_INDEX, iouThreshold) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to read iou_threshold"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadScalar(context, SCORE_THRESHOLD_INDEX, scoreThreshold) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to read score_threshold"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxPerClass <= 0 || maxPerClass > MAX_OUTPUT_SIZE || maxTotal <= 0 || maxTotal > MAX_OUTPUT_SIZE,
                OP_LOGE(context, "max output sizes must be in [1, %d]", MAX_OUTPUT_SIZE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(iouThreshold < 0.0F || iouThreshold > 1.0F, OP_LOGE(context, "iou_threshold must be in [0, 1]"),
                return ge::GRAPH_FAILED);

    const gert::StorageShape* outputStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputStorage);
    const gert::Shape& output = outputStorage->GetStorageShape();
    OP_CHECK_IF(output.GetDimNum() != 3 || output.GetDim(0) != batch || output.GetDim(2) != 4,
                OP_LOGE(context, "invalid nmsed_boxes output shape"), return ge::GRAPH_FAILED);
    const int64_t outputSize = output.GetDim(1);
    OP_CHECK_IF(outputSize <= 0 || outputSize > maxTotal,
                OP_LOGE(context, "output_size must be in [1, max_total_size]"), return ge::GRAPH_FAILED);

    const auto* compileInfo = context->GetCompileInfo<CombinedNonMaxSuppressionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    const int64_t taskCount = batch * classes;
    const int32_t usedCoreNum = static_cast<int32_t>(std::min<int64_t>(compileInfo->coreNum, taskCount));
    OP_CHECK_IF(usedCoreNum <= 0, OP_LOGE(context, "used core num is 0"), return ge::GRAPH_FAILED);

    const uint64_t selectedCount = static_cast<uint64_t>(taskCount) * static_cast<uint64_t>(maxPerClass);
    uint64_t offset = 0;
    tiling.selectedScoresOffset = offset;
    offset = AlignUp(offset + selectedCount * sizeof(float));
    tiling.selectedIndicesOffset = offset;
    offset = AlignUp(offset + selectedCount * sizeof(int32_t));
    tiling.selectedCountsOffset = offset;
    offset = AlignUp(offset + static_cast<uint64_t>(taskCount) * sizeof(int32_t));
    tiling.suppressedOffset = offset;
    offset = AlignUp(offset + static_cast<uint64_t>(usedCoreNum) * static_cast<uint64_t>(numBoxes));
    userWorkspaceSize = offset;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* clipBoxes = attrs->GetBool(ATTR_CLIP_BOXES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, clipBoxes);

    tiling.batchSize = static_cast<int32_t>(batch);
    tiling.numBoxes = static_cast<int32_t>(numBoxes);
    tiling.boxClasses = static_cast<int32_t>(boxClasses);
    tiling.numClasses = static_cast<int32_t>(classes);
    tiling.maxOutputPerClass = maxPerClass;
    tiling.maxTotalSize = maxTotal;
    tiling.outputSize = static_cast<int32_t>(outputSize);
    tiling.usedCoreNum = usedCoreNum;
    tiling.clipBoxes = *clipBoxes ? 1 : 0;
    tiling.iouThreshold = iouThreshold;
    tiling.scoreThreshold = scoreThreshold;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CombinedNonMaxSuppressionTiling(gert::TilingContext* context)
{
    CombinedNonMaxSuppressionTilingData tiling{};
    uint64_t userWorkspaceSize = 0;
    OP_CHECK_IF(ValidateAndFill(context, tiling, userWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "tiling validation failed"), return ge::GRAPH_FAILED);

    auto* tilingData = context->GetTilingData<CombinedNonMaxSuppressionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = tiling;

    const auto* compileInfo = context->GetCompileInfo<CombinedNonMaxSuppressionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    size_t* workspaceSizes = context->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = static_cast<size_t>(compileInfo->sysWorkspaceSize + userWorkspaceSize);
    context->SetBlockDim(tiling.usedCoreNum);
    context->SetScheduleMode(SCHEDULE_MODE);

    // Keep at least 32 KB of UB outside the TPipe local-memory region for the
    // SIMT data cache, matching the Ascend950 Sort operator resource split.
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= SIMT_UB_RESERVE,
                OP_LOGE(context, "UB size %lu must be larger than SIMT reserve %lu", ubSize, SIMT_UB_RESERVE),
                return ge::GRAPH_FAILED);
    // The large-input fallback operates through GM and benefits directly from
    // a larger SIMT data cache. Its TPipe buffers need less than 64 KB, so keep
    // the rest of UB (192 KB on Ascend950) available to the cache.
    const uint64_t localMemorySize = tiling.numBoxes <= HOT_UB_MAX_BOXES ?
                                         ubSize - SIMT_UB_RESERVE :
                                         std::min(FALLBACK_LOCAL_MEMORY_SIZE, ubSize - SIMT_UB_RESERVE);
    const auto localMemoryStatus = context->SetLocalMemorySize(static_cast<uint32_t>(localMemorySize));
    OP_CHECK_IF(localMemoryStatus != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetLocalMemorySize failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CombinedNonMaxSuppressionTilingParse(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<CombinedNonMaxSuppressionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->coreNum = platform.GetCoreNumAiv();
    compileInfo->sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(compileInfo->coreNum <= 0, OP_LOGE(context, "failed to get vector core num"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(CombinedNonMaxSuppression)
    .InputsDataDependency({MAX_PER_CLASS_INDEX, MAX_TOTAL_INDEX, IOU_THRESHOLD_INDEX, SCORE_THRESHOLD_INDEX})
    .Tiling(CombinedNonMaxSuppressionTiling)
    .TilingParse<CombinedNonMaxSuppressionCompileInfo>(CombinedNonMaxSuppressionTilingParse);

} // namespace optiling
