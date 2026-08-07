/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batch_multi_class_non_max_suppression_tiling_arch35.h"
#include "objdetect/batch_multi_class_non_max_suppression/op_kernel/arch35/batch_multi_class_non_max_suppression_tiling_key.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <securec.h>

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kBoxesIndex = 0;
constexpr int64_t kScoresIndex = 1;
constexpr int64_t kClipWindowIndex = 2;
constexpr int64_t kNumValidBoxesIndex = 3;
constexpr int64_t kNmsedBoxesIndex = 0;
constexpr int64_t kNmsedScoresIndex = 1;
constexpr int64_t kNmsedClassesIndex = 2;
constexpr int64_t kNmsedNumIndex = 3;
constexpr int64_t kScoreThresholdAttrIndex = 0;
constexpr int64_t kIouThresholdAttrIndex = 1;
constexpr int64_t kMaxSizePerClassAttrIndex = 2;
constexpr int64_t kMaxTotalSizeAttrIndex = 3;
constexpr int64_t kChangeCoordinateFrameAttrIndex = 4;
constexpr int64_t kTransposeBoxAttrIndex = 5;
constexpr int64_t kMaxOutputSize = 1000;
constexpr uint32_t kScheduleMode = 1;
constexpr int64_t kScratchFloatFieldCount = 5;
constexpr int64_t kVectorFloatFieldCount = 10;
constexpr int64_t kTileAlignment = 64;
constexpr int64_t kMaxTileSize = 4096;
// Leave 32 KiB of the MIX_AIV UB budget outside AIV tile and TopK planning
// for the SIMT side of this kernel.
constexpr uint64_t kSimtUbReserveBytes = 32 * 1024;
constexpr uint64_t kWorkspaceAlignment = 32;
constexpr int64_t kTopKAlignment = 32;
// Keep the cross-class TopK input comfortably below the Ascend950 UB budget.
// Larger candidate lists are merged incrementally in the kernel.
constexpr int64_t kMergeTopKInputCapacity = 2048;

bool AddAligned(uint64_t offset, uint64_t size, uint64_t& nextOffset)
{
    if (offset > std::numeric_limits<uint64_t>::max() - size) {
        return false;
    }
    const uint64_t endOffset = offset + size;
    if (endOffset > std::numeric_limits<uint64_t>::max() - (kWorkspaceAlignment - 1)) {
        return false;
    }
    nextOffset = (endOffset + kWorkspaceAlignment - 1) / kWorkspaceAlignment * kWorkspaceAlignment;
    return true;
}

bool ProductFitsUint32(uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3)
{
    constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();
    uint64_t product = dim0;
    if (product > kUint32Max / dim1) {
        return false;
    }
    product *= dim1;
    if (product > kUint32Max / dim2) {
        return false;
    }
    product *= dim2;
    return product <= kUint32Max / dim3;
}

bool HasPositiveDims(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

struct ParsedTilingParams {
    int64_t batch{0};
    int64_t boxesNum{0};
    int64_t classesNum{0};
    int64_t boxClassesNum{0};
    int64_t maxSizePerClass{0};
    int64_t maxTotalSize{0};
    float scoreThreshold{0.0F};
    float iouThreshold{0.0F};
    ge::DataType boxesType{ge::DT_UNDEFINED};
    bool changeCoordinateFrame{false};
    bool transposeBox{false};
    bool hasClipWindow{false};
    bool hasNumValidBoxes{false};
};

ge::graphStatus ParseRequiredInputsAndAttrs(gert::TilingContext* context, ParsedTilingParams& params)
{
    const gert::StorageShape* boxesInput = context->GetInputShape(kBoxesIndex);
    const gert::StorageShape* scoresInput = context->GetInputShape(kScoresIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresInput);
    const gert::Shape& boxesShape = boxesInput->GetStorageShape();
    const gert::Shape& scoresShape = scoresInput->GetStorageShape();
    OP_CHECK_IF(boxesShape.GetDimNum() != 4 || scoresShape.GetDimNum() != 3,
                OP_LOGE(context, "boxes must be rank 4 and scores must be rank 3."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!HasPositiveDims(boxesShape) || !HasPositiveDims(scoresShape),
                OP_LOGE(context, "Dynamic or non-positive dimensions are not supported by this tiling."),
                return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* scoreThreshold = attrs->GetAttrPointer<float>(kScoreThresholdAttrIndex);
    const float* iouThreshold = attrs->GetAttrPointer<float>(kIouThresholdAttrIndex);
    const int64_t* maxSizePerClass = attrs->GetAttrPointer<int64_t>(kMaxSizePerClassAttrIndex);
    const int64_t* maxTotalSize = attrs->GetAttrPointer<int64_t>(kMaxTotalSizeAttrIndex);
    const bool* changeCoordinateFrame = attrs->GetAttrPointer<bool>(kChangeCoordinateFrameAttrIndex);
    const bool* transposeBox = attrs->GetAttrPointer<bool>(kTransposeBoxAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreThreshold);
    OP_CHECK_NULL_WITH_CONTEXT(context, iouThreshold);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxSizePerClass);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxTotalSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, changeCoordinateFrame);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeBox);
    OP_CHECK_IF(!std::isfinite(*scoreThreshold) || !std::isfinite(*iouThreshold) || *iouThreshold < 0.0F ||
                    *iouThreshold > 1.0F,
                OP_LOGE(context, "score_threshold must be finite and iou_threshold must be finite in [0, 1]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*maxSizePerClass <= 0 || *maxSizePerClass > kMaxOutputSize || *maxTotalSize <= 0 ||
                    *maxTotalSize > kMaxOutputSize,
                OP_LOGE(context, "max_size_per_class and max_total_size must be in [1, %ld].", kMaxOutputSize),
                return ge::GRAPH_FAILED);

    params.batch = boxesShape.GetDim(0);
    params.transposeBox = *transposeBox;
    if (params.transposeBox) {
        params.boxClassesNum = boxesShape.GetDim(1);
        OP_CHECK_IF(boxesShape.GetDim(2) != 4, OP_LOGE(context, "boxes dim 2 must be 4 when transpose_box is true."),
                    return ge::GRAPH_FAILED);
        params.boxesNum = boxesShape.GetDim(3);
    } else {
        params.boxesNum = boxesShape.GetDim(1);
        params.boxClassesNum = boxesShape.GetDim(2);
        OP_CHECK_IF(boxesShape.GetDim(3) != 4, OP_LOGE(context, "boxes last dimension must be 4."),
                    return ge::GRAPH_FAILED);
    }
    params.classesNum = scoresShape.GetDim(2);
    OP_CHECK_IF(scoresShape.GetDim(0) != params.batch || scoresShape.GetDim(1) != params.boxesNum ||
                    (params.boxClassesNum != 1 && params.boxClassesNum != params.classesNum),
                OP_LOGE(context, "boxes and scores shapes are incompatible."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(params.batch > std::numeric_limits<int64_t>::max() / params.classesNum,
                OP_LOGE(context, "batch-by-class task count overflow."), return ge::GRAPH_FAILED);

    const auto* boxesDesc = context->GetInputDesc(kBoxesIndex);
    const auto* scoresDesc = context->GetInputDesc(kScoresIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresDesc);
    params.boxesType = boxesDesc->GetDataType();
    OP_CHECK_IF((params.boxesType != ge::DT_FLOAT16 && params.boxesType != ge::DT_FLOAT) ||
                    scoresDesc->GetDataType() != params.boxesType,
                OP_LOGE(context, "boxes and scores must have the same dtype in {float16, float}."),
                return ge::GRAPH_FAILED);

    params.scoreThreshold = *scoreThreshold;
    params.iouThreshold = *iouThreshold;
    params.maxSizePerClass = *maxSizePerClass;
    params.maxTotalSize = *maxTotalSize;
    params.changeCoordinateFrame = *changeCoordinateFrame;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateOptionalInputs(gert::TilingContext* context, ParsedTilingParams& params)
{
    const gert::StorageShape* clipWindowInput = context->GetInputShape(kClipWindowIndex);
    const gert::StorageShape* numValidBoxesInput = context->GetInputShape(kNumValidBoxesIndex);
    if (clipWindowInput != nullptr) {
        const auto* clipWindowDesc = context->GetInputDesc(kClipWindowIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context, clipWindowDesc);
        // The tiling context compacts a later optional tensor into slot 2 when
        // clip_window is absent. Distinguish that valid [B]/int32 NVB input
        // from clip_window here; the kernel ABI itself retains the original
        // fourth argument position for num_valid_boxes.
        if (clipWindowDesc->GetDataType() == ge::DT_INT32) {
            const gert::Shape& numValidBoxesShape = clipWindowInput->GetStorageShape();
            OP_CHECK_IF(numValidBoxesShape.GetDimNum() != 1 || numValidBoxesShape.GetDim(0) != params.batch,
                        OP_LOGE(context, "num_valid_boxes must have shape [B]."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(numValidBoxesInput != nullptr,
                        OP_LOGE(context, "num_valid_boxes is present in both optional input slots."),
                        return ge::GRAPH_FAILED);
            params.hasNumValidBoxes = true;
        } else {
            params.hasClipWindow = true;
            const gert::Shape& clipWindowShape = clipWindowInput->GetStorageShape();
            OP_CHECK_IF(clipWindowShape.GetDimNum() != 2 || clipWindowShape.GetDim(0) != params.batch ||
                            clipWindowShape.GetDim(1) != 4,
                        OP_LOGE(context, "clip_window must have shape [B, 4]."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(clipWindowDesc->GetDataType() != params.boxesType,
                        OP_LOGE(context, "clip_window dtype must equal boxes dtype."), return ge::GRAPH_FAILED);
        }
    }
    if (numValidBoxesInput != nullptr) {
        const gert::Shape& numValidBoxesShape = numValidBoxesInput->GetStorageShape();
        OP_CHECK_IF(numValidBoxesShape.GetDimNum() != 1 || numValidBoxesShape.GetDim(0) != params.batch,
                    OP_LOGE(context, "num_valid_boxes must have shape [B]."), return ge::GRAPH_FAILED);
        const auto* numValidBoxesDesc = context->GetInputDesc(kNumValidBoxesIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context, numValidBoxesDesc);
        OP_CHECK_IF(numValidBoxesDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(context, "num_valid_boxes must be int32."), return ge::GRAPH_FAILED);
        params.hasNumValidBoxes = true;
    }
    OP_CHECK_IF(params.changeCoordinateFrame && !params.hasClipWindow,
                OP_LOGE(context, "change_coordinate_frame requires clip_window."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateOutputContract(gert::TilingContext* context, const ParsedTilingParams& params)
{
    const gert::StorageShape* outputBoxes = context->GetOutputShape(kNmsedBoxesIndex);
    const gert::StorageShape* outputScores = context->GetOutputShape(kNmsedScoresIndex);
    const gert::StorageShape* outputClasses = context->GetOutputShape(kNmsedClassesIndex);
    const gert::StorageShape* outputNum = context->GetOutputShape(kNmsedNumIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputBoxes);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputScores);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputClasses);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputNum);
    const gert::Shape& outputBoxesShape = outputBoxes->GetStorageShape();
    const gert::Shape& outputScoresShape = outputScores->GetStorageShape();
    const gert::Shape& outputClassesShape = outputClasses->GetStorageShape();
    const gert::Shape& outputNumShape = outputNum->GetStorageShape();
    OP_CHECK_IF(outputBoxesShape.GetDimNum() != 3 || outputBoxesShape.GetDim(0) != params.batch ||
                    outputBoxesShape.GetDim(1) != params.maxTotalSize || outputBoxesShape.GetDim(2) != 4 ||
                    outputScoresShape.GetDimNum() != 2 || outputScoresShape.GetDim(0) != params.batch ||
                    outputScoresShape.GetDim(1) != params.maxTotalSize || outputClassesShape.GetDimNum() != 2 ||
                    outputClassesShape.GetDim(0) != params.batch ||
                    outputClassesShape.GetDim(1) != params.maxTotalSize || outputNumShape.GetDimNum() != 1 ||
                    outputNumShape.GetDim(0) != params.batch,
                OP_LOGE(context, "Output shapes do not match BatchMultiClassNonMaxSuppression contract."),
                return ge::GRAPH_FAILED);
    const auto* outputBoxesDesc = context->GetOutputDesc(kNmsedBoxesIndex);
    const auto* outputScoresDesc = context->GetOutputDesc(kNmsedScoresIndex);
    const auto* outputClassesDesc = context->GetOutputDesc(kNmsedClassesIndex);
    const auto* outputNumDesc = context->GetOutputDesc(kNmsedNumIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputBoxesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputScoresDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputClassesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputNumDesc);
    OP_CHECK_IF(
        outputBoxesDesc->GetDataType() != params.boxesType || outputScoresDesc->GetDataType() != params.boxesType ||
            outputClassesDesc->GetDataType() != params.boxesType || outputNumDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(context, "Output dtypes do not match BatchMultiClassNonMaxSuppression contract."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
ge::graphStatus BatchMultiClassNonMaxSuppressionTiling::CheckAndParse()
{
    ParsedTilingParams params;
    if (ParseRequiredInputsAndAttrs(context_, params) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateOptionalInputs(context_, params) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateOutputContract(context_, params) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    batch_ = params.batch;
    boxesNum_ = params.boxesNum;
    classesNum_ = params.classesNum;
    boxClassesNum_ = params.boxClassesNum;
    maxSizePerClass_ = params.maxSizePerClass;
    maxTotalSize_ = params.maxTotalSize;
    scoreThreshold_ = params.scoreThreshold;
    iouThreshold_ = params.iouThreshold;
    hasClipWindow_ = params.hasClipWindow;
    hasNumValidBoxes_ = params.hasNumValidBoxes;
    changeCoordinateFrame_ = params.changeCoordinateFrame;
    transposeBox_ = params.transposeBox;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchMultiClassNonMaxSuppressionTiling::SetTilingData()
{
    const auto* compileInfo = static_cast<const BatchMultiClassNonMaxSuppressionCompileInfo*>(
        context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    OP_CHECK_IF(compileInfo->coreNum == 0 || compileInfo->ubSize == 0,
                OP_LOGE(context_, "Ascend950 platform resource query failed."), return ge::GRAPH_FAILED);

    tilingData_ = context_->GetTilingData<BatchMultiClassNonMaxSuppressionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    OP_CHECK_IF(memset_s(tilingData_, sizeof(BatchMultiClassNonMaxSuppressionTilingData), 0,
                         sizeof(BatchMultiClassNonMaxSuppressionTilingData)) != EOK,
                OP_LOGE(context_, "Failed to initialize tiling data."), return ge::GRAPH_FAILED);
    const int64_t taskCount = batch_ * classesNum_;
    OP_CHECK_IF(taskCount <= 0, OP_LOGE(context_, "Invalid batch-by-class task count."), return ge::GRAPH_FAILED);
    const int64_t usedCoreNum = std::min<int64_t>(taskCount, static_cast<int64_t>(compileInfo->coreNum));

    OP_CHECK_IF(classesNum_ > std::numeric_limits<int64_t>::max() / maxSizePerClass_,
                OP_LOGE(context_, "Cross-class TopK input size overflow."), return ge::GRAPH_FAILED);
    const int64_t mergeInputCount = classesNum_ * maxSizePerClass_;
    OP_CHECK_IF(mergeInputCount <= 0 || mergeInputCount > std::numeric_limits<int32_t>::max() - kTopKAlignment,
                OP_LOGE(context_, "Cross-class TopK input size is not supported."), return ge::GRAPH_FAILED);
    const int64_t mergeInputSize = std::min<int64_t>(
        kMergeTopKInputCapacity, (mergeInputCount + kTopKAlignment - 1) / kTopKAlignment * kTopKAlignment);
    const int64_t mergeOutputCount = std::min<int64_t>(maxTotalSize_, mergeInputCount);
    const int64_t mergeOutputSize = (maxTotalSize_ + kTopKAlignment - 1) / kTopKAlignment * kTopKAlignment;
    OP_CHECK_IF(mergeInputSize < mergeOutputSize,
                OP_LOGE(context_, "Cross-class TopK merge input capacity is smaller than its output size."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    AscendC::tiling::TopkTiling mergeTopKTiling{};
    OP_CHECK_IF(!AscendC::TopKTilingFunc(ascendcPlatform, static_cast<uint32_t>(mergeInputSize), 1,
                                         static_cast<uint32_t>(mergeOutputCount), sizeof(float), true,
                                         AscendC::TopKMode::TOPK_NORMAL, true, mergeTopKTiling),
                OP_LOGE(context_, "Ascend950 TopK tiling failed for cross-class merge."), return ge::GRAPH_FAILED);
    uint32_t topKMaxTmpBytes = 0;
    uint32_t topKMinTmpBytes = 0;
    OP_CHECK_IF(!AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, static_cast<uint32_t>(mergeInputSize), 1, false, true,
                                               AscendC::TopKMode::TOPK_NORMAL, true, sizeof(float), topKMaxTmpBytes,
                                               topKMinTmpBytes),
                OP_LOGE(context_, "Ascend950 TopK workspace query failed for cross-class merge."),
                return ge::GRAPH_FAILED);
    const uint64_t topKTempBytes = std::max<uint64_t>(topKMaxTmpBytes, kWorkspaceAlignment);
    const uint64_t mergeFixedUbBytes = static_cast<uint64_t>(mergeInputSize) * (sizeof(float) + sizeof(int32_t)) +
                                       static_cast<uint64_t>(mergeOutputSize) * (sizeof(float) + sizeof(int32_t)) +
                                       topKTempBytes;
    OP_CHECK_IF(
        compileInfo->ubSize <= kSimtUbReserveBytes || mergeFixedUbBytes >= compileInfo->ubSize - kSimtUbReserveBytes,
        OP_LOGE(context_, "Insufficient UB for cross-class TopK merge."), return ge::GRAPH_FAILED);
    const uint64_t availableUbBytes = compileInfo->ubSize - kSimtUbReserveBytes - mergeFixedUbBytes;
    const uint64_t bytesPerTileElement = kVectorFloatFieldCount * sizeof(float) + sizeof(uint8_t);
    int64_t tileSize = std::min<int64_t>(boxesNum_,
                                         std::min<int64_t>(kMaxTileSize, availableUbBytes / bytesPerTileElement));
    if (tileSize < mergeOutputSize) {
        const uint64_t minimumReduceBytes = static_cast<uint64_t>(mergeOutputSize) * sizeof(float);
        OP_CHECK_IF(minimumReduceBytes > availableUbBytes,
                    OP_LOGE(context_, "Insufficient UB for cross-class TopK reduction."), return ge::GRAPH_FAILED);
        tileSize = std::min<int64_t>(tileSize, static_cast<int64_t>((availableUbBytes - minimumReduceBytes) /
                                                                    (bytesPerTileElement - sizeof(float))));
    }
    if (tileSize >= kTileAlignment) {
        tileSize = tileSize / kTileAlignment * kTileAlignment;
    }
    OP_CHECK_IF(tileSize <= 0, OP_LOGE(context_, "Insufficient UB for BatchMultiClassNonMaxSuppression tile."),
                return ge::GRAPH_FAILED);
    const int64_t reduceBufferSize = std::max<int64_t>(tileSize, mergeOutputSize);

    const uint64_t boxesNum = static_cast<uint64_t>(boxesNum_);
    const uint64_t taskCountU64 = static_cast<uint64_t>(taskCount);
    const uint64_t maxSizePerClass = static_cast<uint64_t>(maxSizePerClass_);
    OP_CHECK_IF(boxesNum > std::numeric_limits<uint64_t>::max() / sizeof(float),
                OP_LOGE(context_, "Scratch workspace size overflow."), return ge::GRAPH_FAILED);
    const uint64_t scratchFieldBytes = boxesNum * sizeof(float);
    OP_CHECK_IF(scratchFieldBytes > std::numeric_limits<uint64_t>::max() - (kWorkspaceAlignment - 1),
                OP_LOGE(context_, "Scratch workspace size overflow."), return ge::GRAPH_FAILED);
    const uint64_t scratchFieldStride = (scratchFieldBytes + kWorkspaceAlignment - 1) / kWorkspaceAlignment *
                                        kWorkspaceAlignment;
    OP_CHECK_IF(scratchFieldStride > std::numeric_limits<uint64_t>::max() / kScratchFloatFieldCount,
                OP_LOGE(context_, "Scratch workspace size overflow."), return ge::GRAPH_FAILED);
    const uint64_t scratchBytesPerCore = scratchFieldStride * kScratchFloatFieldCount;
    // Non-transpose inputs are gathered by one persistent SIMT VF per AIV
    // core.  Each batch/class task therefore needs its own staging region so
    // the AIV NMS stage can consume it without relaunching the VF task.
    OP_CHECK_IF(taskCountU64 > std::numeric_limits<uint64_t>::max() / scratchBytesPerCore,
                OP_LOGE(context_, "Scratch workspace size overflow."), return ge::GRAPH_FAILED);
    const uint64_t scratchBytes = taskCountU64 * scratchBytesPerCore;
    OP_CHECK_IF(taskCountU64 > std::numeric_limits<uint64_t>::max() / maxSizePerClass,
                OP_LOGE(context_, "Class result workspace size overflow."), return ge::GRAPH_FAILED);
    const uint64_t classResultCount = taskCountU64 * maxSizePerClass;
    OP_CHECK_IF(classResultCount > std::numeric_limits<uint64_t>::max() / (4 * sizeof(float)) ||
                    taskCountU64 > std::numeric_limits<uint64_t>::max() / sizeof(int32_t),
                OP_LOGE(context_, "Class result workspace size overflow."), return ge::GRAPH_FAILED);
    const uint64_t classBoxesBytes = classResultCount * 4 * sizeof(float);
    const uint64_t classScoresBytes = classResultCount * sizeof(float);
    const uint64_t classCountsBytes = taskCountU64 * sizeof(float);
    OP_CHECK_IF(
        static_cast<uint64_t>(batch_) > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(maxTotalSize_),
        OP_LOGE(context_, "Cross-class TopK workspace size overflow."), return ge::GRAPH_FAILED);
    const uint64_t mergeResultCount = static_cast<uint64_t>(batch_) * static_cast<uint64_t>(maxTotalSize_);
    OP_CHECK_IF(mergeResultCount > std::numeric_limits<uint64_t>::max() / sizeof(float),
                OP_LOGE(context_, "Cross-class TopK workspace size overflow."), return ge::GRAPH_FAILED);
    const uint64_t mergeScoresBytes = mergeResultCount * sizeof(float);
    const uint64_t mergeIndicesBytes = mergeResultCount * sizeof(int32_t);
    uint64_t classBoxesOffset = scratchBytes;
    uint64_t classScoresOffset = 0;
    uint64_t classCountsOffset = 0;
    uint64_t mergeScoresOffset = 0;
    uint64_t mergeIndicesOffset = 0;
    uint64_t userWorkspaceBytes = 0;
    OP_CHECK_IF(!AddAligned(classBoxesOffset, classBoxesBytes, classScoresOffset) ||
                    !AddAligned(classScoresOffset, classScoresBytes, classCountsOffset) ||
                    !AddAligned(classCountsOffset, classCountsBytes, mergeScoresOffset) ||
                    !AddAligned(mergeScoresOffset, mergeScoresBytes, mergeIndicesOffset) ||
                    !AddAligned(mergeIndicesOffset, mergeIndicesBytes, userWorkspaceBytes),
                OP_LOGE(context_, "Class result workspace size overflow."), return ge::GRAPH_FAILED);
    tilingData_->batch = batch_;
    tilingData_->boxesNum = boxesNum_;
    tilingData_->classesNum = classesNum_;
    tilingData_->boxClassesNum = boxClassesNum_;
    tilingData_->maxSizePerClass = maxSizePerClass_;
    tilingData_->maxTotalSize = maxTotalSize_;
    tilingData_->usedCoreNum = usedCoreNum;
    tilingData_->tileSize = tileSize;
    tilingData_->reduceBufferSize = reduceBufferSize;
    tilingData_->mergeInputCount = mergeInputCount;
    tilingData_->mergeInputSize = mergeInputSize;
    tilingData_->mergeOutputCount = mergeOutputCount;
    tilingData_->mergeOutputSize = mergeOutputSize;
    tilingData_->scratchFieldStride = scratchFieldStride;
    tilingData_->scratchBytesPerCore = scratchBytesPerCore;
    tilingData_->classBoxesOffset = classBoxesOffset;
    tilingData_->classScoresOffset = classScoresOffset;
    tilingData_->classCountsOffset = classCountsOffset;
    tilingData_->mergeScoresOffset = mergeScoresOffset;
    tilingData_->mergeIndicesOffset = mergeIndicesOffset;
    tilingData_->topKTempBytes = topKTempBytes;
    tilingData_->mergeTopKTiling = mergeTopKTiling;
    tilingData_->scoreThreshold = scoreThreshold_;
    tilingData_->iouThreshold = iouThreshold_;
    tilingData_->hasClipWindow = static_cast<uint8_t>(hasClipWindow_);
    tilingData_->hasNumValidBoxes = static_cast<uint8_t>(hasNumValidBoxes_);
    tilingData_->changeCoordinateFrame = static_cast<uint8_t>(changeCoordinateFrame_);
    tilingData_->transposeBox = static_cast<uint8_t>(transposeBox_);
    tilingData_->use32Index = static_cast<uint8_t>(
        ProductFitsUint32(static_cast<uint64_t>(batch_), boxesNum, static_cast<uint64_t>(classesNum_), 1) &&
        ProductFitsUint32(static_cast<uint64_t>(batch_), boxesNum, static_cast<uint64_t>(boxClassesNum_), 4));

    size_t* workspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    OP_CHECK_IF(userWorkspaceBytes > std::numeric_limits<size_t>::max() - ascendcPlatform.GetLibApiWorkSpaceSize(),
                OP_LOGE(context_, "Workspace size exceeds platform limit."), return ge::GRAPH_FAILED);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + static_cast<size_t>(userWorkspaceBytes);
    context_->SetBlockDim(usedCoreNum);
    context_->SetScheduleMode(kScheduleMode);
    context_->SetTilingKey(BATCH_MULTI_CLASS_NMS_TILING_KEY);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    const size_t tilingDataSize = sizeof(BatchMultiClassNonMaxSuppressionTilingData);
    OP_CHECK_IF(memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), tilingData_, tilingDataSize) != EOK,
                OP_LOGE(context_, "Failed to serialize tiling data."), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchMultiClassNonMaxSuppressionTiling::RunTiling()
{
    const ge::graphStatus status = CheckAndParse();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    return SetTilingData();
}

static ge::graphStatus TilingPrepareForBatchMultiClassNonMaxSuppression(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<BatchMultiClassNonMaxSuppressionCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->coreNum == 0 || compileInfo->ubSize == 0,
                OP_LOGE(context, "Ascend950 platform resource query failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForBatchMultiClassNonMaxSuppression(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    BatchMultiClassNonMaxSuppressionTiling tiling(context);
    return tiling.RunTiling();
}

IMPL_OP_OPTILING(BatchMultiClassNonMaxSuppression)
    .Tiling(TilingForBatchMultiClassNonMaxSuppression)
    .TilingParse<BatchMultiClassNonMaxSuppressionCompileInfo>(TilingPrepareForBatchMultiClassNonMaxSuppression);
} // namespace optiling
