/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <cstring>
#include <limits>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../gaussian_blur_utils.h"
#include "../../op_kernel/arch35/gaussian_blur_tiling_data.h"
#include "../../op_kernel/arch35/gaussian_blur_tiling_key.h"
#include "gaussian_blur_tiling_cost_model.h"

namespace optiling {

static constexpr size_t KSIZE_ATTR_INDEX = 0;
static constexpr size_t SIGMA_X_ATTR_INDEX = 1;
static constexpr size_t SIGMA_Y_ATTR_INDEX = 2;
static constexpr size_t BORDER_TYPE_ATTR_INDEX = 3;
static constexpr uint32_t DEFAULT_DCACHE_SIZE = 128U * 1024U;
static constexpr uint32_t FUSED_DCACHE_SIZE = 64U * 1024U;
static constexpr uint32_t ROW_PIPELINE_UB_BYTES = 2U * GAUSSIAN_BLUR_ROW_UB_BUFFER_BYTES +
                                                  GAUSSIAN_BLUR_ROW_SHARED_UB_BYTES +
                                                  GAUSSIAN_BLUR_KERNEL_MAX_SIZE * sizeof(float);
static constexpr uint32_t ROW_LARGE_TILE_W = GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ? 288U : 192U;
static constexpr uint32_t ROW_LARGE_GATHER_W = GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ? 352U : 240U;
static constexpr uint32_t ROW_LARGE_SHARED_UB_BYTES = GAUSSIAN_BLUR_ROW_TILE_H * ROW_LARGE_GATHER_W *
                                                      GAUSSIAN_BLUR_ROW_UB_MAX_CHANNELS * sizeof(float);
static constexpr uint32_t ROW_LARGE_BUFFER_CHANNELS = GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ?
                                                          GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP :
                                                          GAUSSIAN_BLUR_ROW_UB_MAX_CHANNELS;
static constexpr uint32_t ROW_LARGE_UB_BUFFER_BYTES = GAUSSIAN_BLUR_ROW_TILE_H * ROW_LARGE_GATHER_W *
                                                      ROW_LARGE_BUFFER_CHANNELS * sizeof(float);
static constexpr uint32_t ROW_LARGE_PIPELINE_UB_BYTES = 2U * ROW_LARGE_UB_BUFFER_BYTES +
                                                        (GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ?
                                                             0U :
                                                             ROW_LARGE_SHARED_UB_BYTES) +
                                                        GAUSSIAN_BLUR_KERNEL_MAX_SIZE * sizeof(float);
static constexpr uint32_t ROW_W96_TILE_W = 96U;
static constexpr uint32_t ROW_W96_GATHER_W = 160U;
static constexpr uint32_t ROW_W96_UB_BUFFER_BYTES = GAUSSIAN_BLUR_ROW_TILE_H * ROW_W96_GATHER_W *
                                                    GAUSSIAN_BLUR_ROW_UB_MAX_CHANNELS * sizeof(float);
static constexpr uint32_t ROW_W96_PIPELINE_UB_BYTES = 3U * ROW_W96_UB_BUFFER_BYTES +
                                                      GAUSSIAN_BLUR_KERNEL_MAX_SIZE * sizeof(float);
static constexpr uint32_t FUSED_TILE_W = 32U;
static constexpr uint32_t FUSED_TILE_H = 20U;
static constexpr uint32_t FUSED_CHANNELS = 16U;
static constexpr uint32_t FUSED_C1_K31_TILE_W = 128U;
static constexpr uint64_t FUSED_DIRECT_SIMT_MAX_OUTPUTS = 512U;
static constexpr uint64_t FUSED_DIRECT_SIMT_MAX_WORK = 262144U;
static constexpr uint32_t FUSED_MAX_INPUT_W = FUSED_TILE_W + 21U - 1U;
static constexpr uint32_t FUSED_INPUT_H = FUSED_TILE_H + 4U;
static constexpr uint32_t FUSED_UB_BYTES = (FUSED_MAX_INPUT_W * FUSED_INPUT_H * FUSED_CHANNELS +
                                            FUSED_TILE_W * FUSED_INPUT_H * FUSED_CHANNELS + 2U * 32U) *
                                           sizeof(float);

struct GaussianBlurCompileInfo {};

static uint32_t CeilDiv(uint32_t value, uint32_t divisor) { return (value + divisor - 1U) / divisor; }

static bool IsSupportedKernel(int64_t kernel)
{
    return kernel == 1 || kernel == 3 || kernel == 5 || kernel == 7 || kernel == 9 || kernel == 11 || kernel == 15 ||
           kernel == 21 || kernel == 31;
}

static uint32_t SelectPath(uint32_t channels)
{
    if (channels == 1U) {
        return GAUSSIAN_BLUR_PATH_C1_FAST;
    }
    if (channels == 3U) {
        return GAUSSIAN_BLUR_PATH_C3_FAST;
    }
    if (channels == 4U) {
        return GAUSSIAN_BLUR_PATH_C4_FAST;
    }
    if (channels >= 5U && channels <= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP) {
        return GAUSSIAN_BLUR_PATH_GENERIC_C8;
    }
    return GAUSSIAN_BLUR_PATH_GENERIC_C;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingGaussianBlur(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0U, OP_LOGE(context, "invalid Ascend950 AIV platform information."),
                return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const auto shape = inputShape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() < 2U || shape.GetDimNum() > 3U,
                OP_LOGE(context, "GaussianBlur pass only supports rank 2/3."), return ge::GRAPH_FAILED);
    const int64_t heightDim = shape.GetDim(0);
    const int64_t widthDim = shape.GetDim(1);
    const int64_t channelsDim = shape.GetDimNum() == 3U ? shape.GetDim(2) : 1;
    constexpr int64_t maxUint32 = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    OP_CHECK_IF(heightDim <= 0 || widthDim <= 0 || channelsDim <= 0 || heightDim > maxUint32 || widthDim > maxUint32 ||
                    channelsDim > maxUint32,
                OP_LOGE(context, "GaussianBlur shape dimensions must be in [1, UINT32_MAX]."), return ge::GRAPH_FAILED);
    const uint32_t height = static_cast<uint32_t>(heightDim);
    const uint32_t width = static_cast<uint32_t>(widthDim);
    const uint32_t channels = static_cast<uint32_t>(channelsDim);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* ksize = attrs->GetListInt(KSIZE_ATTR_INDEX);
    const float* sigmaX = attrs->GetFloat(SIGMA_X_ATTR_INDEX);
    const float* sigmaY = attrs->GetFloat(SIGMA_Y_ATTR_INDEX);
    const int64_t* borderType = attrs->GetInt(BORDER_TYPE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksize);
    OP_CHECK_NULL_WITH_CONTEXT(context, sigmaX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sigmaY);
    OP_CHECK_NULL_WITH_CONTEXT(context, borderType);
    OP_CHECK_IF(ksize->GetSize() != 2U, OP_LOGE(context, "ksize must contain 2 elements."), return ge::GRAPH_FAILED);

    constexpr bool rowPass = false;
    constexpr bool fusedPass = true;

    const int64_t* kernelData = ksize->GetData();
    gaussian_blur::CanonicalParams params;
    OP_CHECK_IF(!gaussian_blur::CanonicalizeParams(kernelData[0], kernelData[1], static_cast<double>(*sigmaX),
                                                   static_cast<double>(*sigmaY), *borderType, width, height, params),
                OP_LOGE(context, "failed to canonicalize GaussianBlur pass attributes."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsSupportedKernel(params.kernelW) || !IsSupportedKernel(params.kernelH),
                OP_LOGE(context, "GaussianBlur pass supports K1/K3/K5/K7/K9/K11/K15/K21/K31."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to set GaussianBlur pass workspace."), return ge::GRAPH_FAILED);

    auto* tiling = context->GetTilingData<GaussianBlurTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    std::memset(tiling, 0, sizeof(GaussianBlurTilingData));
    tiling->h = height;
    tiling->w = width;
    tiling->c = channels;
    tiling->pathMode = SelectPath(channels);
    const uint32_t rowVariant = GAUSSIAN_BLUR_PASS_ROW_W128;
    const uint32_t columnVariant = GAUSSIAN_BLUR_PASS_COLUMN_H96;
    const uint32_t rowTileW = GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ?
                                  ROW_LARGE_TILE_W :
                                  (rowVariant == GAUSSIAN_BLUR_PASS_ROW_W192 ?
                                       ROW_LARGE_TILE_W :
                                       (rowVariant == GAUSSIAN_BLUR_PASS_ROW_W96 ? ROW_W96_TILE_W :
                                                                                   GAUSSIAN_BLUR_ROW_TILE_W));
    const uint64_t fusedOutputs = static_cast<uint64_t>(height) * width * channels;
    const uint64_t fusedWork = fusedOutputs * params.kernelW * params.kernelH;
    const bool fusedDirectSimt = fusedPass && GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS && width < 128U &&
                                 fusedOutputs <= FUSED_DIRECT_SIMT_MAX_OUTPUTS &&
                                 fusedWork <= FUSED_DIRECT_SIMT_MAX_WORK;
    const bool fusedC1K31 = fusedPass && GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS && channels == 1U &&
                            params.kernelW == 31 && !fusedDirectSimt;
    const uint32_t tileW = fusedPass ?
                               (GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ? (fusedC1K31 ? FUSED_C1_K31_TILE_W : 128U) :
                                                                           FUSED_TILE_W) :
                               (rowPass ? rowTileW : GAUSSIAN_BLUR_COLUMN_TILE_W);
    tiling->tilesX = CeilDiv(width, tileW);
    const gaussian_blur_cost_model::Problem fusedProblem{height,
                                                         width,
                                                         channels,
                                                         static_cast<uint32_t>(params.kernelW),
                                                         static_cast<uint32_t>(params.kernelH),
                                                         tileW,
                                                         GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP,
                                                         coreNum};
    const bool fusedC1Weighted = fusedPass && GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS && !fusedDirectSimt &&
                                 channels == 1U && tiling->tilesX > 1U && height >= coreNum &&
                                 (gaussian_blur_cost_model::C1HasInteriorTile(fusedProblem) ||
                                  gaussian_blur_cost_model::C1HasSevereTileWeightImbalance(fusedProblem));
    uint32_t targetTilesY = 1U;
    uint32_t tileH = rowPass ? GAUSSIAN_BLUR_ROW_TILE_H : GAUSSIAN_BLUR_COLUMN_TILE_H;
    if (fusedPass) {
        if (GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS) {
            targetTilesY = fusedDirectSimt ? 1U : gaussian_blur_cost_model::SelectTilesY(fusedProblem);
            tileH = CeilDiv(height, targetTilesY);
        } else {
            tileH = FUSED_TILE_H;
        }
    }
    const bool fusedC8FullCore = fusedPass && GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS &&
                                 gaussian_blur_cost_model::ShouldUseFullCoreC8SpatialBudget(fusedProblem, targetTilesY);
    const bool weightedSpatialTiling = fusedC1Weighted || fusedC8FullCore;
    tiling->tilesY = weightedSpatialTiling ? 1U : CeilDiv(height, tileH);
    tiling->reserved[0] = weightedSpatialTiling ? 0U : tileH;
    uint32_t effectiveTileW = tileW;
    if (fusedPass && GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS && channels > 1U && params.kernelW == 31 &&
        tiling->tilesX > 1U && width % tileW < tileW / 4U) {
        effectiveTileW = CeilDiv(width, tiling->tilesX);
    }
    tiling->reserved[1] = effectiveTileW;
    const uint32_t channelTiles = fusedPass ? CeilDiv(channels,
                                                      GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ? 8U : FUSED_CHANNELS) :
                                              (tiling->pathMode == GAUSSIAN_BLUR_PATH_GENERIC_C ?
                                                   CeilDiv(channels, GAUSSIAN_BLUR_CHANNEL_TILE) :
                                                   1U);
    const uint64_t totalTiles = weightedSpatialTiling ?
                                    coreNum :
                                    static_cast<uint64_t>(tiling->tilesX) * tiling->tilesY * channelTiles;
    OP_CHECK_IF(totalTiles == 0U || totalTiles > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "GaussianBlur tile count exceeds uint32 range."), return ge::GRAPH_FAILED);
    tiling->totalTiles = static_cast<uint32_t>(totalTiles);
    tiling->kernelSize = static_cast<uint32_t>(rowPass || fusedPass ? params.kernelW : params.kernelH);
    tiling->kernelSizeY = static_cast<uint32_t>(params.kernelH);
    tiling->radius = (tiling->kernelSize - 1U) / 2U;
    tiling->borderType = static_cast<uint32_t>(params.borderType);
    gaussian_blur::BuildGaussianWeights(tiling->kernelSize, rowPass || fusedPass ? params.sigmaX : params.sigmaY,
                                        tiling->weights, GAUSSIAN_BLUR_KERNEL_MAX_SIZE);
    gaussian_blur::BuildGaussianWeights(params.kernelH, params.sigmaY, tiling->weightsY, GAUSSIAN_BLUR_KERNEL_MAX_SIZE);

    const uint32_t usedCores = tiling->totalTiles < coreNum ? tiling->totalTiles : coreNum;
    context->SetBlockDim(usedCores == 0U ? 1U : usedCores);
    const uint32_t dcacheSize = (fusedPass || GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS) ? FUSED_DCACHE_SIZE :
                                                                                          DEFAULT_DCACHE_SIZE;
    OP_CHECK_IF(ubSize <= dcacheSize,
                OP_LOGE(context, "UB size %lu is not larger than DCache reservation %u.", ubSize, dcacheSize),
                return ge::GRAPH_FAILED);
    const uint32_t localMemorySize = static_cast<uint32_t>(ubSize - dcacheSize);
    const uint32_t columnChannels = GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS && !rowPass &&
                                            tiling->pathMode == GAUSSIAN_BLUR_PATH_GENERIC_C &&
                                            tiling->kernelSize == 31U ?
                                        2U * GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP :
                                        (tiling->pathMode == GAUSSIAN_BLUR_PATH_GENERIC_C8 ?
                                             GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP :
                                             GAUSSIAN_BLUR_CHANNEL_TILE);
    const uint32_t columnSharedBytes = (tileH + 2U * GAUSSIAN_BLUR_COLUMN_BLOCK_Y) * GAUSSIAN_BLUR_COLUMN_BLOCK_X *
                                       columnChannels * sizeof(float);
    const uint32_t k31RingUbBytes = (31U * 128U * 8U + (128U + 30U) * 8U + 128U * 8U) * sizeof(float) +
                                    2U * GAUSSIAN_BLUR_KERNEL_MAX_SIZE * sizeof(float);
    const uint32_t requiredLocalMemory = fusedPass ? (GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ? k31RingUbBytes :
                                                                                                 FUSED_UB_BYTES) :
                                                     (rowPass ? (GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ?
                                                                     ROW_LARGE_PIPELINE_UB_BYTES :
                                                                     (rowVariant == GAUSSIAN_BLUR_PASS_ROW_W192 ?
                                                                          ROW_LARGE_PIPELINE_UB_BYTES :
                                                                          (rowVariant == GAUSSIAN_BLUR_PASS_ROW_W96 ?
                                                                               ROW_W96_PIPELINE_UB_BYTES :
                                                                               ROW_PIPELINE_UB_BYTES))) :
                                                                (GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS ?
                                                                     columnSharedBytes +
                                                                         GAUSSIAN_BLUR_KERNEL_MAX_SIZE * sizeof(float) :
                                                                     columnSharedBytes));
    OP_CHECK_IF(
        localMemorySize < requiredLocalMemory,
        OP_LOGE(context, "local memory %u is smaller than required UB %u.", localMemorySize, requiredLocalMemory),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetLocalMemorySize(localMemorySize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);
    if (GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS) {
        if (fusedPass) {
            context->SetTilingKey(GET_TPL_TILING_KEY(GAUSSIAN_BLUR_PASS_FUSED_K31_C4_RING));
        } else if (rowPass) {
            context->SetTilingKey(GET_TPL_TILING_KEY(GAUSSIAN_BLUR_PASS_ROW_W128));
        } else {
            context->SetTilingKey(GET_TPL_TILING_KEY(GAUSSIAN_BLUR_PASS_COLUMN_H96));
        }
    } else if (fusedPass) {
        context->SetTilingKey(GET_TPL_TILING_KEY(GAUSSIAN_BLUR_PASS_FUSED_GENERIC_C8));
    } else if (rowPass) {
        context->SetTilingKey(GET_TPL_TILING_KEY(rowVariant));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(columnVariant));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseGaussianBlur([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GaussianBlur).Tiling(TilingGaussianBlur).TilingParse<GaussianBlurCompileInfo>(TilingParseGaussianBlur);

} // namespace optiling
