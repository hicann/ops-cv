/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GAUSSIAN_BLUR_TILING_COST_MODEL_H
#define GAUSSIAN_BLUR_TILING_COST_MODEL_H

#include <array>
#include <cstdint>
#include <limits>
#include "gaussian_blur_tiling_cost_model_coefficients.h"

namespace optiling::gaussian_blur_cost_model {

// Constants used by the learned cost model. The integer values describe hardware or algorithmic granularity.
static constexpr uint32_t LEARNED_CHANNEL_GROUP = 8U;   // Channels packed into one learned-model tile.
static constexpr uint32_t LEARNED_VECTOR_WIDTH = 64U;   // Elements processed by one vector operation.
static constexpr uint32_t LEARNED_ROW_GROUP = 8U;       // Rows grouped for the scheduling feature.
static constexpr uint32_t LEARNED_RADIUS_DIVISOR = 2U;  // Kernel diameter to radius conversion.
static constexpr uint32_t LEARNED_CENTER_TAP = 1U;      // Center tap added to the kernel radius.
static constexpr uint32_t LEARNED_BASE_WAVE_COUNT = 1U; // One wave is the baseline scheduling cost.
static constexpr double LEARNED_FEATURE_BIAS = 1.0;     // Bias term used by the learned feature vector.
static constexpr double LEARNED_COUNT_SCALE = 1.0e3;    // Normalize operation-count features to thousands.
static constexpr double LEARNED_BYTE_SCALE = 1.0e6;     // Normalize byte-count features to megabytes.

struct Problem {
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t kernelWidth;
    uint32_t kernelHeight;
    uint32_t tileWidth;
    uint32_t channelGroup;
    uint32_t coreCount;
};

constexpr uint32_t CeilDiv(uint32_t value, uint32_t divisor) { return (value + divisor - 1U) / divisor; }

inline uint32_t C1TileWeight(const Problem& problem, uint32_t tileX)
{
    const uint32_t outputBaseX = tileX * problem.tileWidth;
    const uint32_t activeWidth = outputBaseX + problem.tileWidth <= problem.width ? problem.tileWidth :
                                                                                    problem.width - outputBaseX;
    const uint32_t minimumComputeWidth = problem.kernelWidth == 31U ? 1U : 32U;
    const uint32_t effectiveWidth = activeWidth < minimumComputeWidth ? minimumComputeWidth : activeWidth;
    const uint32_t radiusX = problem.kernelWidth / 2U;
    const uint32_t interiorBegin = outputBaseX < radiusX ?
                                       (radiusX - outputBaseX < activeWidth ? radiusX - outputBaseX : activeWidth) :
                                       0U;
    const uint32_t interiorGlobalEnd = problem.width > radiusX ? problem.width - radiusX : 0U;
    const uint32_t interiorEnd = interiorGlobalEnd > outputBaseX ?
                                     (interiorGlobalEnd - outputBaseX < activeWidth ? interiorGlobalEnd - outputBaseX :
                                                                                      activeWidth) :
                                     interiorBegin;
    const uint32_t interiorWidth = interiorEnd > interiorBegin ? interiorEnd - interiorBegin : 0U;
    const uint32_t boundaryPixels = activeWidth - interiorWidth;
    const uint32_t horizontalTaps = problem.kernelWidth / 2U + 1U;
    const uint32_t verticalTaps = problem.kernelHeight / 2U + 1U;
    const uint32_t boundaryExtraHorizontalPenalty = problem.kernelWidth == 31U ? 2U : 6U;
    return effectiveWidth * (verticalTaps + horizontalTaps) +
           boundaryPixels * horizontalTaps * boundaryExtraHorizontalPenalty;
}

inline bool C1HasInteriorTile(const Problem& problem)
{
    const uint32_t tilesX = CeilDiv(problem.width, problem.tileWidth);
    const uint32_t radiusX = problem.kernelWidth / 2U;
    for (uint32_t tileX = 0U; tileX < tilesX; ++tileX) {
        const uint32_t outputBaseX = tileX * problem.tileWidth;
        const uint32_t activeWidth = outputBaseX + problem.tileWidth <= problem.width ? problem.tileWidth :
                                                                                        problem.width - outputBaseX;
        if (outputBaseX >= radiusX && outputBaseX + activeWidth + radiusX <= problem.width) {
            return true;
        }
    }
    return false;
}

inline bool C1HasSevereTileWeightImbalance(const Problem& problem)
{
    const uint32_t tilesX = CeilDiv(problem.width, problem.tileWidth);
    if (tilesX < 2U) {
        return false;
    }
    uint32_t minWeight = std::numeric_limits<uint32_t>::max();
    uint32_t maxWeight = 0U;
    for (uint32_t tileX = 0U; tileX < tilesX; ++tileX) {
        const uint32_t weight = C1TileWeight(problem, tileX);
        minWeight = weight < minWeight ? weight : minWeight;
        maxWeight = weight > maxWeight ? weight : maxWeight;
    }
    return static_cast<uint64_t>(maxWeight) >= static_cast<uint64_t>(minWeight) * 2U;
}

inline bool ShouldUseFullCoreC8SpatialBudget(const Problem& problem, uint32_t tilesY)
{
    const uint32_t tilesX = CeilDiv(problem.width, problem.tileWidth);
    const uint32_t tasks = tilesX * tilesY;
    if (problem.channels < 2U || problem.channels > problem.channelGroup || problem.kernelWidth != 31U || tilesX < 2U ||
        problem.height < problem.coreCount || tasks >= problem.coreCount) {
        return false;
    }
    // Stay in one scheduling wave and only fill a small under-occupied tail.
    // This lets edge X tiles receive one more Y partition without forcing the
    // interior tile into a second wave.
    return problem.coreCount - tasks <= tilesX;
}

enum class LearnedFamily : uint32_t {
    Direct,
    C1K31,
    C1Tile,
    C8Ring,
    MultiC8,
};

inline uint32_t ActualTilesY(uint32_t height, uint32_t requestedTilesY)
{
    const uint32_t requested = requestedTilesY < 1U ? 1U : (requestedTilesY > height ? height : requestedTilesY);
    return CeilDiv(height, CeilDiv(height, requested));
}

inline uint32_t LearnedTileWidth(const Problem& problem)
{
    if (problem.channels == 1U) {
        return 128U;
    }
    const uint32_t tilesX = CeilDiv(problem.width, 128U);
    if (tilesX > 1U && problem.width % 128U < 32U) {
        return CeilDiv(problem.width, tilesX);
    }
    return 128U;
}

inline LearnedFamily SelectLearnedFamily(const Problem& problem)
{
    const uint64_t outputs = static_cast<uint64_t>(problem.height) * problem.width * problem.channels;
    if (problem.width < 128U && outputs <= 512U && outputs * problem.kernelWidth * problem.kernelHeight <= 262144U) {
        return LearnedFamily::Direct;
    }
    if (problem.channels == 1U) {
        return problem.kernelWidth == 31U ? LearnedFamily::C1K31 : LearnedFamily::C1Tile;
    }
    return problem.channels <= 8U ? LearnedFamily::C8Ring : LearnedFamily::MultiC8;
}

inline const LearnedCoefficients& CoefficientsFor(LearnedFamily family)
{
    switch (family) {
        case LearnedFamily::Direct:
            return LEARNED_DIRECT_COEFFICIENTS;
        case LearnedFamily::C1K31:
            return LEARNED_C1_K31_COEFFICIENTS;
        case LearnedFamily::C1Tile:
            return LEARNED_C1_TILE_COEFFICIENTS;
        case LearnedFamily::C8Ring:
            return LEARNED_C8_RING_COEFFICIENTS;
        default:
            return LEARNED_MULTI_C8_COEFFICIENTS;
    }
}

inline LearnedCoefficients LearnedFeatures(const Problem& problem, uint32_t tilesY)
{
    const uint32_t tileWidth = LearnedTileWidth(problem);
    const uint32_t tilesX = CeilDiv(problem.width, tileWidth);
    const uint32_t channelTiles = CeilDiv(problem.channels, LEARNED_CHANNEL_GROUP);
    const uint32_t tasks = tilesX * channelTiles * tilesY;
    const uint32_t waves = CeilDiv(tasks, problem.coreCount);
    const uint32_t rows = CeilDiv(problem.height, tilesY);
    const uint32_t horizontalRows = rows + problem.kernelHeight - 1U;
    const uint32_t radiusX = problem.kernelWidth / 2U;
    LearnedCoefficients critical{};
    double criticalScore = -1.0;

    for (uint32_t tileX = 0U; tileX < tilesX; ++tileX) {
        const uint32_t outputBaseX = tileX * tileWidth;
        const uint32_t activeWidth = outputBaseX + tileWidth <= problem.width ? tileWidth : problem.width - outputBaseX;
        for (uint32_t channelTile = 0U; channelTile < channelTiles; ++channelTile) {
            const uint32_t channelBase = channelTile * LEARNED_CHANNEL_GROUP;
            const uint32_t realChannels = channelBase + LEARNED_CHANNEL_GROUP <= problem.channels ?
                                              LEARNED_CHANNEL_GROUP :
                                              problem.channels - channelBase;
            const uint32_t packedChannels = problem.channels == 1U ? 1U : LEARNED_CHANNEL_GROUP;
            const uint32_t vectors = CeilDiv(activeWidth * packedChannels, LEARNED_VECTOR_WIDTH);
            const uint32_t inputWidth = activeWidth + problem.kernelWidth - 1U;
            const uint64_t inputSegments = static_cast<uint64_t>(horizontalRows) * inputWidth;
            const uint64_t outputSegments = static_cast<uint64_t>(rows) * activeWidth;
            const uint64_t payloadBytes = (inputSegments + outputSegments) * realChannels * sizeof(float);
            const uint64_t stridedSpanBytes = realChannels < problem.channels ?
                                                  (inputSegments + outputSegments) * problem.channels * sizeof(float) :
                                                  0U;
            const uint64_t dmaSegments = inputSegments + outputSegments;
            const uint32_t boundaryPixels = (radiusX < activeWidth ? radiusX : activeWidth) *
                                            (static_cast<uint32_t>(tileX == 0U) +
                                             static_cast<uint32_t>(tileX + 1U == tilesX));
            const uint32_t vectorElements = activeWidth * packedChannels;
            const uint32_t vectorTail = CeilDiv(vectorElements, LEARNED_VECTOR_WIDTH) * LEARNED_VECTOR_WIDTH -
                                        vectorElements;
            LearnedCoefficients values = {
                LEARNED_FEATURE_BIAS,
                static_cast<double>(waves) * horizontalRows * vectors *
                    (problem.kernelWidth / LEARNED_RADIUS_DIVISOR + LEARNED_CENTER_TAP) / LEARNED_COUNT_SCALE,
                static_cast<double>(waves) * rows * vectors *
                    (problem.kernelHeight / LEARNED_RADIUS_DIVISOR + LEARNED_CENTER_TAP) / LEARNED_COUNT_SCALE,
                static_cast<double>(waves) * payloadBytes / LEARNED_BYTE_SCALE,
                static_cast<double>(waves) * stridedSpanBytes / LEARNED_BYTE_SCALE,
                static_cast<double>(waves) * dmaSegments / LEARNED_COUNT_SCALE,
                static_cast<double>(waves) * (horizontalRows + rows + CeilDiv(rows, LEARNED_ROW_GROUP)) /
                    LEARNED_COUNT_SCALE,
                static_cast<double>(waves) * rows *
                    (LEARNED_CENTER_TAP + problem.kernelHeight / LEARNED_RADIUS_DIVISOR) / LEARNED_COUNT_SCALE,
                static_cast<double>(waves) * horizontalRows * boundaryPixels * packedChannels / LEARNED_COUNT_SCALE,
                static_cast<double>(waves - LEARNED_BASE_WAVE_COUNT),
                static_cast<double>(waves * problem.coreCount - tasks) / (waves * problem.coreCount),
                static_cast<double>(waves) * horizontalRows * activeWidth * (packedChannels - realChannels) /
                    LEARNED_COUNT_SCALE,
                static_cast<double>(waves) * horizontalRows * vectorTail / LEARNED_COUNT_SCALE,
            };
            double score = 0.0;
            for (uint32_t index = 1U; index <= 9U; ++index) {
                score += values[index];
            }
            if (score > criticalScore) {
                critical = values;
                criticalScore = score;
            }
        }
    }
    return critical;
}

inline double EvaluateLearned(const Problem& problem, uint32_t tilesY)
{
    const LearnedCoefficients features = LearnedFeatures(problem, tilesY);
    const LearnedCoefficients& coefficients = CoefficientsFor(SelectLearnedFamily(problem));
    double total = 0.0;
    for (uint32_t index = 0U; index < LEARNED_FEATURE_COUNT; ++index) {
        total += features[index] * coefficients[index];
    }
    return total;
}

inline uint32_t SelectTilesY(const Problem& problem)
{
    const uint32_t tileWidth = LearnedTileWidth(problem);
    const uint32_t baseTasks = CeilDiv(problem.width, tileWidth) * CeilDiv(problem.channels, 8U);
    const std::array<uint32_t, 11> requestedCandidates = {1U,
                                                          2U,
                                                          3U,
                                                          4U,
                                                          5U,
                                                          8U,
                                                          16U,
                                                          32U / baseTasks,
                                                          CeilDiv(32U, baseTasks),
                                                          problem.coreCount / baseTasks,
                                                          CeilDiv(problem.coreCount, baseTasks)};
    uint32_t bestTilesY = 1U;
    double bestCost = std::numeric_limits<double>::max();
    std::array<uint32_t, 11> visited{};
    uint32_t visitedCount = 0U;
    for (const uint32_t requested : requestedCandidates) {
        const uint32_t candidate = ActualTilesY(problem.height, requested);
        bool duplicate = false;
        for (uint32_t index = 0U; index < visitedCount; ++index) {
            duplicate = duplicate || visited[index] == candidate;
        }
        if (duplicate) {
            continue;
        }
        visited[visitedCount++] = candidate;
        const double predicted = EvaluateLearned(problem, candidate);
        if (predicted < bestCost || (predicted == bestCost && candidate < bestTilesY)) {
            bestTilesY = candidate;
            bestCost = predicted;
        }
    }
    return bestTilesY;
}

} // namespace optiling::gaussian_blur_cost_model

#endif // GAUSSIAN_BLUR_TILING_COST_MODEL_H
