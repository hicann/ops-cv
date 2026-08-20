/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

namespace GAUSSIAN_BLUR_IMPL_NAMESPACE {

#ifdef GAUSSIAN_BLUR_ROW_SHARED_W
static constexpr uint32_t ROW_SHARED_W = GAUSSIAN_BLUR_ROW_SHARED_W;
#else
static constexpr uint32_t ROW_SHARED_W =
    (GAUSSIAN_BLUR_ROW_PATCHES + 2U) * GAUSSIAN_BLUR_ROW_BLOCK_X;
#endif
static constexpr uint32_t COLUMN_SHARED_H =
    (GAUSSIAN_BLUR_COLUMN_PATCHES + 2U) * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
static constexpr uint32_t COLUMN_TILE_H =
    GAUSSIAN_BLUR_COLUMN_PATCHES * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
static constexpr uint32_t ROW_SHARED_ELEMENTS = GAUSSIAN_BLUR_ROW_SHARED_ELEMENTS;
static constexpr uint32_t COLUMN_SHARED_ELEMENTS =
    COLUMN_SHARED_H * GAUSSIAN_BLUR_COLUMN_BLOCK_X * GAUSSIAN_BLUR_CHANNEL_TILE;
static constexpr uint32_t WEIGHT_UB_ELEMENTS =
    (GAUSSIAN_BLUR_KERNEL_MAX_SIZE * sizeof(float) + 31U) / 32U * (32U / sizeof(float));

struct PackedC3 {
    float x;
    float y;
    float z;
};

struct GaussianWeightParams {
    float values[GAUSSIAN_BLUR_KERNEL_MAX_SIZE];
};

static_assert(sizeof(PackedC3) == 3U * sizeof(float), "PackedC3 must match contiguous HWC C3 storage");

__simt_callee__ inline int32_t BorderCoord(int32_t coord, int32_t limit, uint32_t borderType)
{
    if (static_cast<uint32_t>(coord) < static_cast<uint32_t>(limit)) {
        return coord;
    }
    if (borderType == GAUSSIAN_BLUR_PADDING_CONSTANT) {
        return -1;
    }
    if (limit <= 1) {
        return 0;
    }
    if (borderType == GAUSSIAN_BLUR_PADDING_REPLICATE) {
        return coord < 0 ? 0 : limit - 1;
    }
    const int32_t delta = borderType == GAUSSIAN_BLUR_PADDING_REFLECT_101 ? 1 : 0;
    while (coord < 0 || coord >= limit) {
        if (coord < 0) {
            coord = -coord - 1 + delta;
        } else {
            coord = limit * 2 - coord - 1 - delta;
        }
    }
    return coord;
}

template <uint32_t ChannelStride, bool DynamicChannels>
__simt_callee__ inline void ResolveChannelTile(
    uint32_t tileId,
    uint32_t spatialTiles,
    uint32_t channels,
    uint32_t& spatialTileId,
    uint32_t& channelOffset,
    uint32_t& outputChannels)
{
    spatialTileId = tileId % spatialTiles;
    if constexpr (DynamicChannels) {
        const uint32_t channelTile = tileId / spatialTiles;
        channelOffset = channelTile * ChannelStride;
        outputChannels = channelOffset + ChannelStride <= channels ?
            ChannelStride : channels - channelOffset;
    } else {
        channelOffset = 0U;
        outputChannels = ChannelStride;
    }
}

template <uint32_t ChannelStride>
__simt_callee__ inline uint32_t RowSharedOffset(uint32_t localRow, uint32_t localX, uint32_t channel)
{
    return (localRow * ROW_SHARED_W + localX) * ChannelStride + channel;
}

template <uint32_t ChannelStride>
__simt_callee__ inline uint32_t ColumnSharedOffset(uint32_t localRow, uint32_t localX, uint32_t channel)
{
    return (localRow * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX) * ChannelStride + channel;
}

__simt_callee__ inline uint64_t PixelOffset(uint32_t y, uint32_t x, uint32_t width)
{
    return static_cast<uint64_t>(y) * width + x;
}

__simt_callee__ inline uint64_t ElementOffset(
    uint32_t y, uint32_t x, uint32_t width, uint32_t channels, uint32_t channelOffset)
{
    return PixelOffset(y, x, width) * channels + channelOffset;
}

__simt_callee__ inline uint64_t ChunkMajorOffset(
    uint32_t y,
    uint32_t x,
    uint32_t height,
    uint32_t width,
    uint32_t channelOffset,
    uint32_t activeChannels)
{
    const uint64_t pixels = static_cast<uint64_t>(height) * width;
    return static_cast<uint64_t>(channelOffset) * pixels +
        PixelOffset(y, x, width) * activeChannels;
}

// These scalar helpers are also used by non-SIMT AICore copy/store paths.
__aicore__ inline int32_t BorderCoordAicore(int32_t coord, int32_t limit, uint32_t borderType)
{
    if (static_cast<uint32_t>(coord) < static_cast<uint32_t>(limit)) {
        return coord;
    }
    if (borderType == GAUSSIAN_BLUR_PADDING_CONSTANT) {
        return -1;
    }
    if (limit <= 1) {
        return 0;
    }
    if (borderType == GAUSSIAN_BLUR_PADDING_REPLICATE) {
        return coord < 0 ? 0 : limit - 1;
    }
    const int32_t delta = borderType == GAUSSIAN_BLUR_PADDING_REFLECT_101 ? 1 : 0;
    while (coord < 0 || coord >= limit) {
        if (coord < 0) {
            coord = -coord - 1 + delta;
        } else {
            coord = limit * 2 - coord - 1 - delta;
        }
    }
    return coord;
}

__aicore__ inline uint64_t PixelOffsetAicore(uint32_t y, uint32_t x, uint32_t width)
{
    return static_cast<uint64_t>(y) * width + x;
}

__aicore__ inline uint64_t ElementOffsetAicore(
    uint32_t y, uint32_t x, uint32_t width, uint32_t channels, uint32_t channelOffset)
{
    return PixelOffsetAicore(y, x, width) * channels + channelOffset;
}

__aicore__ inline uint64_t ChunkMajorOffsetAicore(
    uint32_t y,
    uint32_t x,
    uint32_t height,
    uint32_t width,
    uint32_t channelOffset,
    uint32_t activeChannels)
{
    const uint64_t pixels = static_cast<uint64_t>(height) * width;
    return static_cast<uint64_t>(channelOffset) * pixels +
        PixelOffsetAicore(y, x, width) * activeChannels;
}

template <uint32_t ActiveChannels, uint32_t ChannelStride>
__simt_callee__ inline void LoadGenericRowChannels(
    uint32_t localRow,
    uint32_t sharedX,
    uint64_t sourceBase,
    bool sourceValid,
    __gm__ const float* src,
    __ubuf__ float* shared)
{
    static_assert(ActiveChannels >= 1U && ActiveChannels <= ChannelStride, "invalid generic channel count");
#pragma unroll
    for (uint32_t channel = 0U; channel < ActiveChannels; ++channel) {
        shared[RowSharedOffset<ChannelStride>(localRow, sharedX, channel)] =
            sourceValid ? src[sourceBase + channel] : 0.0f;
    }
}

template <uint32_t ActiveChannels, uint32_t ChannelStride>
__simt_callee__ inline void LoadGenericColumnChannels(
    uint32_t sharedY,
    uint32_t localX,
    uint64_t sourceBase,
    bool sourceValid,
    __gm__ const float* src,
    __ubuf__ float* shared)
{
    static_assert(ActiveChannels >= 1U && ActiveChannels <= ChannelStride, "invalid generic channel count");
#pragma unroll
    for (uint32_t channel = 0U; channel < ActiveChannels; ++channel) {
        shared[ColumnSharedOffset<ChannelStride>(sharedY, localX, channel)] =
            sourceValid ? src[sourceBase + channel] : 0.0f;
    }
}

template <uint32_t ChannelStride, bool DynamicChannels>
__simt_callee__ inline void LoadGenericRowGroup(
    uint32_t outputChannels,
    uint32_t localRow,
    uint32_t sharedX,
    uint64_t sourceBase,
    bool sourceValid,
    __gm__ const float* src,
    __ubuf__ float* shared)
{
    if constexpr (!DynamicChannels) {
        LoadGenericRowChannels<ChannelStride, ChannelStride>(
            localRow, sharedX, sourceBase, sourceValid, src, shared);
    } else if constexpr (ChannelStride >= 8U) {
        if (outputChannels == 8U) {
            LoadGenericRowChannels<8U, ChannelStride>(localRow, sharedX, sourceBase, sourceValid, src, shared);
        } else if (outputChannels == 7U) {
            LoadGenericRowChannels<7U, ChannelStride>(localRow, sharedX, sourceBase, sourceValid, src, shared);
        } else if (outputChannels == 6U) {
            LoadGenericRowChannels<6U, ChannelStride>(localRow, sharedX, sourceBase, sourceValid, src, shared);
        } else {
            LoadGenericRowChannels<5U, ChannelStride>(localRow, sharedX, sourceBase, sourceValid, src, shared);
        }
    } else if (outputChannels == 4U) {
        auto* shared4 = reinterpret_cast<__ubuf__ float4*>(shared);
        const uint32_t sharedOffset = localRow * ROW_SHARED_W + sharedX;
        if (!sourceValid) {
            shared4[sharedOffset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else if ((sourceBase & 3U) == 0U) {
            auto* src4 = reinterpret_cast<__gm__ const float4*>(src);
            shared4[sharedOffset] = src4[sourceBase / 4U];
        } else {
            shared4[sharedOffset] = make_float4(
                src[sourceBase], src[sourceBase + 1U], src[sourceBase + 2U], src[sourceBase + 3U]);
        }
    } else if (outputChannels == 3U) {
        LoadGenericRowChannels<3U, ChannelStride>(localRow, sharedX, sourceBase, sourceValid, src, shared);
    } else if (outputChannels == 2U) {
        LoadGenericRowChannels<2U, ChannelStride>(localRow, sharedX, sourceBase, sourceValid, src, shared);
    } else {
        LoadGenericRowChannels<1U, ChannelStride>(localRow, sharedX, sourceBase, sourceValid, src, shared);
    }
}

template <uint32_t ChannelStride, bool DynamicChannels>
__simt_callee__ inline void LoadGenericColumnGroup(
    uint32_t outputChannels,
    uint32_t sharedY,
    uint32_t localX,
    uint64_t sourceBase,
    bool sourceValid,
    __gm__ const float* src,
    __ubuf__ float* shared)
{
    if constexpr (!DynamicChannels) {
        LoadGenericColumnChannels<ChannelStride, ChannelStride>(
            sharedY, localX, sourceBase, sourceValid, src, shared);
    } else if constexpr (ChannelStride >= 8U) {
        if (outputChannels == 8U) {
            LoadGenericColumnChannels<8U, ChannelStride>(sharedY, localX, sourceBase, sourceValid, src, shared);
        } else if (outputChannels == 7U) {
            LoadGenericColumnChannels<7U, ChannelStride>(sharedY, localX, sourceBase, sourceValid, src, shared);
        } else if (outputChannels == 6U) {
            LoadGenericColumnChannels<6U, ChannelStride>(sharedY, localX, sourceBase, sourceValid, src, shared);
        } else {
            LoadGenericColumnChannels<5U, ChannelStride>(sharedY, localX, sourceBase, sourceValid, src, shared);
        }
    } else if (outputChannels == 4U) {
        auto* shared4 = reinterpret_cast<__ubuf__ float4*>(shared);
        const uint32_t sharedOffset = sharedY * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX;
        if (!sourceValid) {
            shared4[sharedOffset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else if ((sourceBase & 3U) == 0U) {
            auto* src4 = reinterpret_cast<__gm__ const float4*>(src);
            shared4[sharedOffset] = src4[sourceBase / 4U];
        } else {
            shared4[sharedOffset] = make_float4(
                src[sourceBase], src[sourceBase + 1U], src[sourceBase + 2U], src[sourceBase + 3U]);
        }
    } else if (outputChannels == 3U) {
        LoadGenericColumnChannels<3U, ChannelStride>(sharedY, localX, sourceBase, sourceValid, src, shared);
    } else if (outputChannels == 2U) {
        LoadGenericColumnChannels<2U, ChannelStride>(sharedY, localX, sourceBase, sourceValid, src, shared);
    } else {
        LoadGenericColumnChannels<1U, ChannelStride>(sharedY, localX, sourceBase, sourceValid, src, shared);
    }
}

template <uint32_t ActiveChannels, uint32_t ChannelStride, uint32_t KernelSize>
__simt_callee__ inline void ComputeGenericRowChannels(
    uint32_t kernelSize,
    uint32_t localRow,
    uint32_t sharedCenter,
    uint32_t anchor,
    uint64_t outputBase,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    for (uint32_t channel = 0U; channel < ActiveChannels; ++channel) {
        float sum = 0.0f;
        if constexpr (KernelSize == 0U) {
            uint32_t kernelIndex = 0U;
            for (; kernelIndex + 3U < kernelSize; kernelIndex += 4U) {
#pragma unroll
                for (uint32_t offset = 0U; offset < 4U; ++offset) {
                    sum += shared[RowSharedOffset<ChannelStride>(
                               localRow, sharedCenter - anchor + kernelIndex + offset, channel)] *
                        weights[kernelIndex + offset];
                }
            }
            for (; kernelIndex < kernelSize; ++kernelIndex) {
                sum += shared[RowSharedOffset<ChannelStride>(
                           localRow, sharedCenter - anchor + kernelIndex, channel)] * weights[kernelIndex];
            }
        } else {
#pragma unroll
            for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
                sum += shared[RowSharedOffset<ChannelStride>(
                           localRow, sharedCenter - anchor + kernelIndex, channel)] * weights[kernelIndex];
            }
        }
        dst[outputBase + channel] = sum;
    }
}

template <uint32_t ActiveChannels, uint32_t ChannelStride, uint32_t KernelSize>
__simt_callee__ inline void ComputeGenericColumnChannels(
    uint32_t kernelSize,
    uint32_t localX,
    uint32_t sharedCenter,
    uint32_t anchor,
    uint64_t outputBase,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    for (uint32_t channel = 0U; channel < ActiveChannels; ++channel) {
        float sum = 0.0f;
        if constexpr (KernelSize == 0U) {
            uint32_t kernelIndex = 0U;
            for (; kernelIndex + 3U < kernelSize; kernelIndex += 4U) {
#pragma unroll
                for (uint32_t offset = 0U; offset < 4U; ++offset) {
                    sum += shared[ColumnSharedOffset<ChannelStride>(
                               sharedCenter - anchor + kernelIndex + offset, localX, channel)] *
                        weights[kernelIndex + offset];
                }
            }
            for (; kernelIndex < kernelSize; ++kernelIndex) {
                sum += shared[ColumnSharedOffset<ChannelStride>(
                           sharedCenter - anchor + kernelIndex, localX, channel)] * weights[kernelIndex];
            }
        } else {
#pragma unroll
            for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
                sum += shared[ColumnSharedOffset<ChannelStride>(
                           sharedCenter - anchor + kernelIndex, localX, channel)] * weights[kernelIndex];
            }
        }
        dst[outputBase + channel] = sum;
    }
}

template <uint32_t KernelSize>
__simt_callee__ inline void ComputeGenericRowC8Subgroup(
    uint32_t outputChannels,
    uint32_t kernelSize,
    uint32_t localRow,
    uint32_t sharedCenter,
    uint32_t anchor,
    uint64_t outputBase,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    const uint32_t subgroupOffset = threadIdx.z * 4U;
    if (subgroupOffset >= outputChannels) {
        return;
    }
    auto* shared4 = reinterpret_cast<__ubuf__ const float4*>(shared);
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const uint32_t count = KernelSize == 0U ? kernelSize : KernelSize;
    for (uint32_t kernelIndex = 0U; kernelIndex < count; ++kernelIndex) {
        const uint32_t pixelBase =
            (localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex) * 2U + threadIdx.z;
        const float4 pixel = shared4[pixelBase];
        const float weight = weights[kernelIndex];
        sum.x += pixel.x * weight;
        sum.y += pixel.y * weight;
        sum.z += pixel.z * weight;
        sum.w += pixel.w * weight;
    }
    const uint32_t active = outputChannels - subgroupOffset;
    dst[outputBase + subgroupOffset] = sum.x;
    if (active >= 2U) dst[outputBase + subgroupOffset + 1U] = sum.y;
    if (active >= 3U) dst[outputBase + subgroupOffset + 2U] = sum.z;
    if (active >= 4U) dst[outputBase + subgroupOffset + 3U] = sum.w;
}

template <uint32_t KernelSize>
__simt_callee__ inline void ComputeGenericRowC8ChunkMajorSubgroup(
    uint32_t height,
    uint32_t width,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t outputY,
    uint32_t outputX,
    uint32_t kernelSize,
    uint32_t localRow,
    uint32_t sharedCenter,
    uint32_t anchor,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    const uint32_t subgroupOffset = threadIdx.z * GAUSSIAN_BLUR_CHANNEL_TILE;
    if (subgroupOffset >= outputChannels) {
        return;
    }
    auto* shared4 = reinterpret_cast<__ubuf__ const float4*>(shared);
    float4 sum;
    if constexpr (KernelSize == 31U) {
        const uint32_t centerBase =
            (localRow * ROW_SHARED_W + sharedCenter) * 2U + threadIdx.z;
        const float4 center = shared4[centerBase];
        const float centerWeight = weights[15U];
        sum = make_float4(
            center.x * centerWeight, center.y * centerWeight,
            center.z * centerWeight, center.w * centerWeight);
        for (uint32_t offset = 1U; offset <= 15U; ++offset) {
            const float4 left = shared4[centerBase - offset * 2U];
            const float4 right = shared4[centerBase + offset * 2U];
            const float weight = weights[15U - offset];
            sum.x += (left.x + right.x) * weight;
            sum.y += (left.y + right.y) * weight;
            sum.z += (left.z + right.z) * weight;
            sum.w += (left.w + right.w) * weight;
        }
    } else {
        sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const uint32_t count = KernelSize == 0U ? kernelSize : KernelSize;
        for (uint32_t kernelIndex = 0U; kernelIndex < count; ++kernelIndex) {
            const uint32_t pixelBase =
                (localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex) * 2U + threadIdx.z;
            const float4 pixel = shared4[pixelBase];
            const float weight = weights[kernelIndex];
            sum.x += pixel.x * weight;
            sum.y += pixel.y * weight;
            sum.z += pixel.z * weight;
            sum.w += pixel.w * weight;
        }
    }
    const uint32_t active = outputChannels - subgroupOffset < GAUSSIAN_BLUR_CHANNEL_TILE ?
        outputChannels - subgroupOffset : GAUSSIAN_BLUR_CHANNEL_TILE;
    const uint64_t outputBase = ChunkMajorOffset(
        outputY, outputX, height, width, channelOffset + subgroupOffset, active);
    dst[outputBase] = sum.x;
    if (active >= 2U) dst[outputBase + 1U] = sum.y;
    if (active >= 3U) dst[outputBase + 2U] = sum.z;
    if (active >= 4U) dst[outputBase + 3U] = sum.w;
}

template <uint32_t ChannelStride, bool DynamicChannels, uint32_t KernelSize>
__simt_callee__ inline void ComputeGenericRowGroup(
    uint32_t outputChannels,
    uint32_t kernelSize,
    uint32_t localRow,
    uint32_t sharedCenter,
    uint32_t anchor,
    uint64_t outputBase,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    if constexpr (!DynamicChannels) {
        ComputeGenericRowChannels<ChannelStride, ChannelStride, KernelSize>(
            kernelSize, localRow, sharedCenter, anchor, outputBase, weights, shared, dst);
    } else if constexpr (ChannelStride >= 8U) {
        if (outputChannels == 8U) {
            ComputeGenericRowChannels<8U, ChannelStride, KernelSize>(
                kernelSize, localRow, sharedCenter, anchor, outputBase, weights, shared, dst);
        } else if (outputChannels == 7U) {
            ComputeGenericRowChannels<7U, ChannelStride, KernelSize>(
                kernelSize, localRow, sharedCenter, anchor, outputBase, weights, shared, dst);
        } else if (outputChannels == 6U) {
            ComputeGenericRowChannels<6U, ChannelStride, KernelSize>(
                kernelSize, localRow, sharedCenter, anchor, outputBase, weights, shared, dst);
        } else {
            ComputeGenericRowChannels<5U, ChannelStride, KernelSize>(
                kernelSize, localRow, sharedCenter, anchor, outputBase, weights, shared, dst);
        }
    } else if (outputChannels == 4U) {
        auto* shared4 = reinterpret_cast<__ubuf__ const float4*>(shared);
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if constexpr (KernelSize == 0U) {
            uint32_t kernelIndex = 0U;
            for (; kernelIndex + 3U < kernelSize; kernelIndex += 4U) {
#pragma unroll
                for (uint32_t offset = 0U; offset < 4U; ++offset) {
                    const float4 pixel = shared4[
                        localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex + offset];
                    const float weight = weights[kernelIndex + offset];
                    sum.x += pixel.x * weight;
                    sum.y += pixel.y * weight;
                    sum.z += pixel.z * weight;
                    sum.w += pixel.w * weight;
                }
            }
            for (; kernelIndex < kernelSize; ++kernelIndex) {
                const float4 pixel = shared4[localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex];
                const float weight = weights[kernelIndex];
                sum.x += pixel.x * weight;
                sum.y += pixel.y * weight;
                sum.z += pixel.z * weight;
                sum.w += pixel.w * weight;
            }
        } else if constexpr (KernelSize == 31U) {
            const float4 center = shared4[localRow * ROW_SHARED_W + sharedCenter];
            const float centerWeight = weights[anchor];
            sum = make_float4(
                center.x * centerWeight, center.y * centerWeight,
                center.z * centerWeight, center.w * centerWeight);
#pragma unroll
            for (uint32_t offset = 1U; offset <= anchor; ++offset) {
                const float4 left = shared4[localRow * ROW_SHARED_W + sharedCenter - offset];
                const float4 right = shared4[localRow * ROW_SHARED_W + sharedCenter + offset];
                const float weight = weights[anchor - offset];
                sum.x += (left.x + right.x) * weight;
                sum.y += (left.y + right.y) * weight;
                sum.z += (left.z + right.z) * weight;
                sum.w += (left.w + right.w) * weight;
            }
        } else {
#pragma unroll
            for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
                const float4 pixel = shared4[
                    localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex];
                const float weight = weights[kernelIndex];
                sum.x += pixel.x * weight;
                sum.y += pixel.y * weight;
                sum.z += pixel.z * weight;
                sum.w += pixel.w * weight;
            }
        }
        if ((outputBase & 3U) == 0U) {
            auto* dst4 = reinterpret_cast<__gm__ float4*>(dst);
            dst4[outputBase / 4U] = sum;
        } else {
            dst[outputBase] = sum.x;
            dst[outputBase + 1U] = sum.y;
            dst[outputBase + 2U] = sum.z;
            dst[outputBase + 3U] = sum.w;
        }
    } else if (outputChannels == 3U) {
        ComputeGenericRowChannels<3U, ChannelStride, KernelSize>(
            kernelSize, localRow, sharedCenter, anchor, outputBase, weights, shared, dst);
    } else if (outputChannels == 2U) {
        ComputeGenericRowChannels<2U, ChannelStride, KernelSize>(
            kernelSize, localRow, sharedCenter, anchor, outputBase, weights, shared, dst);
    } else {
        ComputeGenericRowChannels<1U, ChannelStride, KernelSize>(
            kernelSize, localRow, sharedCenter, anchor, outputBase, weights, shared, dst);
    }
}

template <uint32_t ChannelStride, bool DynamicChannels, uint32_t KernelSize>
__simt_callee__ inline void ComputeGenericColumnGroup(
    uint32_t outputChannels,
    uint32_t kernelSize,
    uint32_t localX,
    uint32_t sharedCenter,
    uint32_t anchor,
    uint64_t outputBase,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    if constexpr (!DynamicChannels) {
        ComputeGenericColumnChannels<ChannelStride, ChannelStride, KernelSize>(
            kernelSize, localX, sharedCenter, anchor, outputBase, weights, shared, dst);
    } else if constexpr (ChannelStride >= 8U) {
        if (outputChannels == 8U) {
            ComputeGenericColumnChannels<8U, ChannelStride, KernelSize>(
                kernelSize, localX, sharedCenter, anchor, outputBase, weights, shared, dst);
        } else if (outputChannels == 7U) {
            ComputeGenericColumnChannels<7U, ChannelStride, KernelSize>(
                kernelSize, localX, sharedCenter, anchor, outputBase, weights, shared, dst);
        } else if (outputChannels == 6U) {
            ComputeGenericColumnChannels<6U, ChannelStride, KernelSize>(
                kernelSize, localX, sharedCenter, anchor, outputBase, weights, shared, dst);
        } else {
            ComputeGenericColumnChannels<5U, ChannelStride, KernelSize>(
                kernelSize, localX, sharedCenter, anchor, outputBase, weights, shared, dst);
        }
    } else if (outputChannels == 4U) {
        auto* shared4 = reinterpret_cast<__ubuf__ const float4*>(shared);
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if constexpr (KernelSize == 0U) {
            uint32_t kernelIndex = 0U;
            for (; kernelIndex + 3U < kernelSize; kernelIndex += 4U) {
#pragma unroll
                for (uint32_t offset = 0U; offset < 4U; ++offset) {
                    const float4 pixel = shared4[
                        (sharedCenter - anchor + kernelIndex + offset) *
                            GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX];
                    const float weight = weights[kernelIndex + offset];
                    sum.x += pixel.x * weight;
                    sum.y += pixel.y * weight;
                    sum.z += pixel.z * weight;
                    sum.w += pixel.w * weight;
                }
            }
            for (; kernelIndex < kernelSize; ++kernelIndex) {
                const float4 pixel = shared4[
                    (sharedCenter - anchor + kernelIndex) * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX];
                const float weight = weights[kernelIndex];
                sum.x += pixel.x * weight;
                sum.y += pixel.y * weight;
                sum.z += pixel.z * weight;
                sum.w += pixel.w * weight;
            }
        } else {
#pragma unroll
            for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
                const float4 pixel = shared4[
                    (sharedCenter - anchor + kernelIndex) * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX];
                const float weight = weights[kernelIndex];
                sum.x += pixel.x * weight;
                sum.y += pixel.y * weight;
                sum.z += pixel.z * weight;
                sum.w += pixel.w * weight;
            }
        }
        if ((outputBase & 3U) == 0U) {
            auto* dst4 = reinterpret_cast<__gm__ float4*>(dst);
            dst4[outputBase / 4U] = sum;
        } else {
            dst[outputBase] = sum.x;
            dst[outputBase + 1U] = sum.y;
            dst[outputBase + 2U] = sum.z;
            dst[outputBase + 3U] = sum.w;
        }
    } else if (outputChannels == 3U) {
        ComputeGenericColumnChannels<3U, ChannelStride, KernelSize>(
            kernelSize, localX, sharedCenter, anchor, outputBase, weights, shared, dst);
    } else if (outputChannels == 2U) {
        ComputeGenericColumnChannels<2U, ChannelStride, KernelSize>(
            kernelSize, localX, sharedCenter, anchor, outputBase, weights, shared, dst);
    } else {
        ComputeGenericColumnChannels<1U, ChannelStride, KernelSize>(
            kernelSize, localX, sharedCenter, anchor, outputBase, weights, shared, dst);
    }
}

template <uint32_t KernelSize>
__simt_callee__ inline void ComputeGenericColumnGroupSymmetric(
    uint32_t localX,
    uint32_t sharedCenter,
    uint64_t outputBase,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    static_assert(KernelSize > 1U && (KernelSize & 1U) == 1U);
    constexpr uint32_t anchor = (KernelSize - 1U) / 2U;
    auto* shared4 = reinterpret_cast<__ubuf__ const float4*>(shared);
    const float4 center =
        shared4[sharedCenter * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX];
    const float centerWeight = weights[anchor];
    float4 sum0 = make_float4(
        center.x * centerWeight, center.y * centerWeight,
        center.z * centerWeight, center.w * centerWeight);
    float4 sum1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (uint32_t offset = 1U; offset <= anchor; ++offset) {
        const float4 upper = shared4[
            (sharedCenter - offset) * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX];
        const float4 lower = shared4[
            (sharedCenter + offset) * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX];
        const float weight = weights[anchor - offset];
        if ((offset & 1U) == 0U) {
            sum0.x += (upper.x + lower.x) * weight;
            sum0.y += (upper.y + lower.y) * weight;
            sum0.z += (upper.z + lower.z) * weight;
            sum0.w += (upper.w + lower.w) * weight;
        } else {
            sum1.x += (upper.x + lower.x) * weight;
            sum1.y += (upper.y + lower.y) * weight;
            sum1.z += (upper.z + lower.z) * weight;
            sum1.w += (upper.w + lower.w) * weight;
        }
    }
    const float4 sum = make_float4(
        sum0.x + sum1.x, sum0.y + sum1.y,
        sum0.z + sum1.z, sum0.w + sum1.w);
    if ((outputBase & 3U) == 0U) {
        auto* dst4 = reinterpret_cast<__gm__ float4*>(dst);
        dst4[outputBase / 4U] = sum;
    } else {
        dst[outputBase] = sum.x;
        dst[outputBase + 1U] = sum.y;
        dst[outputBase + 2U] = sum.z;
        dst[outputBase + 3U] = sum.w;
    }
}

__simt_callee__ inline void ComputeGenericColumnGroupK31Symmetric(
    uint32_t localX,
    uint32_t sharedCenter,
    uint64_t outputBase,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    auto* shared4 = reinterpret_cast<__ubuf__ const float4*>(shared);
    const float4 center = shared4[
        sharedCenter * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX];
    const float centerWeight = weights[15U];
    float4 sum = make_float4(
        center.x * centerWeight, center.y * centerWeight,
        center.z * centerWeight, center.w * centerWeight);
#pragma unroll
    for (uint32_t offset = 1U; offset <= 15U; ++offset) {
        const float4 upper = shared4[
            (sharedCenter - offset) * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX];
        const float4 lower = shared4[
            (sharedCenter + offset) * GAUSSIAN_BLUR_COLUMN_BLOCK_X + localX];
        const float weight = weights[15U - offset];
        sum.x += (upper.x + lower.x) * weight;
        sum.y += (upper.y + lower.y) * weight;
        sum.z += (upper.z + lower.z) * weight;
        sum.w += (upper.w + lower.w) * weight;
    }
    if ((outputBase & 3U) == 0U) {
        auto* dst4 = reinterpret_cast<__gm__ float4*>(dst);
        dst4[outputBase / 4U] = sum;
    } else {
        dst[outputBase] = sum.x;
        dst[outputBase + 1U] = sum.y;
        dst[outputBase + 2U] = sum.z;
        dst[outputBase + 3U] = sum.w;
    }
}

template <uint32_t ChannelStride, bool DynamicChannels, bool CheckBounds>
__simt_callee__ inline void LoadRowSegment(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t outputY,
    uint32_t localRow,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t borderType,
    int32_t segmentBaseX,
    uint32_t sharedBaseX,
    __gm__ const float* src,
    __ubuf__ float* shared)
{
    const int32_t rawX = segmentBaseX + static_cast<int32_t>(threadIdx.x);
    int32_t sourceX = -1;
    if (outputY < height) {
        if constexpr (CheckBounds) {
            const bool segmentInBounds = segmentBaseX >= 0 &&
                segmentBaseX + static_cast<int32_t>(GAUSSIAN_BLUR_ROW_BLOCK_X) <=
                    static_cast<int32_t>(width);
            sourceX = segmentInBounds ? rawX : BorderCoord(rawX, static_cast<int32_t>(width), borderType);
        } else {
            sourceX = rawX;
        }
    }
    const uint32_t sharedX = sharedBaseX + threadIdx.x;

    if constexpr (ChannelStride == 4U && !DynamicChannels) {
        auto* src4 = reinterpret_cast<__gm__ const float4*>(src);
        auto* shared4 = reinterpret_cast<__ubuf__ float4*>(shared);
        const uint32_t sharedOffset = localRow * ROW_SHARED_W + sharedX;
        shared4[sharedOffset] = sourceX >= 0 ?
            src4[PixelOffset(outputY, static_cast<uint32_t>(sourceX), width)] :
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else if constexpr (ChannelStride == 3U && !DynamicChannels) {
        auto* src3 = reinterpret_cast<__gm__ const PackedC3*>(src);
        auto* shared3 = reinterpret_cast<__ubuf__ PackedC3*>(shared);
        const uint32_t sharedOffset = localRow * ROW_SHARED_W + sharedX;
        if (sourceX >= 0) {
            const uint64_t sourceOffset = PixelOffset(outputY, static_cast<uint32_t>(sourceX), width);
            shared3[sharedOffset].x = src3[sourceOffset].x;
            shared3[sharedOffset].y = src3[sourceOffset].y;
            shared3[sharedOffset].z = src3[sourceOffset].z;
        } else {
            shared3[sharedOffset].x = 0.0f;
            shared3[sharedOffset].y = 0.0f;
            shared3[sharedOffset].z = 0.0f;
        }
    } else {
        const bool sourceValid = sourceX >= 0;
        const uint64_t sourceBase = sourceValid ?
            ElementOffset(outputY, static_cast<uint32_t>(sourceX), width, channels, channelOffset) : 0U;
        LoadGenericRowGroup<ChannelStride, DynamicChannels>(
            outputChannels, localRow, sharedX, sourceBase, sourceValid, src, shared);
    }
}

template <uint32_t ChannelStride, bool DynamicChannels, bool CheckBounds>
__simt_callee__ inline void LoadColumnSegment(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t outputX,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t borderType,
    int32_t segmentBaseY,
    uint32_t sharedBaseY,
    __gm__ const float* src,
    __ubuf__ float* shared)
{
    const int32_t rawY = segmentBaseY + static_cast<int32_t>(threadIdx.y);
    int32_t sourceY = -1;
    if (outputX < width) {
        if constexpr (CheckBounds) {
            const bool segmentInBounds = segmentBaseY >= 0 &&
                segmentBaseY + static_cast<int32_t>(GAUSSIAN_BLUR_COLUMN_BLOCK_Y) <=
                    static_cast<int32_t>(height);
            sourceY = segmentInBounds ? rawY : BorderCoord(rawY, static_cast<int32_t>(height), borderType);
        } else {
            sourceY = rawY;
        }
    }
    const uint32_t sharedY = sharedBaseY + threadIdx.y;

    if constexpr (ChannelStride == 4U && !DynamicChannels) {
        auto* src4 = reinterpret_cast<__gm__ const float4*>(src);
        auto* shared4 = reinterpret_cast<__ubuf__ float4*>(shared);
        const uint32_t sharedOffset =
            sharedY * GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x;
        shared4[sharedOffset] = sourceY >= 0 ?
            src4[PixelOffset(static_cast<uint32_t>(sourceY), outputX, width)] :
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else if constexpr (ChannelStride == 3U && !DynamicChannels) {
        auto* src3 = reinterpret_cast<__gm__ const PackedC3*>(src);
        auto* shared3 = reinterpret_cast<__ubuf__ PackedC3*>(shared);
        const uint32_t sharedOffset =
            sharedY * GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x;
        if (sourceY >= 0) {
            const uint64_t sourceOffset = PixelOffset(static_cast<uint32_t>(sourceY), outputX, width);
            shared3[sharedOffset].x = src3[sourceOffset].x;
            shared3[sharedOffset].y = src3[sourceOffset].y;
            shared3[sharedOffset].z = src3[sourceOffset].z;
        } else {
            shared3[sharedOffset].x = 0.0f;
            shared3[sharedOffset].y = 0.0f;
            shared3[sharedOffset].z = 0.0f;
        }
    } else {
        const bool sourceValid = sourceY >= 0;
        const uint64_t sourceBase = sourceValid ?
            (ChannelStride == GAUSSIAN_BLUR_CHANNEL_TILE && DynamicChannels ?
                ChunkMajorOffset(static_cast<uint32_t>(sourceY), outputX, height, width,
                    channelOffset, outputChannels) :
                ElementOffset(static_cast<uint32_t>(sourceY), outputX, width, channels, channelOffset)) : 0U;
        LoadGenericColumnGroup<ChannelStride, DynamicChannels>(
            outputChannels, sharedY, threadIdx.x, sourceBase, sourceValid, src, shared);
    }
}

template <uint32_t ChannelStride, bool DynamicChannels, bool CheckBounds>
__simt_callee__ inline void LoadRowTile(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t outputY,
    uint32_t localRow,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t borderType,
    uint32_t tileBaseX,
    __gm__ const float* src,
    __ubuf__ float* shared)
{
    LoadRowSegment<ChannelStride, DynamicChannels, CheckBounds>(
        height, width, channels, outputY, localRow, channelOffset, outputChannels, borderType,
        static_cast<int32_t>(tileBaseX) - static_cast<int32_t>(GAUSSIAN_BLUR_ROW_BLOCK_X),
        0U, src, shared);
#pragma unroll
    for (uint32_t patch = 0U; patch < GAUSSIAN_BLUR_ROW_PATCHES; ++patch) {
        LoadRowSegment<ChannelStride, DynamicChannels, CheckBounds>(
            height, width, channels, outputY, localRow, channelOffset, outputChannels, borderType,
            static_cast<int32_t>(tileBaseX + patch * GAUSSIAN_BLUR_ROW_BLOCK_X),
            (patch + 1U) * GAUSSIAN_BLUR_ROW_BLOCK_X, src, shared);
    }
    LoadRowSegment<ChannelStride, DynamicChannels, CheckBounds>(
        height, width, channels, outputY, localRow, channelOffset, outputChannels, borderType,
        static_cast<int32_t>(tileBaseX + GAUSSIAN_BLUR_ROW_TILE_W),
        (GAUSSIAN_BLUR_ROW_PATCHES + 1U) * GAUSSIAN_BLUR_ROW_BLOCK_X, src, shared);
}

template <uint32_t ChannelStride, bool DynamicChannels, bool CheckBounds>
__simt_callee__ inline void LoadColumnTile(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t outputX,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t borderType,
    uint32_t tileBaseY,
    __gm__ const float* src,
    __ubuf__ float* shared)
{
    LoadColumnSegment<ChannelStride, DynamicChannels, CheckBounds>(
        height, width, channels, outputX, channelOffset, outputChannels, borderType,
        static_cast<int32_t>(tileBaseY) - static_cast<int32_t>(GAUSSIAN_BLUR_COLUMN_BLOCK_Y),
        0U, src, shared);
#pragma unroll
    for (uint32_t patch = 0U; patch < GAUSSIAN_BLUR_COLUMN_PATCHES; ++patch) {
        LoadColumnSegment<ChannelStride, DynamicChannels, CheckBounds>(
            height, width, channels, outputX, channelOffset, outputChannels, borderType,
            static_cast<int32_t>(tileBaseY + patch * GAUSSIAN_BLUR_COLUMN_BLOCK_Y),
            (patch + 1U) * GAUSSIAN_BLUR_COLUMN_BLOCK_Y, src, shared);
    }
    LoadColumnSegment<ChannelStride, DynamicChannels, CheckBounds>(
        height, width, channels, outputX, channelOffset, outputChannels, borderType,
        static_cast<int32_t>(tileBaseY + COLUMN_TILE_H),
        (GAUSSIAN_BLUR_COLUMN_PATCHES + 1U) * GAUSSIAN_BLUR_COLUMN_BLOCK_Y, src, shared);
}

template <uint32_t ChannelStride, bool DynamicChannels, uint32_t KernelSize, bool EdgeOnly,
          typename WeightStorage>
__simt_callee__ inline void RunRowTiles(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    WeightStorage weights,
    __gm__ const float* src,
    __gm__ float* dst,
    __ubuf__ float* shared)
{
    const uint32_t anchor = KernelSize == 0U ? kernelSize / 2U : (KernelSize - 1U) / 2U;
    const uint32_t spatialTiles = tilesX * tilesY;
    auto* src4 = reinterpret_cast<__gm__ const float4*>(src);
    auto* dst4 = reinterpret_cast<__gm__ float4*>(dst);
    auto* shared4 = reinterpret_cast<__ubuf__ float4*>(shared);
    auto* src3 = reinterpret_cast<__gm__ const PackedC3*>(src);
    auto* dst3 = reinterpret_cast<__gm__ PackedC3*>(dst);
    auto* shared3 = reinterpret_cast<__ubuf__ PackedC3*>(shared);

    for (uint32_t tileId = coreIndex; tileId < totalTiles; tileId += coreCount) {
        uint32_t spatialTileId = 0U;
        uint32_t channelOffset = 0U;
        uint32_t outputChannels = 0U;
        ResolveChannelTile<ChannelStride, DynamicChannels>(
            tileId, spatialTiles, channels, spatialTileId, channelOffset, outputChannels);
        const uint32_t tileX = spatialTileId % tilesX;
        const uint32_t tileY = spatialTileId / tilesX;
        const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_ROW_TILE_W;
        const int32_t xStart = static_cast<int32_t>(tileBaseX + threadIdx.x);
        const bool interiorX = tileBaseX >= GAUSSIAN_BLUR_ROW_BLOCK_X &&
            tileBaseX + ROW_SHARED_W - GAUSSIAN_BLUR_ROW_BLOCK_X <= width;
        if constexpr (EdgeOnly) {
            const uint32_t channelGroupOffset = channelOffset - channelOffset %
                GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP;
            const bool handledByInteriorPipeline = !DynamicChannels ||
                (ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP ?
                    channelOffset + outputChannels == channels :
                    outputChannels == ChannelStride &&
                        channelGroupOffset + GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP <= channels);
            if (handledByInteriorPipeline && interiorX &&
                tileY * GAUSSIAN_BLUR_ROW_TILE_H + GAUSSIAN_BLUR_ROW_TILE_H <= height) {
                continue;
            }
        }
        constexpr bool c8Subgroups =
            ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP && DynamicChannels;
        constexpr uint32_t rowIterations = c8Subgroups ? 2U : 1U;
        for (uint32_t rowIteration = 0U; rowIteration < rowIterations; ++rowIteration) {
        const uint32_t localRow = threadIdx.y +
            rowIteration * (GAUSSIAN_BLUR_ROW_TILE_H / rowIterations);
        const uint32_t outputY = tileY * GAUSSIAN_BLUR_ROW_TILE_H + localRow;
        if (!c8Subgroups || threadIdx.z == 0U) {
            if (interiorX) {
                for (uint32_t segment = 0U; segment < GAUSSIAN_BLUR_ROW_PATCHES + 2U; ++segment) {
                    const uint32_t sourceX = tileBaseX - GAUSSIAN_BLUR_ROW_BLOCK_X + threadIdx.x +
                        segment * GAUSSIAN_BLUR_ROW_BLOCK_X;
                    const uint32_t sharedX = threadIdx.x + segment * GAUSSIAN_BLUR_ROW_BLOCK_X;
                    if constexpr (ChannelStride == 4U && !DynamicChannels) {
                        shared4[localRow * ROW_SHARED_W + sharedX] = outputY < height ?
                            src4[PixelOffset(outputY, sourceX, width)] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    } else if constexpr (ChannelStride == 3U && !DynamicChannels) {
                        const uint32_t sharedOffset = localRow * ROW_SHARED_W + sharedX;
                        if (outputY < height) {
                            const uint64_t sourceOffset = PixelOffset(outputY, sourceX, width);
                            shared3[sharedOffset].x = src3[sourceOffset].x;
                            shared3[sharedOffset].y = src3[sourceOffset].y;
                            shared3[sharedOffset].z = src3[sourceOffset].z;
                        } else {
                            shared3[sharedOffset].x = 0.0f;
                            shared3[sharedOffset].y = 0.0f;
                            shared3[sharedOffset].z = 0.0f;
                        }
                    } else {
                        const bool sourceValid = outputY < height;
                        const uint64_t sourceBase = sourceValid ?
                            ElementOffset(outputY, sourceX, width, channels, channelOffset) : 0U;
                        LoadGenericRowGroup<ChannelStride, DynamicChannels>(
                            outputChannels, localRow, sharedX, sourceBase, sourceValid, src, shared);
                    }
                }
            } else {
                LoadRowTile<ChannelStride, DynamicChannels, true>(
                    height, width, channels, outputY, localRow, channelOffset, outputChannels,
                    borderType, tileBaseX, src, shared);
            }
        }
        asc_syncthreads();

        for (uint32_t patch = 0U; patch < GAUSSIAN_BLUR_ROW_PATCHES; ++patch) {
            const uint32_t outputX = static_cast<uint32_t>(xStart) +
                patch * GAUSSIAN_BLUR_ROW_BLOCK_X;
            if (outputY < height && outputX < width) {
                const uint32_t sharedCenter = threadIdx.x + GAUSSIAN_BLUR_ROW_BLOCK_X +
                    patch * GAUSSIAN_BLUR_ROW_BLOCK_X;
                if constexpr (ChannelStride == 4U && !DynamicChannels) {
                    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if constexpr (KernelSize == 0U) {
                        uint32_t kernelIndex = 0U;
                        for (; kernelIndex + 3U < kernelSize; kernelIndex += 4U) {
#pragma unroll
                            for (uint32_t offset = 0U; offset < 4U; ++offset) {
                                const float4 pixel = shared4[
                                    localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex + offset];
                                const float weight = weights[kernelIndex + offset];
                                sum.x += pixel.x * weight;
                                sum.y += pixel.y * weight;
                                sum.z += pixel.z * weight;
                                sum.w += pixel.w * weight;
                            }
                        }
                        for (; kernelIndex < kernelSize; ++kernelIndex) {
                            const float4 pixel = shared4[
                                localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex];
                            const float weight = weights[kernelIndex];
                            sum.x += pixel.x * weight;
                            sum.y += pixel.y * weight;
                            sum.z += pixel.z * weight;
                            sum.w += pixel.w * weight;
                        }
                    } else {
#pragma unroll
                        for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
                            const float4 pixel = shared4[
                                localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex];
                            const float weight = weights[kernelIndex];
                            sum.x += pixel.x * weight;
                            sum.y += pixel.y * weight;
                            sum.z += pixel.z * weight;
                            sum.w += pixel.w * weight;
                        }
                    }
                    dst4[PixelOffset(outputY, outputX, width)] = sum;
                } else if constexpr (ChannelStride == 3U && !DynamicChannels) {
                    PackedC3 sum{0.0f, 0.0f, 0.0f};
                    if constexpr (KernelSize == 0U) {
                        uint32_t kernelIndex = 0U;
                        for (; kernelIndex + 3U < kernelSize; kernelIndex += 4U) {
#pragma unroll
                            for (uint32_t offset = 0U; offset < 4U; ++offset) {
                                const uint32_t sharedOffset =
                                    localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex + offset;
                                const float weight = weights[kernelIndex + offset];
                                sum.x += shared3[sharedOffset].x * weight;
                                sum.y += shared3[sharedOffset].y * weight;
                                sum.z += shared3[sharedOffset].z * weight;
                            }
                        }
                        for (; kernelIndex < kernelSize; ++kernelIndex) {
                            const uint32_t sharedOffset =
                                localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex;
                            const float weight = weights[kernelIndex];
                            sum.x += shared3[sharedOffset].x * weight;
                            sum.y += shared3[sharedOffset].y * weight;
                            sum.z += shared3[sharedOffset].z * weight;
                        }
                    } else {
#pragma unroll
                        for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
                            const uint32_t sharedOffset =
                                localRow * ROW_SHARED_W + sharedCenter - anchor + kernelIndex;
                            const float weight = weights[kernelIndex];
                            sum.x += shared3[sharedOffset].x * weight;
                            sum.y += shared3[sharedOffset].y * weight;
                            sum.z += shared3[sharedOffset].z * weight;
                        }
                    }
                    const uint64_t outputOffset = PixelOffset(outputY, outputX, width);
                    dst3[outputOffset].x = sum.x;
                    dst3[outputOffset].y = sum.y;
                    dst3[outputOffset].z = sum.z;
                } else if constexpr (c8Subgroups) {
                    ComputeGenericRowC8Subgroup<KernelSize>(
                        outputChannels, kernelSize, localRow, sharedCenter, anchor,
                        ElementOffset(outputY, outputX, width, channels, channelOffset), weights, shared, dst);
                } else {
                    const uint64_t outputBase =
                        ChannelStride == GAUSSIAN_BLUR_CHANNEL_TILE && DynamicChannels ?
                            ChunkMajorOffset(outputY, outputX, height, width, channelOffset, outputChannels) :
                            ElementOffset(outputY, outputX, width, channels, channelOffset);
                    ComputeGenericRowGroup<ChannelStride, DynamicChannels, KernelSize>(
                        outputChannels, kernelSize, localRow, sharedCenter, anchor,
                        outputBase, weights, shared, dst);
                }
            }
        }
        asc_syncthreads();
        }
    }
}

template <uint32_t ChannelStride, bool DynamicChannels, uint32_t KernelSize>
__simt_vf__ __aicore__ __launch_bounds__(512) inline void RowInteriorUbTile(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t kernelSize,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    const uint32_t anchor = KernelSize == 0U ? kernelSize / 2U : (KernelSize - 1U) / 2U;
    auto* shared4 = reinterpret_cast<__ubuf__ const float4*>(shared);
    auto* shared3 = reinterpret_cast<__ubuf__ const PackedC3*>(shared);
    auto* dst4 = reinterpret_cast<__gm__ float4*>(dst);
    auto* dst3 = reinterpret_cast<__gm__ PackedC3*>(dst);
    const uint32_t outputY = tileBaseY + threadIdx.y;

#pragma unroll
    for (uint32_t patch = 0U; patch < GAUSSIAN_BLUR_ROW_PATCHES; ++patch) {
        const uint32_t outputX = tileBaseX + threadIdx.x + patch * GAUSSIAN_BLUR_ROW_BLOCK_X;
        if (outputX >= width) {
            continue;
        }
        const uint32_t sharedCenter = threadIdx.x + GAUSSIAN_BLUR_ROW_BLOCK_X +
            patch * GAUSSIAN_BLUR_ROW_BLOCK_X;
        if constexpr (ChannelStride == 4U && !DynamicChannels) {
            const uint32_t rowBase = threadIdx.y * ROW_SHARED_W;
            float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if constexpr (KernelSize == 31U) {
                const float4 center = shared4[rowBase + sharedCenter];
                sum = make_float4(
                    center.x * weights[anchor], center.y * weights[anchor],
                    center.z * weights[anchor], center.w * weights[anchor]);
                for (uint32_t offset = 1U; offset <= anchor; ++offset) {
                    const float4 left = shared4[rowBase + sharedCenter - offset];
                    const float4 right = shared4[rowBase + sharedCenter + offset];
                    const float weight = weights[anchor - offset];
                    sum.x += (left.x + right.x) * weight;
                    sum.y += (left.y + right.y) * weight;
                    sum.z += (left.z + right.z) * weight;
                    sum.w += (left.w + right.w) * weight;
                }
            } else if constexpr (KernelSize == 0U) {
                for (uint32_t k = 0U; k < kernelSize; ++k) {
                    const float4 pixel = shared4[rowBase + sharedCenter - anchor + k];
                    const float weight = weights[k];
                    sum.x += pixel.x * weight;
                    sum.y += pixel.y * weight;
                    sum.z += pixel.z * weight;
                    sum.w += pixel.w * weight;
                }
            } else {
#pragma unroll
                for (uint32_t k = 0U; k < KernelSize; ++k) {
                    const float4 pixel = shared4[rowBase + sharedCenter - anchor + k];
                    const float weight = weights[k];
                    sum.x += pixel.x * weight;
                    sum.y += pixel.y * weight;
                    sum.z += pixel.z * weight;
                    sum.w += pixel.w * weight;
                }
            }
            dst4[PixelOffset(outputY, outputX, width)] = sum;
        } else if constexpr (ChannelStride == 3U && !DynamicChannels) {
            const uint32_t rowBase = threadIdx.y * ROW_SHARED_W;
            PackedC3 sum{0.0f, 0.0f, 0.0f};
            if constexpr (KernelSize == 31U) {
                const uint32_t centerOffset = rowBase + sharedCenter;
                sum.x = shared3[centerOffset].x * weights[anchor];
                sum.y = shared3[centerOffset].y * weights[anchor];
                sum.z = shared3[centerOffset].z * weights[anchor];
                for (uint32_t offset = 1U; offset <= anchor; ++offset) {
                    const uint32_t leftOffset = centerOffset - offset;
                    const uint32_t rightOffset = centerOffset + offset;
                    const float weight = weights[anchor - offset];
                    sum.x += (shared3[leftOffset].x + shared3[rightOffset].x) * weight;
                    sum.y += (shared3[leftOffset].y + shared3[rightOffset].y) * weight;
                    sum.z += (shared3[leftOffset].z + shared3[rightOffset].z) * weight;
                }
            } else if constexpr (KernelSize == 0U) {
                for (uint32_t k = 0U; k < kernelSize; ++k) {
                    const uint32_t offset = rowBase + sharedCenter - anchor + k;
                    const float weight = weights[k];
                    sum.x += shared3[offset].x * weight;
                    sum.y += shared3[offset].y * weight;
                    sum.z += shared3[offset].z * weight;
                }
            } else {
#pragma unroll
                for (uint32_t k = 0U; k < KernelSize; ++k) {
                    const uint32_t offset = rowBase + sharedCenter - anchor + k;
                    const float weight = weights[k];
                    sum.x += shared3[offset].x * weight;
                    sum.y += shared3[offset].y * weight;
                    sum.z += shared3[offset].z * weight;
                }
            }
            const uint64_t outputOffset = PixelOffset(outputY, outputX, width);
            dst3[outputOffset].x = sum.x;
            dst3[outputOffset].y = sum.y;
            dst3[outputOffset].z = sum.z;
        } else if constexpr (ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP && DynamicChannels) {
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
            if constexpr (KernelSize == 31U) {
                ComputeGenericRowC8ChunkMajorSubgroup<KernelSize>(
                    height, width, channelOffset, outputChannels, outputY, outputX,
                    kernelSize, threadIdx.y, sharedCenter, anchor, weights, shared, dst);
            } else
#endif
            if (channels > GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP) {
                ComputeGenericRowC8ChunkMajorSubgroup<KernelSize>(
                    height, width, channelOffset, outputChannels, outputY, outputX,
                    kernelSize, threadIdx.y, sharedCenter, anchor, weights, shared, dst);
            } else {
                ComputeGenericRowC8Subgroup<KernelSize>(
                    outputChannels, kernelSize, threadIdx.y, sharedCenter, anchor,
                    ElementOffset(outputY, outputX, width, channels, channelOffset), weights, shared, dst);
            }
        } else {
            const uint64_t outputBase =
                ChannelStride == GAUSSIAN_BLUR_CHANNEL_TILE && DynamicChannels ?
                    ChunkMajorOffset(outputY, outputX, height, width, channelOffset, outputChannels) :
                    ElementOffset(outputY, outputX, width, channels, channelOffset);
            ComputeGenericRowGroup<ChannelStride, DynamicChannels, KernelSize>(
                outputChannels, kernelSize, threadIdx.y, sharedCenter, anchor,
                outputBase, weights, shared, dst);
        }
    }
}

template <uint32_t ChannelStride, bool DynamicChannels, uint32_t KernelSize,
          bool SymmetricKernel = false, bool EdgeOnly = false, bool InteriorOnly = false,
          bool TailOnly = false>
__simt_callee__ inline void RunColumnTiles(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __gm__ const float* src,
    __gm__ float* dst,
    __ubuf__ float* shared)
{
    const uint32_t anchor = KernelSize == 0U ? kernelSize / 2U : (KernelSize - 1U) / 2U;
    const uint32_t spatialTiles = tilesX * tilesY;
    const uint32_t interiorTilesX = width / GAUSSIAN_BLUR_COLUMN_TILE_W;
    const uint32_t interiorTilesY = height >
            COLUMN_SHARED_H - GAUSSIAN_BLUR_COLUMN_BLOCK_Y ?
        (height - (COLUMN_SHARED_H - GAUSSIAN_BLUR_COLUMN_BLOCK_Y)) /
            COLUMN_TILE_H : 0U;
    const uint32_t interiorSpatialTiles = interiorTilesX * interiorTilesY;
    const uint32_t fullChannelTiles = DynamicChannels &&
            ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP ?
        1U : channels / ChannelStride;
    const uint32_t channelTiles = DynamicChannels ?
        (channels + ChannelStride - 1U) / ChannelStride : 1U;
    const uint32_t tailChannelTiles = channelTiles - fullChannelTiles;
    const uint32_t topEdgeTiles = tilesX;
    const uint32_t middleEdgeColumns = tilesX - interiorTilesX;
    const uint32_t middleEdgeTiles = interiorTilesY * middleEdgeColumns;
    const uint32_t bottomStartY = interiorTilesY + 1U;
    const uint32_t bottomEdgeRows = tilesY > bottomStartY ? tilesY - bottomStartY : 0U;
    const uint32_t bottomEdgeTiles = bottomEdgeRows * tilesX;
    const uint32_t edgeSpatialTiles = topEdgeTiles + middleEdgeTiles + bottomEdgeTiles;
    const uint32_t activeTotalTiles = TailOnly ? spatialTiles * tailChannelTiles : (InteriorOnly ?
        interiorSpatialTiles * fullChannelTiles :
        (EdgeOnly && DynamicChannels ?
            edgeSpatialTiles * fullChannelTiles + spatialTiles * tailChannelTiles : totalTiles));
    auto* src4 = reinterpret_cast<__gm__ const float4*>(src);
    auto* dst4 = reinterpret_cast<__gm__ float4*>(dst);
    auto* shared4 = reinterpret_cast<__ubuf__ float4*>(shared);
    auto* src3 = reinterpret_cast<__gm__ const PackedC3*>(src);
    auto* dst3 = reinterpret_cast<__gm__ PackedC3*>(dst);
    auto* shared3 = reinterpret_cast<__ubuf__ PackedC3*>(shared);

    for (uint32_t tileId = coreIndex; tileId < activeTotalTiles; tileId += coreCount) {
        uint32_t spatialTileId = 0U;
        uint32_t channelOffset = 0U;
        uint32_t outputChannels = 0U;
        uint32_t tileX = 0U;
        uint32_t tileY = 0U;
        if constexpr (TailOnly) {
            spatialTileId = tileId % spatialTiles;
            channelOffset = (fullChannelTiles + tileId / spatialTiles) * ChannelStride;
            outputChannels = channels - channelOffset < ChannelStride ?
                channels - channelOffset : ChannelStride;
            tileX = spatialTileId % tilesX;
            tileY = spatialTileId / tilesX;
        } else if constexpr (InteriorOnly) {
            spatialTileId = tileId % interiorSpatialTiles;
            channelOffset = (tileId / interiorSpatialTiles) * ChannelStride;
            outputChannels = DynamicChannels ?
                (channelOffset + ChannelStride <= channels ? ChannelStride : channels - channelOffset) :
                ChannelStride;
            tileX = spatialTileId % interiorTilesX;
            tileY = spatialTileId / interiorTilesX + 1U;
        } else if constexpr (EdgeOnly && DynamicChannels) {
            const uint32_t fullChannelEdgeTasks = edgeSpatialTiles * fullChannelTiles;
            if (tileId < fullChannelEdgeTasks) {
                const uint32_t edgeSpatialId = tileId % edgeSpatialTiles;
                channelOffset = (tileId / edgeSpatialTiles) * ChannelStride;
                outputChannels = channelOffset + ChannelStride <= channels ?
                    ChannelStride : channels - channelOffset;
                if (edgeSpatialId < topEdgeTiles) {
                    tileX = edgeSpatialId;
                    tileY = 0U;
                } else if (edgeSpatialId < topEdgeTiles + middleEdgeTiles) {
                    const uint32_t middleId = edgeSpatialId - topEdgeTiles;
                    tileX = interiorTilesX + middleId % middleEdgeColumns;
                    tileY = middleId / middleEdgeColumns + 1U;
                } else {
                    const uint32_t bottomId = edgeSpatialId - topEdgeTiles - middleEdgeTiles;
                    tileX = bottomId % tilesX;
                    tileY = bottomStartY + bottomId / tilesX;
                }
            } else {
                const uint32_t tailId = tileId - fullChannelEdgeTasks;
                spatialTileId = tailId % spatialTiles;
                channelOffset = (fullChannelTiles + tailId / spatialTiles) * ChannelStride;
                outputChannels = channels - channelOffset;
                tileX = spatialTileId % tilesX;
                tileY = spatialTileId / tilesX;
            }
        } else {
            ResolveChannelTile<ChannelStride, DynamicChannels>(
                tileId, spatialTiles, channels, spatialTileId, channelOffset, outputChannels);
            tileX = spatialTileId % tilesX;
            tileY = spatialTileId / tilesX;
        }
        const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_COLUMN_TILE_W;
        const uint32_t outputX = tileBaseX + threadIdx.x;
        const uint32_t tileBaseY = tileY * COLUMN_TILE_H;
        const int32_t yStart = static_cast<int32_t>(tileBaseY + threadIdx.y);
        const bool interiorY = InteriorOnly ||
            (tileBaseY >= GAUSSIAN_BLUR_COLUMN_BLOCK_Y &&
             tileBaseY + COLUMN_SHARED_H - GAUSSIAN_BLUR_COLUMN_BLOCK_Y <= height);
        if (interiorY) {
            for (uint32_t segment = 0U; segment < GAUSSIAN_BLUR_COLUMN_PATCHES + 2U; ++segment) {
                const uint32_t sourceY = tileBaseY - GAUSSIAN_BLUR_COLUMN_BLOCK_Y + threadIdx.y +
                    segment * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
                const uint32_t sharedY = threadIdx.y + segment * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
                if constexpr (ChannelStride == 4U && !DynamicChannels) {
                    const uint32_t sharedOffset =
                        sharedY * GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x;
                    shared4[sharedOffset] = outputX < width ?
                        src4[PixelOffset(sourceY, outputX, width)] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                } else if constexpr (ChannelStride == 3U && !DynamicChannels) {
                    const uint32_t sharedOffset =
                        sharedY * GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x;
                    if (outputX < width) {
                        const uint64_t sourceOffset = PixelOffset(sourceY, outputX, width);
                        shared3[sharedOffset].x = src3[sourceOffset].x;
                        shared3[sharedOffset].y = src3[sourceOffset].y;
                        shared3[sharedOffset].z = src3[sourceOffset].z;
                    } else {
                        shared3[sharedOffset].x = 0.0f;
                        shared3[sharedOffset].y = 0.0f;
                        shared3[sharedOffset].z = 0.0f;
                    }
                } else {
                    const bool sourceValid = InteriorOnly || outputX < width;
                    const uint64_t sourceBase = sourceValid ?
                        (ChannelStride == GAUSSIAN_BLUR_CHANNEL_TILE && DynamicChannels ?
                            ChunkMajorOffset(sourceY, outputX, height, width,
                                channelOffset, outputChannels) :
                            ElementOffset(sourceY, outputX, width, channels, channelOffset)) : 0U;
                    LoadGenericColumnGroup<ChannelStride, DynamicChannels>(
                        outputChannels, sharedY, threadIdx.x, sourceBase, sourceValid, src, shared);
                }
            }
        } else {
            LoadColumnTile<ChannelStride, DynamicChannels, true>(
                height, width, channels, outputX, channelOffset, outputChannels, borderType, tileBaseY, src, shared);
        }
        asc_syncthreads();

        for (uint32_t patch = 0U; patch < GAUSSIAN_BLUR_COLUMN_PATCHES; ++patch) {
            const uint32_t outputY = static_cast<uint32_t>(yStart) +
                patch * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
            if (InteriorOnly || (outputX < width && outputY < height)) {
                const uint32_t sharedCenter = threadIdx.y + GAUSSIAN_BLUR_COLUMN_BLOCK_Y +
                    patch * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
                if constexpr (ChannelStride == 4U && !DynamicChannels) {
                    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if constexpr (KernelSize == 0U) {
                        uint32_t kernelIndex = 0U;
                        for (; kernelIndex + 3U < kernelSize; kernelIndex += 4U) {
#pragma unroll
                            for (uint32_t offset = 0U; offset < 4U; ++offset) {
                                const float4 pixel = shared4[
                                    (sharedCenter - anchor + kernelIndex + offset) *
                                        GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x];
                                const float weight = weights[kernelIndex + offset];
                                sum.x += pixel.x * weight;
                                sum.y += pixel.y * weight;
                                sum.z += pixel.z * weight;
                                sum.w += pixel.w * weight;
                            }
                        }
                        for (; kernelIndex < kernelSize; ++kernelIndex) {
                            const float4 pixel = shared4[
                                (sharedCenter - anchor + kernelIndex) *
                                    GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x];
                            const float weight = weights[kernelIndex];
                            sum.x += pixel.x * weight;
                            sum.y += pixel.y * weight;
                            sum.z += pixel.z * weight;
                            sum.w += pixel.w * weight;
                        }
                    } else {
#pragma unroll
                        for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
                            const float4 pixel = shared4[
                                (sharedCenter - anchor + kernelIndex) *
                                    GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x];
                            const float weight = weights[kernelIndex];
                            sum.x += pixel.x * weight;
                            sum.y += pixel.y * weight;
                            sum.z += pixel.z * weight;
                            sum.w += pixel.w * weight;
                        }
                    }
                    dst4[PixelOffset(outputY, outputX, width)] = sum;
                } else if constexpr (ChannelStride == 3U && !DynamicChannels) {
                    PackedC3 sum{0.0f, 0.0f, 0.0f};
                    if constexpr (KernelSize == 0U) {
                        uint32_t kernelIndex = 0U;
                        for (; kernelIndex + 3U < kernelSize; kernelIndex += 4U) {
#pragma unroll
                            for (uint32_t offset = 0U; offset < 4U; ++offset) {
                                const uint32_t sharedOffset =
                                    (sharedCenter - anchor + kernelIndex + offset) *
                                        GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x;
                                const float weight = weights[kernelIndex + offset];
                                sum.x += shared3[sharedOffset].x * weight;
                                sum.y += shared3[sharedOffset].y * weight;
                                sum.z += shared3[sharedOffset].z * weight;
                            }
                        }
                        for (; kernelIndex < kernelSize; ++kernelIndex) {
                            const uint32_t sharedOffset =
                                (sharedCenter - anchor + kernelIndex) *
                                    GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x;
                            const float weight = weights[kernelIndex];
                            sum.x += shared3[sharedOffset].x * weight;
                            sum.y += shared3[sharedOffset].y * weight;
                            sum.z += shared3[sharedOffset].z * weight;
                        }
                    } else {
#pragma unroll
                        for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
                            const uint32_t sharedOffset =
                                (sharedCenter - anchor + kernelIndex) *
                                    GAUSSIAN_BLUR_COLUMN_BLOCK_X + threadIdx.x;
                            const float weight = weights[kernelIndex];
                            sum.x += shared3[sharedOffset].x * weight;
                            sum.y += shared3[sharedOffset].y * weight;
                            sum.z += shared3[sharedOffset].z * weight;
                        }
                    }
                    const uint64_t outputOffset = PixelOffset(outputY, outputX, width);
                    dst3[outputOffset].x = sum.x;
                    dst3[outputOffset].y = sum.y;
                    dst3[outputOffset].z = sum.z;
                } else {
                    const uint64_t outputBase =
                        ElementOffset(outputY, outputX, width, channels, channelOffset);
                    if constexpr (SymmetricKernel) {
                        if (outputChannels == 4U) {
                            if constexpr (KernelSize == 31U) {
                                ComputeGenericColumnGroupK31Symmetric(
                                    threadIdx.x, sharedCenter, outputBase, weights, shared, dst);
                            } else {
                                ComputeGenericColumnGroupSymmetric<KernelSize>(
                                    threadIdx.x, sharedCenter, outputBase, weights, shared, dst);
                            }
                        } else {
                            ComputeGenericColumnGroup<ChannelStride, DynamicChannels, KernelSize>(
                                outputChannels, kernelSize, threadIdx.x, sharedCenter, anchor,
                                outputBase, weights, shared, dst);
                        }
                    } else {
                        ComputeGenericColumnGroup<ChannelStride, DynamicChannels, KernelSize>(
                            outputChannels, kernelSize, threadIdx.x, sharedCenter, anchor,
                            outputBase, weights, shared, dst);
                    }
                }
            }
        }
        asc_syncthreads();
    }
}

template <uint32_t KernelSize>
__simt_vf__ __aicore__ __launch_bounds__(64) inline void ColumnInteriorXMajorSlidingTile(
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    const uint32_t localX = threadIdx.x;
    const uint32_t outputX = tileBaseX + localX;
    constexpr uint32_t anchor = KernelSize / 2U;
    auto* shared4 = reinterpret_cast<__ubuf__ const float4*>(shared);
    if (threadIdx.y >= GAUSSIAN_BLUR_COLUMN_BLOCK_Y / 4U) {
        return;
    }
    const uint32_t outputLane = threadIdx.y * 4U;
    for (uint32_t patch = 0U; patch < GAUSSIAN_BLUR_COLUMN_PATCHES; ++patch) {
        const uint32_t outputY0 = tileBaseY + outputLane + patch * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
        const uint32_t outputY1 = outputY0 + 1U;
        const uint32_t outputY2 = outputY0 + 2U;
        const uint32_t outputY3 = outputY0 + 3U;
        const uint32_t sharedCenter0 = GAUSSIAN_BLUR_COLUMN_BLOCK_Y + outputLane +
            patch * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
        const uint32_t windowBase = localX * COLUMN_SHARED_H + sharedCenter0 - anchor;
        float4 sum0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 sum1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 sum2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 sum3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 pixel0 = shared4[windowBase];
        float4 pixel1 = shared4[windowBase + 1U];
        float4 pixel2 = shared4[windowBase + 2U];
#pragma unroll
        for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
            const float4 pixel3 = shared4[windowBase + kernelIndex + 3U];
            const float weight = weights[kernelIndex];
            sum0.x += pixel0.x * weight;
            sum0.y += pixel0.y * weight;
            sum0.z += pixel0.z * weight;
            sum0.w += pixel0.w * weight;
            sum1.x += pixel1.x * weight;
            sum1.y += pixel1.y * weight;
            sum1.z += pixel1.z * weight;
            sum1.w += pixel1.w * weight;
            sum2.x += pixel2.x * weight;
            sum2.y += pixel2.y * weight;
            sum2.z += pixel2.z * weight;
            sum2.w += pixel2.w * weight;
            sum3.x += pixel3.x * weight;
            sum3.y += pixel3.y * weight;
            sum3.z += pixel3.z * weight;
            sum3.w += pixel3.w * weight;
            pixel0 = pixel1;
            pixel1 = pixel2;
            pixel2 = pixel3;
        }
        const uint64_t outputBase0 = ElementOffset(outputY0, outputX, width, channels, channelOffset);
        const uint64_t outputBase1 = ElementOffset(outputY1, outputX, width, channels, channelOffset);
        const uint64_t outputBase2 = ElementOffset(outputY2, outputX, width, channels, channelOffset);
        const uint64_t outputBase3 = ElementOffset(outputY3, outputX, width, channels, channelOffset);
        dst[outputBase0] = sum0.x;
        dst[outputBase0 + 1U] = sum0.y;
        dst[outputBase0 + 2U] = sum0.z;
        dst[outputBase0 + 3U] = sum0.w;
        dst[outputBase1] = sum1.x;
        dst[outputBase1 + 1U] = sum1.y;
        dst[outputBase1 + 2U] = sum1.z;
        dst[outputBase1 + 3U] = sum1.w;
        dst[outputBase2] = sum2.x;
        dst[outputBase2 + 1U] = sum2.y;
        dst[outputBase2 + 2U] = sum2.z;
        dst[outputBase2 + 3U] = sum2.w;
        dst[outputBase3] = sum3.x;
        dst[outputBase3 + 1U] = sum3.y;
        dst[outputBase3 + 2U] = sum3.z;
        dst[outputBase3 + 3U] = sum3.w;
    }
}

template <uint32_t KernelSize>
__simt_vf__ __aicore__ __launch_bounds__(512) inline void ColumnInteriorXMajorC8Tile(
    uint32_t width,
    uint32_t channels,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
    const uint32_t localX = threadIdx.x;
    const uint32_t outputX = tileBaseX + localX;
    constexpr uint32_t anchor = KernelSize / 2U;
    auto* shared4 = reinterpret_cast<__ubuf__ const float4*>(shared);
    const uint32_t subgroupOffset = threadIdx.z * 4U;
    if (subgroupOffset >= channels) {
        return;
    }
    for (uint32_t patch = 0U; patch < GAUSSIAN_BLUR_COLUMN_PATCHES; ++patch) {
        const uint32_t outputY = tileBaseY + threadIdx.y + patch * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
        const uint32_t sharedCenter = GAUSSIAN_BLUR_COLUMN_BLOCK_Y + threadIdx.y +
            patch * GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (uint32_t kernelIndex = 0U; kernelIndex < KernelSize; ++kernelIndex) {
            const uint32_t sharedBase =
                (localX * COLUMN_SHARED_H + sharedCenter - anchor + kernelIndex) *
                2U + threadIdx.z;
            const float4 pixel = shared4[sharedBase];
            const float weight = weights[kernelIndex];
            sum.x += pixel.x * weight;
            sum.y += pixel.y * weight;
            sum.z += pixel.z * weight;
            sum.w += pixel.w * weight;
        }
        const uint64_t outputBase = ElementOffset(outputY, outputX, width, channels, 0U);
        const uint32_t active = channels - subgroupOffset;
        dst[outputBase + subgroupOffset] = sum.x;
        if (active >= 2U) dst[outputBase + subgroupOffset + 1U] = sum.y;
        if (active >= 3U) dst[outputBase + subgroupOffset + 2U] = sum.z;
        if (active >= 4U) dst[outputBase + subgroupOffset + 3U] = sum.w;
    }
}

__aicore__ inline void DispatchColumnInteriorXMajorC8Tile(
    uint32_t width,
    uint32_t channels,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t kernelSize,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst)
{
#define LAUNCH_COLUMN_C8(K) \
    asc_vf_call<ColumnInteriorXMajorC8Tile<K>>( \
        dim3{GAUSSIAN_BLUR_COLUMN_BLOCK_X, GAUSSIAN_BLUR_COLUMN_BLOCK_Y, 2U}, \
        width, channels, tileBaseX, tileBaseY, weights, shared, dst)
    if (kernelSize == 1U) {
        LAUNCH_COLUMN_C8(1U);
    } else if (kernelSize == 3U) {
        LAUNCH_COLUMN_C8(3U);
    } else if (kernelSize == 5U) {
        LAUNCH_COLUMN_C8(5U);
    } else if (kernelSize == 7U) {
        LAUNCH_COLUMN_C8(7U);
    } else if (kernelSize == 9U) {
        LAUNCH_COLUMN_C8(9U);
    } else if (kernelSize == 11U) {
        LAUNCH_COLUMN_C8(11U);
    } else if (kernelSize == 15U) {
        LAUNCH_COLUMN_C8(15U);
    } else if (kernelSize == 21U) {
        LAUNCH_COLUMN_C8(21U);
    } else {
        LAUNCH_COLUMN_C8(31U);
    }
#undef LAUNCH_COLUMN_C8
}

template <uint32_t ChannelStride, bool DynamicChannels>
__simt_callee__ inline void DispatchRowKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __gm__ const float* src,
    __gm__ float* dst,
    __ubuf__ float* shared)
{
#define ARGS height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY, kernelSize, borderType, \
    weights, src, dst, shared
    if (kernelSize == 1U) {
        RunRowTiles<ChannelStride, DynamicChannels, 1U, false>(ARGS);
    } else if (kernelSize == 3U) {
        RunRowTiles<ChannelStride, DynamicChannels, 3U, false>(ARGS);
    } else if (kernelSize == 5U) {
        RunRowTiles<ChannelStride, DynamicChannels, 5U, false>(ARGS);
    } else if (kernelSize == 7U) {
        RunRowTiles<ChannelStride, DynamicChannels, 7U, false>(ARGS);
    } else if (kernelSize == 9U) {
        RunRowTiles<ChannelStride, DynamicChannels, 9U, false>(ARGS);
    } else if (kernelSize == 11U) {
        RunRowTiles<ChannelStride, DynamicChannels, 11U, false>(ARGS);
    } else if (kernelSize == 15U) {
        RunRowTiles<ChannelStride, DynamicChannels, 15U, false>(ARGS);
    } else if (kernelSize == 21U) {
        RunRowTiles<ChannelStride, DynamicChannels, 21U, false>(ARGS);
    } else if (kernelSize == 31U) {
        RunRowTiles<ChannelStride, DynamicChannels, 31U, false>(ARGS);
    } else {
        RunRowTiles<ChannelStride, DynamicChannels, 0U, false>(ARGS);
    }
#undef ARGS
}

template <uint32_t ChannelStride, bool DynamicChannels,
          bool EdgeOnly = false, bool InteriorOnly = false>
__simt_callee__ inline void DispatchColumnKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __gm__ const float* src,
    __gm__ float* dst,
    __ubuf__ float* shared)
{
#define COLUMN_ARGS height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY, kernelSize, \
    borderType, weights, src, dst, shared
#if GAUSSIAN_BLUR_ENABLE_COLUMN_HOT_K_SPECIALIZATION
    if (kernelSize == 3U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 3U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 5U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 5U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 11U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 11U, DynamicChannels, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 21U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 21U, DynamicChannels, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
    } else if (kernelSize == 31U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 31U, DynamicChannels, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
#endif
    } else {
        RunColumnTiles<ChannelStride, DynamicChannels, 0U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    }
#else
    if (kernelSize == 1U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 1U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 3U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 3U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 5U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 5U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 7U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 7U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 9U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 9U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 11U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 11U, DynamicChannels, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 15U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 15U, DynamicChannels, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 21U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 21U, DynamicChannels, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else if (kernelSize == 31U) {
        RunColumnTiles<ChannelStride, DynamicChannels, 31U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    } else {
        RunColumnTiles<ChannelStride, DynamicChannels, 0U, false, EdgeOnly, InteriorOnly>(COLUMN_ARGS);
    }
#endif
#undef COLUMN_ARGS
}

#ifndef GAUSSIAN_BLUR_COLUMN_ONLY
template <bool RowPass>
__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurPassKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    uint32_t pathMode,
    __ubuf__ const float* weights,
    __gm__ const float* src,
    __gm__ float* dst)
{
    __ubuf__ float shared[ROW_SHARED_ELEMENTS];
    if (pathMode == GAUSSIAN_BLUR_PATH_C1_FAST) {
        DispatchRowKernel<1U, false>(height, width, channels, totalTiles, coreIndex, coreCount,
                                          tilesX, tilesY, kernelSize,
                                          borderType, weights, src, dst, shared);
    } else if (pathMode == GAUSSIAN_BLUR_PATH_C3_FAST) {
        DispatchRowKernel<3U, false>(height, width, channels, totalTiles, coreIndex, coreCount,
                                          tilesX, tilesY, kernelSize,
                                          borderType, weights, src, dst, shared);
    } else if (pathMode == GAUSSIAN_BLUR_PATH_C4_FAST) {
        DispatchRowKernel<4U, false>(height, width, channels, totalTiles, coreIndex, coreCount,
                                          tilesX, tilesY, kernelSize,
                                          borderType, weights, src, dst, shared);
    } else {
        DispatchRowKernel<GAUSSIAN_BLUR_CHANNEL_TILE, true>(
            height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY, kernelSize, borderType,
            weights, src, dst, shared);
    }
}
#endif

#if !defined(GAUSSIAN_BLUR_COLUMN_ONLY)
__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurRowC8Kernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __gm__ const float* src,
    __gm__ float* dst)
{
    __ubuf__ float shared[
        GAUSSIAN_BLUR_ROW_TILE_H * ROW_SHARED_W * GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP];
    DispatchRowKernel<GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP, true>(
        height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY,
        kernelSize, borderType, weights, src, dst, shared);
}
#endif

#ifndef GAUSSIAN_BLUR_ROW_ONLY
__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurColumnKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    uint32_t pathMode,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ const float* src,
    __gm__ float* dst)
{
    if (pathMode == GAUSSIAN_BLUR_PATH_C1_FAST) {
        DispatchColumnKernel<1U, false>(height, width, channels, totalTiles, coreIndex, coreCount,
            tilesX, tilesY, kernelSize, borderType, weights, src, dst, shared);
    } else if (pathMode == GAUSSIAN_BLUR_PATH_C3_FAST) {
        DispatchColumnKernel<3U, false>(height, width, channels, totalTiles, coreIndex, coreCount,
            tilesX, tilesY, kernelSize, borderType, weights, src, dst, shared);
    } else if (pathMode == GAUSSIAN_BLUR_PATH_C4_FAST) {
        DispatchColumnKernel<4U, false>(height, width, channels, totalTiles, coreIndex, coreCount,
            tilesX, tilesY, kernelSize, borderType, weights, src, dst, shared);
    } else {
        DispatchColumnKernel<GAUSSIAN_BLUR_CHANNEL_TILE, true, false, false>(
            height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY,
            kernelSize, borderType, weights, src, dst, shared);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurColumnC8Kernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ const float* src,
    __gm__ float* dst)
{
    DispatchColumnKernel<GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP, true>(
        height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY,
        kernelSize, borderType, weights, src, dst, shared);
}

__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurColumnC8EdgeKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ const float* src,
    __gm__ float* dst)
{
    DispatchColumnKernel<GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP, true, true, false>(
        height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY,
        kernelSize, borderType, weights, src, dst, shared);
}

__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurColumnInteriorKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ const float* src,
    __gm__ float* dst)
{
    DispatchColumnKernel<GAUSSIAN_BLUR_CHANNEL_TILE, true, false, true>(
        height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY,
        kernelSize, borderType, weights, src, dst, shared);
}

__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurColumnEdgeKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ const float* src,
    __gm__ float* dst)
{
    DispatchColumnKernel<GAUSSIAN_BLUR_CHANNEL_TILE, true, true, false>(
        height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY,
        kernelSize, borderType, weights, src, dst, shared);
}

__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurColumnGenericK31Kernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ const float* src,
    __gm__ float* dst)
{
    RunColumnTiles<GAUSSIAN_BLUR_CHANNEL_TILE, true, 31U, true, false, false>(
        height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY,
        31U, borderType, weights, src, dst, shared);
}

__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS)
inline void GaussianBlurColumnGenericK31InteriorKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ const float* src,
    __gm__ float* dst)
{
    RunColumnTiles<GAUSSIAN_BLUR_CHANNEL_TILE, true, 31U, true, false, true>(
        height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY,
        31U, borderType, weights, src, dst, shared);
}

#if !GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS)
inline void GaussianBlurColumnGenericK31TailKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ const float* src,
    __gm__ float* dst)
{
    RunColumnTiles<GAUSSIAN_BLUR_CHANNEL_TILE, true, 31U, true, false, false, true>(
        height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY,
        31U, borderType, weights, src, dst, shared);
}
#endif
#endif

template <uint32_t ChannelStride, bool DynamicChannels, uint32_t KernelSize>
__aicore__ inline void LaunchRowInteriorUbTile(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t kernelSize,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst,
    uint32_t threadRows = GAUSSIAN_BLUR_ROW_BLOCK_Y)
{
    asc_vf_call<RowInteriorUbTile<ChannelStride, DynamicChannels, KernelSize>>(
        dim3{GAUSSIAN_BLUR_ROW_BLOCK_X, threadRows,
             ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP && DynamicChannels ?
                 ChannelStride / GAUSSIAN_BLUR_CHANNEL_TILE : 1U},
        height, width, channels, channelOffset, outputChannels, tileBaseX, tileBaseY, kernelSize, weights, shared, dst);
}

template <uint32_t ChannelStride, bool DynamicChannels>
__aicore__ inline void DispatchRowInteriorUbTile(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t kernelSize,
    __ubuf__ const float* weights,
    __ubuf__ const float* shared,
    __gm__ float* dst,
    uint32_t threadRows = GAUSSIAN_BLUR_ROW_BLOCK_Y)
{
#define LAUNCH_ROW_UB(K) \
    LaunchRowInteriorUbTile<ChannelStride, DynamicChannels, K>( \
        height, width, channels, channelOffset, outputChannels, tileBaseX, tileBaseY, kernelSize, weights, shared, dst, \
        threadRows)
#if GAUSSIAN_BLUR_ENABLE_ROW_HOT_K_SPECIALIZATION
    if (kernelSize == 3U) {
        LAUNCH_ROW_UB(3U);
    } else if (kernelSize == 5U) {
        LAUNCH_ROW_UB(5U);
    } else if (kernelSize == 11U) {
        LAUNCH_ROW_UB(11U);
    } else if (kernelSize == 21U) {
        LAUNCH_ROW_UB(21U);
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
    } else if (kernelSize == 31U) {
        LAUNCH_ROW_UB(31U);
#endif
    } else {
        LAUNCH_ROW_UB(0U);
    }
#else
    if (kernelSize == 1U) {
        LAUNCH_ROW_UB(1U);
    } else if (kernelSize == 3U) {
        LAUNCH_ROW_UB(3U);
    } else if (kernelSize == 5U) {
        LAUNCH_ROW_UB(5U);
    } else if (kernelSize == 7U) {
        LAUNCH_ROW_UB(7U);
    } else if (kernelSize == 9U) {
        LAUNCH_ROW_UB(9U);
    } else if (kernelSize == 11U) {
        LAUNCH_ROW_UB(11U);
    } else if (kernelSize == 15U) {
        LAUNCH_ROW_UB(15U);
    } else if (kernelSize == 21U) {
        LAUNCH_ROW_UB(21U);
    } else if (kernelSize == 31U) {
        LAUNCH_ROW_UB(0U);
    } else {
        LAUNCH_ROW_UB(0U);
    }
#endif
#undef LAUNCH_ROW_UB
}

#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
static constexpr uint32_t K31_ROW_C4_SIMD_CHANNELS = 4U;
static constexpr uint32_t K31_ROW_C4_SIMD_ROW_ELEMENTS =
    GAUSSIAN_BLUR_ROW_TILE_W * K31_ROW_C4_SIMD_CHANNELS;
static constexpr uint32_t K31_ROW_C4_SIMD_INPUT_ROW_ELEMENTS =
    ROW_SHARED_W * K31_ROW_C4_SIMD_CHANNELS;
static constexpr uint32_t K31_ROW_C4_SIMD_CENTER_OFFSET =
    GAUSSIAN_BLUR_ROW_BLOCK_X * K31_ROW_C4_SIMD_CHANNELS;
static constexpr bool K31_ROW_C4_SIMD_ENABLED = false;

__simd_vf__ inline void GaussianBlurRowC4K31SimdInteriorVF(
    __ubuf__ float* shared,
    __ubuf__ const float* weights)
{
    auto mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    constexpr uint32_t vectorElements = 64U;
    constexpr uint32_t vectorsPerRow = K31_ROW_C4_SIMD_ROW_ELEMENTS / vectorElements;
    AscendC::MicroAPI::UnalignRegForLoad leftUnalign;
    AscendC::MicroAPI::UnalignRegForLoad rightUnalign;
    for (uint32_t row = 0U; row < GAUSSIAN_BLUR_ROW_TILE_H; ++row) {
        __ubuf__ float* rowBase = shared + row * K31_ROW_C4_SIMD_INPUT_ROW_ELEMENTS;
        for (uint32_t vectorIndex = 0U; vectorIndex < vectorsPerRow; ++vectorIndex) {
            const uint32_t outputOffset = vectorIndex * vectorElements;
            const uint32_t centerOffset = K31_ROW_C4_SIMD_CENTER_OFFSET + outputOffset;
            AscendC::MicroAPI::RegTensor<float> sum;
            AscendC::MicroAPI::RegTensor<float> pair;
            AscendC::MicroAPI::RegTensor<float> left;
            AscendC::MicroAPI::RegTensor<float> right;
            AscendC::MicroAPI::RegTensor<float> weight;
            AscendC::MicroAPI::RegTensor<float> product;
            AscendC::MicroAPI::LoadAlign<float>(sum, rowBase + centerOffset);
            AscendC::MicroAPI::Duplicate(weight, weights[15U]);
            AscendC::MicroAPI::Mul(sum, sum, weight, mask);
#pragma unroll 1
            for (uint32_t offset = 1U; offset <= 9U; ++offset) {
                const uint32_t elementOffset = offset * K31_ROW_C4_SIMD_CHANNELS;
                if ((offset & 1U) == 0U) {
                    AscendC::MicroAPI::LoadAlign<float>(
                        left, rowBase + centerOffset - elementOffset);
                    AscendC::MicroAPI::LoadAlign<float>(
                        right, rowBase + centerOffset + elementOffset);
                } else {
                    __ubuf__ float* leftAddress = rowBase + centerOffset - elementOffset;
                    __ubuf__ float* rightAddress = rowBase + centerOffset + elementOffset;
                    AscendC::MicroAPI::LoadUnAlignPre(leftUnalign, leftAddress);
                    AscendC::MicroAPI::LoadUnAlign(left, leftUnalign, leftAddress);
                    AscendC::MicroAPI::LoadUnAlignPre(rightUnalign, rightAddress);
                    AscendC::MicroAPI::LoadUnAlign(right, rightUnalign, rightAddress);
                }
                AscendC::MicroAPI::Add(pair, left, right, mask);
                AscendC::MicroAPI::Duplicate(weight, weights[15U - offset]);
                AscendC::MicroAPI::Mul(product, pair, weight, mask);
                AscendC::MicroAPI::Add(sum, sum, product, mask);
            }
            // This prefix is no longer needed by later output vectors.
            AscendC::MicroAPI::StoreAlign<float>(rowBase + outputOffset, sum, mask);
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(512) inline void GaussianBlurRowC4K31SimdStoreKernel(
    uint32_t width,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    __ubuf__ const float* output,
    __gm__ float* dst)
{
    const uint32_t localX = threadIdx.x;
    if (localX >= GAUSSIAN_BLUR_ROW_TILE_W) {
        return;
    }
    auto* output4 = reinterpret_cast<__ubuf__ const float4*>(output);
    auto* dst4 = reinterpret_cast<__gm__ float4*>(dst);
#pragma unroll
    for (uint32_t row = 0U; row < GAUSSIAN_BLUR_ROW_TILE_H; ++row) {
        dst4[PixelOffset(tileBaseY + row, tileBaseX + localX, width)] =
            output4[row * ROW_SHARED_W + localX];
    }
}

__aicore__ inline void ProcessRowC4K31SimdInteriorTile(
    uint32_t width,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ float* dst)
{
    asc_vf_call<GaussianBlurRowC4K31SimdInteriorVF>(shared, weights);
    asc_vf_call<GaussianBlurRowC4K31SimdStoreKernel>(
        dim3{GAUSSIAN_BLUR_ROW_TILE_W, 1U, 1U},
        width, tileBaseX, tileBaseY, shared, dst);
}

static constexpr uint32_t K31_ROW_C16_SIMD_CHANNELS = 16U;
static constexpr uint32_t K31_ROW_C16_SIMD_INPUT_ROW_ELEMENTS =
    ROW_SHARED_W * K31_ROW_C16_SIMD_CHANNELS;
static constexpr uint32_t K31_ROW_C16_SIMD_OUTPUT_ROW_ELEMENTS =
    GAUSSIAN_BLUR_ROW_TILE_W * K31_ROW_C16_SIMD_CHANNELS;
static constexpr uint32_t K31_ROW_C16_SIMD_CENTER_OFFSET =
    GAUSSIAN_BLUR_ROW_BLOCK_X * K31_ROW_C16_SIMD_CHANNELS;

__simd_vf__ inline void GaussianBlurRowC16K31SimdInteriorVF(
    __ubuf__ float* shared,
    __ubuf__ const float* weights,
    uint32_t rows)
{
    auto mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    constexpr uint32_t vectorElements = 64U;
    constexpr uint32_t vectorsPerRow = K31_ROW_C16_SIMD_OUTPUT_ROW_ELEMENTS / vectorElements;
    for (uint32_t row = 0U; row < rows; ++row) {
        __ubuf__ float* rowBase = shared + row * K31_ROW_C16_SIMD_INPUT_ROW_ELEMENTS;
        for (uint32_t vectorIndex = 0U; vectorIndex < vectorsPerRow; ++vectorIndex) {
            const uint32_t outputOffset = vectorIndex * vectorElements;
            const uint32_t centerOffset = K31_ROW_C16_SIMD_CENTER_OFFSET + outputOffset;
            AscendC::MicroAPI::RegTensor<float> sum;
            AscendC::MicroAPI::RegTensor<float> pair;
            AscendC::MicroAPI::RegTensor<float> left;
            AscendC::MicroAPI::RegTensor<float> right;
            AscendC::MicroAPI::RegTensor<float> weight;
            AscendC::MicroAPI::LoadAlign<float>(sum, rowBase + centerOffset);
            AscendC::MicroAPI::Duplicate(weight, weights[15U]);
            AscendC::MicroAPI::Mul(sum, sum, weight, mask);
#pragma unroll 1
            for (uint32_t offset = 1U; offset <= 15U; ++offset) {
                const uint32_t elementOffset = offset * K31_ROW_C16_SIMD_CHANNELS;
                AscendC::MicroAPI::LoadAlign<float>(left, rowBase + centerOffset - elementOffset);
                AscendC::MicroAPI::LoadAlign<float>(right, rowBase + centerOffset + elementOffset);
                AscendC::MicroAPI::Add(pair, left, right, mask);
                AscendC::MicroAPI::Duplicate(weight, weights[15U - offset]);
                AscendC::MicroAPI::MulAddDst(sum, pair, weight, mask);
            }
            // The output prefix trails the earliest source tap by 17 pixels,
            // so forward in-place compaction cannot overwrite a future load.
            AscendC::MicroAPI::StoreAlign<float>(rowBase + outputOffset, sum, mask);
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(2048) inline void GaussianBlurRowC16K31SimdStoreKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t rows,
    __ubuf__ const float* output,
    __gm__ float* dst)
{
    const uint32_t localX = threadIdx.x;
    const uint32_t subgroup = threadIdx.z;
    const uint32_t subgroupOffset = subgroup * GAUSSIAN_BLUR_CHANNEL_TILE;
    if (localX >= GAUSSIAN_BLUR_ROW_TILE_W || subgroupOffset >= outputChannels) {
        return;
    }
    const uint32_t active = outputChannels - subgroupOffset < GAUSSIAN_BLUR_CHANNEL_TILE ?
        outputChannels - subgroupOffset : GAUSSIAN_BLUR_CHANNEL_TILE;
    for (uint32_t row = 0U; row < rows; ++row) {
        const uint32_t inputBase = row * K31_ROW_C16_SIMD_INPUT_ROW_ELEMENTS +
            localX * K31_ROW_C16_SIMD_CHANNELS + subgroupOffset;
        const uint64_t outputBase = ChunkMajorOffset(
            tileBaseY + row, tileBaseX + localX, height, width,
            channelOffset + subgroupOffset, active);
        dst[outputBase] = output[inputBase];
        if (active >= 2U) dst[outputBase + 1U] = output[inputBase + 1U];
        if (active >= 3U) dst[outputBase + 2U] = output[inputBase + 2U];
        if (active >= 4U) dst[outputBase + 3U] = output[inputBase + 3U];
    }
}

__aicore__ inline void ProcessRowC16K31SimdInteriorTile(
    uint32_t height,
    uint32_t width,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t rows,
    __ubuf__ const float* weights,
    __ubuf__ float* shared,
    __gm__ float* dst)
{
    asc_vf_call<GaussianBlurRowC16K31SimdInteriorVF>(shared, weights, rows);
    asc_vf_call<GaussianBlurRowC16K31SimdStoreKernel>(
        dim3{GAUSSIAN_BLUR_ROW_TILE_W, 1U,
             K31_ROW_C16_SIMD_CHANNELS / GAUSSIAN_BLUR_CHANNEL_TILE},
        height, width, channelOffset, outputChannels, tileBaseX, tileBaseY,
        rows, shared, dst);
}

__simt_vf__ __aicore__ __launch_bounds__(64) inline void GaussianBlurRowC16K31FillHaloKernel(
    uint32_t rows,
    uint32_t inputPixels,
    uint32_t validLocalStart,
    uint32_t validLocalEnd,
    int32_t logicalStartX,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t sourceY,
    __gm__ const float* src,
    __ubuf__ float* local)
{
    const uint32_t haloPixels = validLocalStart + inputPixels - validLocalEnd;
    for (uint32_t task = threadIdx.x; task < rows * haloPixels; task += blockDim.x) {
        const uint32_t row = task / haloPixels;
        const uint32_t haloIndex = task % haloPixels;
        const uint32_t localX = haloIndex < validLocalStart ?
            haloIndex : validLocalEnd + haloIndex - validLocalStart;
        const uint32_t sourceX = static_cast<uint32_t>(BorderCoord(
            logicalStartX + static_cast<int32_t>(localX), static_cast<int32_t>(width),
            GAUSSIAN_BLUR_PADDING_REFLECT_101));
        const uint64_t sourceBase = ElementOffset(sourceY + row, sourceX, width, channels, channelOffset);
        const uint32_t localBase = (row * ROW_SHARED_W + localX) * K31_ROW_C16_SIMD_CHANNELS;
#pragma unroll
        for (uint32_t channel = 0U; channel < K31_ROW_C16_SIMD_CHANNELS; ++channel) {
            local[localBase + channel] = channel < outputChannels ? src[sourceBase + channel] : 0.0f;
        }
    }
}

__simd_vf__ inline void GaussianBlurRowC16K31SimdAllVF(
    __ubuf__ float* shared,
    __ubuf__ const float* weights,
    uint32_t rows,
    uint32_t outputPixels)
{
    auto mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    constexpr uint32_t vectorElements = 64U;
    const uint32_t vectorsPerRow = (outputPixels * K31_ROW_C16_SIMD_CHANNELS + vectorElements - 1U) /
        vectorElements;
    for (uint32_t row = 0U; row < rows; ++row) {
        __ubuf__ float* rowBase = shared + row * K31_ROW_C16_SIMD_INPUT_ROW_ELEMENTS;
        for (uint32_t vectorIndex = 0U; vectorIndex < vectorsPerRow; ++vectorIndex) {
            const uint32_t outputOffset = vectorIndex * vectorElements;
            const uint32_t centerOffset = K31_ROW_C16_SIMD_CENTER_OFFSET + outputOffset;
            AscendC::MicroAPI::RegTensor<float> sum;
            AscendC::MicroAPI::RegTensor<float> pair;
            AscendC::MicroAPI::RegTensor<float> left;
            AscendC::MicroAPI::RegTensor<float> right;
            AscendC::MicroAPI::RegTensor<float> weight;
            AscendC::MicroAPI::RegTensor<float> product;
            AscendC::MicroAPI::LoadAlign<float>(sum, rowBase + centerOffset);
            AscendC::MicroAPI::Duplicate(weight, weights[15U]);
            AscendC::MicroAPI::Mul(sum, sum, weight, mask);
#pragma unroll 1
            for (uint32_t offset = 1U; offset <= 15U; ++offset) {
                const uint32_t elementOffset = offset * K31_ROW_C16_SIMD_CHANNELS;
                AscendC::MicroAPI::LoadAlign<float>(left, rowBase + centerOffset - elementOffset);
                AscendC::MicroAPI::LoadAlign<float>(right, rowBase + centerOffset + elementOffset);
                AscendC::MicroAPI::Add(pair, left, right, mask);
                AscendC::MicroAPI::Duplicate(weight, weights[15U - offset]);
                AscendC::MicroAPI::Mul(product, pair, weight, mask);
                AscendC::MicroAPI::Add(sum, sum, product, mask);
            }
            AscendC::MicroAPI::StoreAlign<float>(rowBase + outputOffset, sum, mask);
        }
    }
}

__aicore__ inline void CopyRowC16K31AllToUb(
    const AscendC::GlobalTensor<float>& srcGlobal,
    AscendC::LocalTensor<float>& local,
    uint32_t tileBaseX,
    uint32_t sourceY,
    uint32_t rows,
    uint32_t outputPixels,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t outputChannels)
{
    const uint32_t computePixels = (outputPixels + 3U) / 4U * 4U;
    const uint32_t inputPixels = computePixels + 30U;
    const int32_t logicalStartX = static_cast<int32_t>(tileBaseX) - 15;
    const uint32_t validLocalStart = logicalStartX < 0 ? static_cast<uint32_t>(-logicalStartX) : 0U;
    const uint32_t validSourceStart = logicalStartX < 0 ? 0U : static_cast<uint32_t>(logicalStartX);
    const uint32_t logicalEnd = static_cast<uint32_t>(logicalStartX + static_cast<int32_t>(inputPixels));
    const uint32_t validSourceEnd = logicalEnd < width ? logicalEnd : width;
    const uint32_t validPixels = validSourceEnd - validSourceStart;
    const uint32_t validLocalEnd = validLocalStart + validPixels;
    const uint32_t blockBytes = outputChannels * sizeof(float);
    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(validPixels), blockBytes,
        static_cast<int64_t>(channels - outputChannels) * static_cast<int64_t>(sizeof(float)), 0, 0U};
    AscendC::DataCopyPadExtParams<float> pad{
        true, 0U, static_cast<uint8_t>(K31_ROW_C16_SIMD_CHANNELS - outputChannels), 0.0f};
    for (uint32_t row = 0U; row < rows; ++row) {
        const uint64_t sourceOffset = ElementOffsetAicore(
            sourceY + row, validSourceStart, width, channels, channelOffset);
        AscendC::DataCopyPad(
            local[row * ROW_SHARED_W * K31_ROW_C16_SIMD_CHANNELS +
                  validLocalStart * K31_ROW_C16_SIMD_CHANNELS],
            srcGlobal[sourceOffset], params, pad);
    }
}

__aicore__ inline void StoreRowC16K31GroupMajor(
    AscendC::GlobalTensor<float>& dstGlobal,
    AscendC::LocalTensor<float>& local,
    uint32_t height,
    uint32_t width,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t rows,
    uint32_t outputPixels)
{
    const uint64_t pixels = static_cast<uint64_t>(height) * width;
    const uint32_t blockBytes = outputChannels * sizeof(float);
    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(outputPixels), blockBytes, 0, 0, 0U};
    for (uint32_t row = 0U; row < rows; ++row) {
        const uint64_t destinationOffset = static_cast<uint64_t>(channelOffset) * pixels +
            PixelOffsetAicore(tileBaseY + row, tileBaseX, width) * outputChannels;
        AscendC::DataCopyPad(
            dstGlobal[destinationOffset],
            local[row * ROW_SHARED_W * K31_ROW_C16_SIMD_CHANNELS], params);
    }
}

__aicore__ inline void ProcessRowK31C16GroupMajorAllTiles(
    GM_ADDR src, GM_ADDR dst, const GaussianBlurTilingData* tilingData)
{
    using namespace AscendC;
    LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    LocalTensor<float> weightTensor = ubAllocator.Alloc<float>(WEIGHT_UB_ELEMENTS);
    LocalTensor<float> local = ubAllocator.Alloc<float>(GAUSSIAN_BLUR_ROW_UB_BUFFER_BYTES / sizeof(float));
    __ubuf__ float* weights = reinterpret_cast<__ubuf__ float*>(weightTensor.GetPhyAddr());
#pragma unroll
    for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
        weights[index] = tilingData->weights[index];
    }
    DataSyncBarrier<MemDsbT::UB>();
    GlobalTensor<float> srcGlobal;
    GlobalTensor<float> dstGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst));
    TPipe pipe;
    const int32_t eventMte2ToV = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    const int32_t eventVToMte3 = static_cast<int32_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    const int32_t eventMte3ToMte2 = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    const uint32_t spatialTiles = tilingData->tilesX * tilingData->tilesY;
    const uint32_t channelGroups = (tilingData->c + K31_ROW_C16_SIMD_CHANNELS - 1U) /
        K31_ROW_C16_SIMD_CHANNELS;
    const uint32_t workItemsPerTile = GAUSSIAN_BLUR_ROW_TILE_H / 4U;
    const uint32_t totalWorkItems = spatialTiles * channelGroups * workItemsPerTile;
    for (uint32_t workId = GetBlockIdx(); workId < totalWorkItems; workId += GetBlockNum()) {
        const uint32_t microTile = workId % workItemsPerTile;
        const uint32_t tileId = workId / workItemsPerTile;
        const uint32_t spatialTile = tileId % spatialTiles;
        const uint32_t channelOffset = (tileId / spatialTiles) * K31_ROW_C16_SIMD_CHANNELS;
        const uint32_t outputChannels = channelOffset + K31_ROW_C16_SIMD_CHANNELS <= tilingData->c ?
            K31_ROW_C16_SIMD_CHANNELS : tilingData->c - channelOffset;
        const uint32_t tileX = spatialTile % tilingData->tilesX;
        const uint32_t tileY = spatialTile / tilingData->tilesX;
        const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_ROW_TILE_W;
        const uint32_t tileBaseY = tileY * GAUSSIAN_BLUR_ROW_TILE_H + microTile * 4U;
        if (tileBaseY >= tilingData->h) continue;
        const uint32_t rows = tileBaseY + 4U <= tilingData->h ? 4U : tilingData->h - tileBaseY;
        const uint32_t outputPixels = tileBaseX + GAUSSIAN_BLUR_ROW_TILE_W <= tilingData->w ?
            GAUSSIAN_BLUR_ROW_TILE_W : tilingData->w - tileBaseX;
        const uint32_t computePixels = (outputPixels + 3U) / 4U * 4U;
        const uint32_t inputPixels = computePixels + 30U;
        const int32_t logicalStartX = static_cast<int32_t>(tileBaseX) - 15;
        const uint32_t validLocalStart = logicalStartX < 0 ? static_cast<uint32_t>(-logicalStartX) : 0U;
        const uint32_t validSourceStart = logicalStartX < 0 ? 0U : static_cast<uint32_t>(logicalStartX);
        const uint32_t logicalEnd = static_cast<uint32_t>(logicalStartX + static_cast<int32_t>(inputPixels));
        const uint32_t validSourceEnd = logicalEnd < tilingData->w ? logicalEnd : tilingData->w;
        const uint32_t validLocalEnd = validLocalStart + validSourceEnd - validSourceStart;
        CopyRowC16K31AllToUb(srcGlobal, local, tileBaseX, tileBaseY, rows, outputPixels,
                            tilingData->w, tilingData->c, channelOffset, outputChannels);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        if (validLocalStart != 0U || validLocalEnd != inputPixels) {
            asc_vf_call<GaussianBlurRowC16K31FillHaloKernel>(
                dim3{64U, 1U, 1U}, rows, inputPixels, validLocalStart, validLocalEnd,
                logicalStartX, tilingData->w, tilingData->c, channelOffset, outputChannels,
                tileBaseY, reinterpret_cast<__gm__ const float*>(src),
                reinterpret_cast<__ubuf__ float*>(local.GetPhyAddr()));
        }
        asc_vf_call<GaussianBlurRowC16K31SimdAllVF>(
            reinterpret_cast<__ubuf__ float*>(local.GetPhyAddr()), weights, rows, outputPixels);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        StoreRowC16K31GroupMajor(dstGlobal, local, tilingData->h, tilingData->w,
                                channelOffset, outputChannels, tileBaseX, tileBaseY, rows, outputPixels);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    }
}
#endif

template <uint32_t ChannelStride, bool DynamicChannels>
__aicore__ inline bool IsFullRowInteriorTile(
    uint32_t tileId, uint32_t spatialTiles, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t tilesX)
{
    const uint32_t spatialTileId = tileId % spatialTiles;
    if constexpr (DynamicChannels) {
        constexpr uint32_t dynamicChannelGroup =
            ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP ?
                ChannelStride : GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP;
        const uint32_t channelOffset =
            (tileId / spatialTiles) * dynamicChannelGroup;
        if constexpr (ChannelStride < GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP) {
            if (channelOffset + GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP > channels) {
                return false;
            }
        } else {
            if (channelOffset >= channels) {
                return false;
            }
        }
    }
    const uint32_t tileX = spatialTileId % tilesX;
    const uint32_t tileY = spatialTileId / tilesX;
    const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_ROW_TILE_W;
    const uint32_t tileBaseY = tileY * GAUSSIAN_BLUR_ROW_TILE_H;
    return tileBaseX >= GAUSSIAN_BLUR_ROW_BLOCK_X &&
        tileBaseX + ROW_SHARED_W - GAUSSIAN_BLUR_ROW_BLOCK_X <= width &&
        tileBaseY + GAUSSIAN_BLUR_ROW_TILE_H <= height;
}

template <uint32_t ChannelStride, bool DynamicChannels>
__aicore__ inline uint32_t FindNextRowInteriorTile(
    uint32_t candidate,
    uint32_t totalTiles,
    uint32_t spatialTiles,
    uint32_t coreCount,
    uint32_t height,
    uint32_t width,
    uint32_t tilesX,
    uint32_t channels)
{
    while (candidate < totalTiles &&
           !IsFullRowInteriorTile<ChannelStride, DynamicChannels>(
               candidate, spatialTiles, channels, height, width, tilesX)) {
        candidate += coreCount;
    }
    return candidate;
}

template <uint32_t ChannelStride, bool DynamicChannels>
__aicore__ inline void CopyRowInteriorTileToUb(
    const AscendC::GlobalTensor<float>& srcGlobal,
    AscendC::LocalTensor<float>& local,
    uint32_t tileId,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t tilesX)
{
    const uint32_t tileX = tileId % tilesX;
    const uint32_t tileY = tileId / tilesX;
    const uint32_t sourceX = tileX * GAUSSIAN_BLUR_ROW_TILE_W -
        GAUSSIAN_BLUR_ROW_BLOCK_X;
    const uint32_t sourceY = tileY * GAUSSIAN_BLUR_ROW_TILE_H;
    AscendC::DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
    if constexpr (!DynamicChannels) {
        const uint64_t sourceOffset =
            (static_cast<uint64_t>(sourceY) * width + sourceX) * ChannelStride;
        const uint32_t rowBytes = ROW_SHARED_W * ChannelStride * sizeof(float);
        const uint32_t sourceStrideBytes = (width - ROW_SHARED_W) * ChannelStride * sizeof(float);
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(GAUSSIAN_BLUR_ROW_TILE_H), rowBytes, sourceStrideBytes, 0U, 0U};
        AscendC::DataCopyPad(local, srcGlobal[sourceOffset], params, pad);
    } else {
        const uint32_t pixelBytes = outputChannels * sizeof(float);
        const uint32_t sourceStrideBytes = (channels - outputChannels) * sizeof(float);
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(ROW_SHARED_W), pixelBytes, sourceStrideBytes, 0U, 0U};
        for (uint32_t row = 0U; row < GAUSSIAN_BLUR_ROW_TILE_H; ++row) {
            const uint64_t sourceOffset =
                ((static_cast<uint64_t>(sourceY + row) * width + sourceX) * channels) + channelOffset;
            AscendC::DataCopyPad<float, AscendC::PaddingMode::Compact>(
                local[row * ROW_SHARED_W * ChannelStride], srcGlobal[sourceOffset], params, pad);
        }
    }
}

template <uint32_t ChannelStride>
__aicore__ inline void CopyGenericRowInteriorMicroTileToUb(
    const AscendC::GlobalTensor<float>& srcGlobal,
    AscendC::LocalTensor<float>& local,
    uint32_t spatialTileId,
    uint32_t rowOffset,
    uint32_t width,
    uint32_t channels,
    uint32_t activeChannels,
    uint32_t channelOffset,
    uint32_t tilesX,
    uint32_t microTileRows)
{
    const uint32_t tileX = spatialTileId % tilesX;
    const uint32_t tileY = spatialTileId / tilesX;
    const uint32_t sourceX = tileX * GAUSSIAN_BLUR_ROW_TILE_W - GAUSSIAN_BLUR_ROW_BLOCK_X;
    const uint32_t sourceY = tileY * GAUSSIAN_BLUR_ROW_TILE_H + rowOffset;
    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(ROW_SHARED_W),
        static_cast<uint32_t>(activeChannels * sizeof(float)),
        static_cast<int64_t>(channels - activeChannels) * static_cast<int64_t>(sizeof(float)),
        static_cast<int64_t>(0), 0U};
    AscendC::DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
    for (uint32_t row = 0U; row < microTileRows; ++row) {
        const uint64_t sourceOffset =
            ((static_cast<uint64_t>(sourceY + row) * width + sourceX) * channels) + channelOffset;
        if constexpr (ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP) {
            AscendC::DataCopyPad<float, AscendC::PaddingMode::Normal>(
                local[row * ROW_SHARED_W * ChannelStride], srcGlobal[sourceOffset], params, pad);
        } else {
            AscendC::DataCopyPad<float, AscendC::PaddingMode::Compact>(
                local[row * ROW_SHARED_W * ChannelStride], srcGlobal[sourceOffset], params, pad);
        }
    }
}

template <uint32_t ChannelStride, bool DynamicChannels>
__aicore__ inline void ProcessRowInteriorPipeline(
    GM_ADDR src,
    GM_ADDR dst,
    const GaussianBlurTilingData* tilingData)
{
    using namespace AscendC;
    GlobalTensor<float> srcGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    TPipe pipe;
    TBuf<TPosition::VECCALC> weightBuffer;
    TBuf<TPosition::VECCALC> buffer0;
    TBuf<TPosition::VECCALC> buffer1;
    pipe.InitBuffer(weightBuffer, WEIGHT_UB_ELEMENTS * sizeof(float));
    pipe.InitBuffer(buffer0, GAUSSIAN_BLUR_ROW_UB_BUFFER_BYTES);
    pipe.InitBuffer(buffer1, GAUSSIAN_BLUR_ROW_UB_BUFFER_BYTES);
    LocalTensor<float> weightTensor = weightBuffer.Get<float>();
    LocalTensor<float> local0 = buffer0.Get<float>();
    LocalTensor<float> local1 = buffer1.Get<float>();
    __ubuf__ float* weights = reinterpret_cast<__ubuf__ float*>(weightTensor.GetPhyAddr());
#pragma unroll
    for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
        weights[index] = tilingData->weights[index];
    }
    DataSyncBarrier<MemDsbT::UB>();
    const int32_t eventMte2ToV = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    const int32_t eventVToMte2 = static_cast<int32_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    const uint32_t spatialTiles = tilingData->tilesX * tilingData->tilesY;
    constexpr uint32_t dynamicChannelGroup =
        ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP ?
            ChannelStride : GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP;
    const uint32_t channelTiles = DynamicChannels ?
        (tilingData->c + dynamicChannelGroup - 1U) / dynamicChannelGroup : 1U;
    const uint32_t pipelineTiles = spatialTiles * channelTiles;
    const uint32_t coreCount = GetBlockNum();
    uint32_t currentTile = FindNextRowInteriorTile<ChannelStride, DynamicChannels>(
        GetBlockIdx(), pipelineTiles, spatialTiles, coreCount,
        tilingData->h, tilingData->w, tilingData->tilesX, tilingData->c);
    if (currentTile >= pipelineTiles) {
        return;
    }

    if constexpr (DynamicChannels) {
        constexpr uint32_t microTileRows = 4U;
        constexpr uint32_t microTilesPerSubgroup = GAUSSIAN_BLUR_ROW_TILE_H / microTileRows;
        constexpr uint32_t subgroupCount = ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP ?
            1U : GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP / GAUSSIAN_BLUR_CHANNEL_TILE;
        constexpr uint32_t workItemsPerTile = microTilesPerSubgroup * subgroupCount;
        uint32_t currentWorkItem = 0U;
        uint32_t currentBuffer = 0U;

        const uint32_t firstSpatialTile = currentTile % spatialTiles;
        const uint32_t firstChannelOffset =
            (currentTile / spatialTiles) * dynamicChannelGroup;
        const uint32_t firstActiveChannels = firstChannelOffset + ChannelStride <= tilingData->c ?
            ChannelStride : tilingData->c - firstChannelOffset;
        CopyGenericRowInteriorMicroTileToUb<ChannelStride>(
            srcGlobal, local0, firstSpatialTile, 0U, tilingData->w, tilingData->c,
            ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP ? firstActiveChannels : ChannelStride,
            firstChannelOffset, tilingData->tilesX, microTileRows);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        while (currentTile < pipelineTiles) {
            const uint32_t spatialTileId = currentTile % spatialTiles;
            const uint32_t channelOffset =
                (currentTile / spatialTiles) * dynamicChannelGroup;
            const uint32_t subgroup = currentWorkItem / microTilesPerSubgroup;
            const uint32_t microTile = currentWorkItem % microTilesPerSubgroup;
            const uint32_t subgroupOffset =
                channelOffset + subgroup * GAUSSIAN_BLUR_CHANNEL_TILE;
            const uint32_t rowOffset = microTile * microTileRows;
            const uint32_t tileX = spatialTileId % tilingData->tilesX;
            const uint32_t tileY = spatialTileId / tilingData->tilesX;

            uint32_t nextTile = currentTile;
            uint32_t nextWorkItem = currentWorkItem + 1U;
            if (nextWorkItem == workItemsPerTile) {
                nextTile = FindNextRowInteriorTile<ChannelStride, DynamicChannels>(
                    currentTile + coreCount, pipelineTiles, spatialTiles, coreCount,
                    tilingData->h, tilingData->w, tilingData->tilesX, tilingData->c);
                nextWorkItem = 0U;
            }
            const uint32_t nextBuffer = currentBuffer ^ 1U;
            if (nextTile < pipelineTiles) {
                const uint32_t nextSpatialTile = nextTile % spatialTiles;
                const uint32_t nextChannelOffset =
                    (nextTile / spatialTiles) * dynamicChannelGroup;
                const uint32_t nextActiveChannels = nextChannelOffset + ChannelStride <= tilingData->c ?
                    ChannelStride : tilingData->c - nextChannelOffset;
                const uint32_t nextSubgroup = nextWorkItem / microTilesPerSubgroup;
                const uint32_t nextMicroTile = nextWorkItem % microTilesPerSubgroup;
                CopyGenericRowInteriorMicroTileToUb<ChannelStride>(
                    srcGlobal, nextBuffer == 0U ? local0 : local1, nextSpatialTile,
                    nextMicroTile * microTileRows, tilingData->w, tilingData->c,
                    ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP ? nextActiveChannels : ChannelStride,
                    nextChannelOffset + nextSubgroup * GAUSSIAN_BLUR_CHANNEL_TILE,
                    tilingData->tilesX, microTileRows);
            }

            const uint32_t activeChannels = channelOffset + ChannelStride <= tilingData->c ?
                ChannelStride : tilingData->c - channelOffset;

#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
            if constexpr (ChannelStride == K31_ROW_C16_SIMD_CHANNELS) {
                ProcessRowC16K31SimdInteriorTile(
                    tilingData->h, tilingData->w, subgroupOffset,
                    activeChannels, tileX * GAUSSIAN_BLUR_ROW_TILE_W,
                    tileY * GAUSSIAN_BLUR_ROW_TILE_H + rowOffset,
                    microTileRows, weights,
                    reinterpret_cast<__ubuf__ float*>(
                        (currentBuffer == 0U ? local0 : local1).GetPhyAddr()),
                    reinterpret_cast<__gm__ float*>(dst));
            } else {
#endif
                DispatchRowInteriorUbTile<ChannelStride, true>(
                    tilingData->h, tilingData->w, tilingData->c, subgroupOffset,
                    ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP ? activeChannels : ChannelStride,
                    tileX * GAUSSIAN_BLUR_ROW_TILE_W,
                    tileY * GAUSSIAN_BLUR_ROW_TILE_H + rowOffset,
                    tilingData->kernelSize, weights,
                    reinterpret_cast<__ubuf__ const float*>(
                        (currentBuffer == 0U ? local0 : local1).GetPhyAddr()),
                    reinterpret_cast<__gm__ float*>(dst), microTileRows);
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
            }
#endif
            SetFlag<HardEvent::V_MTE2>(eventVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventVToMte2);
            if (nextTile < pipelineTiles) {
                SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
            }
            currentTile = nextTile;
            currentWorkItem = nextWorkItem;
            currentBuffer = nextBuffer;
        }
        return;
    }
    uint32_t currentSpatialTile = currentTile % spatialTiles;
    uint32_t currentChannelOffset = DynamicChannels ?
        (currentTile / spatialTiles) * GAUSSIAN_BLUR_CHANNEL_TILE : 0U;
    uint32_t currentOutputChannels = DynamicChannels ?
        (currentChannelOffset + GAUSSIAN_BLUR_CHANNEL_TILE <= tilingData->c ?
            GAUSSIAN_BLUR_CHANNEL_TILE : tilingData->c - currentChannelOffset) : ChannelStride;
    CopyRowInteriorTileToUb<ChannelStride, DynamicChannels>(
        srcGlobal, local0, currentSpatialTile, tilingData->w, tilingData->c,
        currentChannelOffset, currentOutputChannels, tilingData->tilesX);
    SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
    uint32_t currentBuffer = 0U;
    while (currentTile < pipelineTiles) {
        const uint32_t nextTile = FindNextRowInteriorTile<ChannelStride, DynamicChannels>(
            currentTile + coreCount, pipelineTiles, spatialTiles, coreCount,
            tilingData->h, tilingData->w, tilingData->tilesX, tilingData->c);
        const uint32_t nextBuffer = currentBuffer ^ 1U;
        if (nextTile < pipelineTiles) {
            const uint32_t nextSpatialTile = nextTile % spatialTiles;
            const uint32_t nextChannelOffset = DynamicChannels ?
                (nextTile / spatialTiles) * GAUSSIAN_BLUR_CHANNEL_TILE : 0U;
            const uint32_t nextOutputChannels = DynamicChannels ?
                (nextChannelOffset + GAUSSIAN_BLUR_CHANNEL_TILE <= tilingData->c ?
                    GAUSSIAN_BLUR_CHANNEL_TILE : tilingData->c - nextChannelOffset) : ChannelStride;
            CopyRowInteriorTileToUb<ChannelStride, DynamicChannels>(
                srcGlobal, nextBuffer == 0U ? local0 : local1, nextSpatialTile,
                tilingData->w, tilingData->c, nextChannelOffset, nextOutputChannels, tilingData->tilesX);
        }

        const uint32_t tileX = currentSpatialTile % tilingData->tilesX;
        const uint32_t tileY = currentSpatialTile / tilingData->tilesX;
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
        if constexpr (ChannelStride == K31_ROW_C4_SIMD_CHANNELS && !DynamicChannels) {
            if (K31_ROW_C4_SIMD_ENABLED && tilingData->kernelSize == 31U) {
                ProcessRowC4K31SimdInteriorTile(
                    tilingData->w,
                    tileX * GAUSSIAN_BLUR_ROW_TILE_W,
                    tileY * GAUSSIAN_BLUR_ROW_TILE_H,
                    weights,
                    reinterpret_cast<__ubuf__ float*>(
                        (currentBuffer == 0U ? local0 : local1).GetPhyAddr()),
                    reinterpret_cast<__gm__ float*>(dst));
            } else {
                DispatchRowInteriorUbTile<ChannelStride, DynamicChannels>(
                    tilingData->h, tilingData->w, tilingData->c,
                    currentChannelOffset, currentOutputChannels,
                    tileX * GAUSSIAN_BLUR_ROW_TILE_W,
                    tileY * GAUSSIAN_BLUR_ROW_TILE_H, tilingData->kernelSize, weights,
                    reinterpret_cast<__ubuf__ const float*>(
                        (currentBuffer == 0U ? local0 : local1).GetPhyAddr()),
                    reinterpret_cast<__gm__ float*>(dst));
            }
        } else {
#endif
        DispatchRowInteriorUbTile<ChannelStride, DynamicChannels>(
            tilingData->h, tilingData->w, tilingData->c, currentChannelOffset, currentOutputChannels,
            tileX * GAUSSIAN_BLUR_ROW_TILE_W,
            tileY * GAUSSIAN_BLUR_ROW_TILE_H, tilingData->kernelSize, weights,
            reinterpret_cast<__ubuf__ const float*>(
                (currentBuffer == 0U ? local0 : local1).GetPhyAddr()),
            reinterpret_cast<__gm__ float*>(dst));
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
        }
#endif
        SetFlag<HardEvent::V_MTE2>(eventVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventVToMte2);

        if (nextTile < pipelineTiles) {
            SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        }
        currentTile = nextTile;
        currentBuffer = nextBuffer;
        if (currentTile < pipelineTiles) {
            currentSpatialTile = currentTile % spatialTiles;
            currentChannelOffset = DynamicChannels ?
                (currentTile / spatialTiles) * GAUSSIAN_BLUR_CHANNEL_TILE : 0U;
            currentOutputChannels = DynamicChannels ?
                (currentChannelOffset + GAUSSIAN_BLUR_CHANNEL_TILE <= tilingData->c ?
                    GAUSSIAN_BLUR_CHANNEL_TILE : tilingData->c - currentChannelOffset) : ChannelStride;
        }
    }
}

__aicore__ inline bool IsFullColumnInteriorTile(
    uint32_t tileId,
    uint32_t spatialTiles,
    uint32_t channels,
    uint32_t height,
    uint32_t width,
    uint32_t tilesX)
{
    const uint32_t spatialTileId = tileId % spatialTiles;
    const uint32_t channelOffset = (tileId / spatialTiles) * GAUSSIAN_BLUR_CHANNEL_TILE;
    if (channelOffset + GAUSSIAN_BLUR_CHANNEL_TILE > channels) {
        return false;
    }
    const uint32_t tileX = spatialTileId % tilesX;
    const uint32_t tileY = spatialTileId / tilesX;
    const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_COLUMN_TILE_W;
    const uint32_t tileBaseY = tileY * COLUMN_TILE_H;
    return tileBaseX + GAUSSIAN_BLUR_COLUMN_TILE_W <= width &&
        tileBaseY >= GAUSSIAN_BLUR_COLUMN_BLOCK_Y &&
        tileBaseY + COLUMN_SHARED_H - GAUSSIAN_BLUR_COLUMN_BLOCK_Y <= height;
}

__aicore__ inline uint32_t FindNextColumnInteriorTile(
    uint32_t candidate,
    uint32_t totalTiles,
    uint32_t spatialTiles,
    uint32_t coreCount,
    uint32_t channels,
    uint32_t height,
    uint32_t width,
    uint32_t tilesX)
{
    while (candidate < totalTiles &&
           !IsFullColumnInteriorTile(candidate, spatialTiles, channels, height, width, tilesX)) {
        candidate += coreCount;
    }
    return candidate;
}

__aicore__ inline void CopyColumnInteriorGroupToUb(
    const AscendC::GlobalTensor<float>& srcGlobal,
    AscendC::LocalTensor<float> local,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t groupX,
    uint32_t height,
    uint32_t width,
    uint32_t channelOffset)
{
    constexpr uint32_t groupWidth = 4U;
    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(COLUMN_SHARED_H), GAUSSIAN_BLUR_CHANNEL_TILE * sizeof(float),
        static_cast<int64_t>(width * GAUSSIAN_BLUR_CHANNEL_TILE - GAUSSIAN_BLUR_CHANNEL_TILE) *
            static_cast<int64_t>(sizeof(float)),
        static_cast<int64_t>(0), 0U};
    AscendC::DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
    const uint32_t sourceY = tileBaseY - GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
    for (uint32_t localX = 0U; localX < groupWidth; ++localX) {
        const uint32_t sourceX = tileBaseX + groupX + localX;
        const uint64_t sourceOffset = ChunkMajorOffsetAicore(
            sourceY, sourceX, height, width, channelOffset, GAUSSIAN_BLUR_CHANNEL_TILE);
        AscendC::DataCopyPad<float, AscendC::PaddingMode::Compact>(
            local[localX * COLUMN_SHARED_H * GAUSSIAN_BLUR_CHANNEL_TILE],
            srcGlobal[sourceOffset], params, pad);
    }
}

__aicore__ inline void CopyColumnInteriorC8ToUb(
    const AscendC::GlobalTensor<float>& srcGlobal,
    AscendC::LocalTensor<float>& local,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t width,
    uint32_t channels)
{
    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(COLUMN_SHARED_H),
        static_cast<uint32_t>(channels * sizeof(float)),
        static_cast<int64_t>(width * channels - channels) * static_cast<int64_t>(sizeof(float)),
        static_cast<int64_t>(0), 0U};
    AscendC::DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
    const uint32_t sourceY = tileBaseY - GAUSSIAN_BLUR_COLUMN_BLOCK_Y;
    for (uint32_t localX = 0U; localX < GAUSSIAN_BLUR_COLUMN_TILE_W; ++localX) {
        const uint64_t sourceOffset =
            (static_cast<uint64_t>(sourceY) * width + tileBaseX + localX) * channels;
        AscendC::DataCopyPad<float, AscendC::PaddingMode::Normal>(
            local[localX * COLUMN_SHARED_H * GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP],
            srcGlobal[sourceOffset], params, pad);
    }
}

__aicore__ inline bool IsC8ColumnInteriorSpatialTile(
    uint32_t spatialTileId,
    uint32_t height,
    uint32_t width,
    uint32_t tilesX)
{
    const uint32_t tileX = spatialTileId % tilesX;
    const uint32_t tileY = spatialTileId / tilesX;
    const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_COLUMN_TILE_W;
    const uint32_t tileBaseY = tileY * COLUMN_TILE_H;
    return tileBaseX + GAUSSIAN_BLUR_COLUMN_TILE_W <= width &&
        tileBaseY >= GAUSSIAN_BLUR_COLUMN_BLOCK_Y &&
        tileBaseY + COLUMN_SHARED_H - GAUSSIAN_BLUR_COLUMN_BLOCK_Y <= height;
}

__aicore__ inline uint32_t FindNextC8ColumnInteriorSpatialTile(
    uint32_t candidate,
    uint32_t spatialTiles,
    uint32_t coreCount,
    uint32_t height,
    uint32_t width,
    uint32_t tilesX)
{
    while (candidate < spatialTiles &&
           !IsC8ColumnInteriorSpatialTile(candidate, height, width, tilesX)) {
        candidate += coreCount;
    }
    return candidate;
}

__aicore__ inline void ProcessColumnC8InteriorMte(
    GM_ADDR src,
    GM_ADDR dst,
    const GaussianBlurTilingData* tilingData,
    __ubuf__ float* weights,
    AscendC::LocalTensor<float>& local)
{
    using namespace AscendC;
    GlobalTensor<float> srcGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    const uint32_t spatialTiles = tilingData->tilesX * tilingData->tilesY;
    const uint32_t coreCount = GetBlockNum();
    uint32_t spatialTileId = FindNextC8ColumnInteriorSpatialTile(
        GetBlockIdx(), spatialTiles, coreCount, tilingData->h, tilingData->w, tilingData->tilesX);
    if (spatialTileId >= spatialTiles) {
        return;
    }
    TPipe pipe;
    const int32_t eventMte2ToV = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    const int32_t eventVToMte2 = static_cast<int32_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    while (spatialTileId < spatialTiles) {
        const uint32_t tileX = spatialTileId % tilingData->tilesX;
        const uint32_t tileY = spatialTileId / tilingData->tilesX;
        const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_COLUMN_TILE_W;
        const uint32_t tileBaseY = tileY * COLUMN_TILE_H;
        CopyColumnInteriorC8ToUb(
            srcGlobal, local, tileBaseX, tileBaseY, tilingData->w, tilingData->c);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        DispatchColumnInteriorXMajorC8Tile(
            tilingData->w, tilingData->c, tileBaseX, tileBaseY, tilingData->kernelSize,
            weights, reinterpret_cast<__ubuf__ const float*>(local.GetPhyAddr()),
            reinterpret_cast<__gm__ float*>(dst));
        SetFlag<HardEvent::V_MTE2>(eventVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventVToMte2);
        spatialTileId = FindNextC8ColumnInteriorSpatialTile(
            spatialTileId + coreCount, spatialTiles, coreCount,
            tilingData->h, tilingData->w, tilingData->tilesX);
    }
}

__aicore__ inline void ProcessColumnInteriorPipeline(
    GM_ADDR src,
    GM_ADDR dst,
    const GaussianBlurTilingData* tilingData,
    __ubuf__ const float* weights,
    AscendC::LocalTensor<float>& local0,
    AscendC::LocalTensor<float>& local1)
{
    using namespace AscendC;
    GlobalTensor<float> srcGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    const uint32_t spatialTiles = tilingData->tilesX * tilingData->tilesY;
    const uint32_t channelTiles =
        (tilingData->c + GAUSSIAN_BLUR_CHANNEL_TILE - 1U) / GAUSSIAN_BLUR_CHANNEL_TILE;
    const uint32_t totalTiles = spatialTiles * channelTiles;
    const uint32_t coreCount = GetBlockNum();
    uint32_t tileId = FindNextColumnInteriorTile(
        GetBlockIdx(), totalTiles, spatialTiles, coreCount, tilingData->c,
        tilingData->h, tilingData->w, tilingData->tilesX);
    if (tileId >= totalTiles) {
        return;
    }
    TPipe pipe;
    const int32_t eventMte2ToV = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    const int32_t eventVToMte2 = static_cast<int32_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    constexpr uint32_t groupWidth = 4U;
    constexpr uint32_t groupCount = GAUSSIAN_BLUR_COLUMN_TILE_W / groupWidth;
    while (tileId < totalTiles) {
        const uint32_t spatialTileId = tileId % spatialTiles;
        const uint32_t channelOffset =
            (tileId / spatialTiles) * GAUSSIAN_BLUR_CHANNEL_TILE;
        const uint32_t tileX = spatialTileId % tilingData->tilesX;
        const uint32_t tileY = spatialTileId / tilingData->tilesX;
        const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_COLUMN_TILE_W;
        const uint32_t tileBaseY = tileY * COLUMN_TILE_H;
        for (uint32_t group = 0U; group < groupCount; ++group) {
            CopyColumnInteriorGroupToUb(
                srcGlobal, local0[group * groupWidth * COLUMN_SHARED_H * GAUSSIAN_BLUR_CHANNEL_TILE],
                tileBaseX, tileBaseY, group * groupWidth,
                tilingData->h, tilingData->w, channelOffset);
        }
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        __ubuf__ const float* currentLocal =
            reinterpret_cast<__ubuf__ const float*>(local0.GetPhyAddr());
        asc_vf_call<ColumnInteriorXMajorSlidingTile<21U>>(
            dim3{GAUSSIAN_BLUR_COLUMN_TILE_W, GAUSSIAN_BLUR_COLUMN_BLOCK_Y, 1U},
            tilingData->w, tilingData->c, channelOffset, tileBaseX, tileBaseY,
            weights, currentLocal, reinterpret_cast<__gm__ float*>(dst));
        SetFlag<HardEvent::V_MTE2>(eventVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventVToMte2);
        tileId = FindNextColumnInteriorTile(
            tileId + coreCount, totalTiles, spatialTiles, coreCount, tilingData->c,
            tilingData->h, tilingData->w, tilingData->tilesX);
    }
}

#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
__aicore__ inline uint32_t Reflect101ColumnCoord(int32_t coord, uint32_t height)
{
    int32_t reflected = coord;
    const int32_t limit = static_cast<int32_t>(height);
    while (reflected < 0 || reflected >= limit) {
        reflected = reflected < 0 ? -reflected : 2 * limit - reflected - 2;
    }
    return static_cast<uint32_t>(reflected);
}

static constexpr uint32_t K31_COLUMN_C16_CHANNELS = 16U;
static constexpr uint32_t K31_COLUMN_C16_ROW_ELEMENTS =
    GAUSSIAN_BLUR_COLUMN_TILE_W * K31_COLUMN_C16_CHANNELS;

__aicore__ inline void CopyColumnK31C16RowRangeToUb(
    const AscendC::GlobalTensor<float>& srcGlobal,
    AscendC::LocalTensor<float>& local,
    uint32_t destinationRow,
    uint32_t sourceRow,
    uint32_t rowCount,
    uint32_t tileBaseX,
    uint32_t activeWidth,
    uint32_t height,
    uint32_t width,
    uint32_t channelOffset,
    uint32_t outputChannels)
{
    constexpr uint32_t fullRowBytes = K31_COLUMN_C16_ROW_ELEMENTS * sizeof(float);
    const uint64_t pixels = static_cast<uint64_t>(height) * width;
    const uint64_t sourceOffset = static_cast<uint64_t>(channelOffset) * pixels +
        PixelOffsetAicore(sourceRow, tileBaseX, width) * outputChannels;
    if (outputChannels < K31_COLUMN_C16_CHANNELS) {
        const uint32_t blockLen = outputChannels * sizeof(float);
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(activeWidth), blockLen, 0, 0, 0U};
        AscendC::DataCopyPadExtParams<float> pad{
            true, 0U, static_cast<uint8_t>(K31_COLUMN_C16_CHANNELS - outputChannels), 0.0f};
        AscendC::DataCopyPad(
            local[destinationRow * K31_COLUMN_C16_ROW_ELEMENTS],
            srcGlobal[sourceOffset], params, pad);
        return;
    }

    const uint32_t blockLen = activeWidth * K31_COLUMN_C16_CHANNELS * sizeof(float);
    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(rowCount), blockLen,
        static_cast<int64_t>((width - activeWidth) * K31_COLUMN_C16_CHANNELS * sizeof(float)),
        static_cast<int64_t>((fullRowBytes - blockLen) / 32U), 0U};
    AscendC::DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0f};
    AscendC::DataCopyPad(
        local[destinationRow * K31_COLUMN_C16_ROW_ELEMENTS], srcGlobal[sourceOffset], params, pad);
}

__simd_vf__ inline void GaussianBlurColumnK31C16SimdVF(
    __ubuf__ float* input,
    __ubuf__ const float* weights,
    uint32_t outputRows)
{
    auto mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    constexpr uint32_t vectorsPerRow = K31_COLUMN_C16_ROW_ELEMENTS / 64U;
    for (uint32_t outputRow = 0U; outputRow < outputRows; ++outputRow) {
        const uint32_t centerRow = outputRow + 15U;
        for (uint32_t vectorIndex = 0U; vectorIndex < vectorsPerRow; ++vectorIndex) {
            const uint32_t vectorOffset = vectorIndex * 64U;
            AscendC::MicroAPI::RegTensor<float> sum;
            AscendC::MicroAPI::RegTensor<float> pair;
            AscendC::MicroAPI::RegTensor<float> upper;
            AscendC::MicroAPI::RegTensor<float> lower;
            AscendC::MicroAPI::RegTensor<float> weight;
            AscendC::MicroAPI::LoadAlign<float>(
                sum, input + centerRow * K31_COLUMN_C16_ROW_ELEMENTS + vectorOffset);
            AscendC::MicroAPI::Duplicate(weight, weights[15U]);
            AscendC::MicroAPI::Mul(sum, sum, weight, mask);
#pragma unroll 1
            for (uint32_t offset = 1U; offset <= 15U; ++offset) {
                AscendC::MicroAPI::LoadAlign<float>(
                    upper, input + (centerRow - offset) * K31_COLUMN_C16_ROW_ELEMENTS + vectorOffset);
                AscendC::MicroAPI::LoadAlign<float>(
                    lower, input + (centerRow + offset) * K31_COLUMN_C16_ROW_ELEMENTS + vectorOffset);
                AscendC::MicroAPI::Add(pair, upper, lower, mask);
                AscendC::MicroAPI::Duplicate(weight, weights[15U - offset]);
                AscendC::MicroAPI::MulAddDst(sum, pair, weight, mask);
            }
            AscendC::MicroAPI::StoreAlign<float>(
                input + outputRow * K31_COLUMN_C16_ROW_ELEMENTS + vectorOffset, sum, mask);
        }
    }
}

__aicore__ inline void StoreColumnK31C16ToHwc(
    AscendC::GlobalTensor<float>& dstGlobal,
    AscendC::LocalTensor<float>& local,
    uint32_t outputRows,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t tileBaseX,
    uint32_t tileBaseY,
    uint32_t activeWidth,
    uint32_t outputChannels)
{
    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(activeWidth),
        static_cast<uint32_t>(outputChannels * sizeof(float)), 0,
        static_cast<int64_t>(channels - outputChannels) * static_cast<int64_t>(sizeof(float)), 0U};
    for (uint32_t row = 0U; row < outputRows; ++row) {
        const uint64_t destinationOffset = ElementOffsetAicore(
            tileBaseY + row, tileBaseX, width, channels, channelOffset);
        AscendC::DataCopyPad(
            dstGlobal[destinationOffset], local[row * K31_COLUMN_C16_ROW_ELEMENTS], params);
    }
}

__aicore__ inline void ProcessColumnK31C16GroupMajor(
    GM_ADDR src,
    GM_ADDR dst,
    const GaussianBlurTilingData* tilingData,
    __ubuf__ float* weights,
    AscendC::LocalTensor<float>& input)
{
    using namespace AscendC;
    GlobalTensor<float> srcGlobal;
    GlobalTensor<float> dstGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst));
    const uint32_t spatialTiles = tilingData->tilesX * tilingData->tilesY;
    const uint32_t channelGroups =
        (tilingData->c + K31_COLUMN_C16_CHANNELS - 1U) / K31_COLUMN_C16_CHANNELS;
    const uint32_t totalTiles = spatialTiles * channelGroups;
    TPipe pipe;
    const int32_t eventMte2ToV = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    const int32_t eventVToMte3 = static_cast<int32_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    const int32_t eventMte3ToMte2 = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    for (uint32_t tileId = GetBlockIdx(); tileId < totalTiles; tileId += GetBlockNum()) {
        const uint32_t spatialTile = tileId % spatialTiles;
        const uint32_t tileX = spatialTile % tilingData->tilesX;
        const uint32_t tileY = spatialTile / tilingData->tilesX;
        const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_COLUMN_TILE_W;
        const uint32_t tileBaseY = tileY * COLUMN_TILE_H;
        const uint32_t channelOffset = (tileId / spatialTiles) * K31_COLUMN_C16_CHANNELS;
        const uint32_t outputChannels = channelOffset + K31_COLUMN_C16_CHANNELS <= tilingData->c ?
            K31_COLUMN_C16_CHANNELS : tilingData->c - channelOffset;
        const uint32_t activeWidth = tileBaseX + GAUSSIAN_BLUR_COLUMN_TILE_W <= tilingData->w ?
            GAUSSIAN_BLUR_COLUMN_TILE_W : tilingData->w - tileBaseX;
        const uint32_t outputRows = tileBaseY + COLUMN_TILE_H <= tilingData->h ?
            COLUMN_TILE_H : tilingData->h - tileBaseY;
        const uint32_t inputRows = outputRows + 30U;
        uint32_t localRow = 0U;
        while (localRow < inputRows) {
            const uint32_t sourceRow = Reflect101ColumnCoord(
                static_cast<int32_t>(tileBaseY + localRow) - 15, tilingData->h);
            uint32_t rowCount = 1U;
            if (outputChannels == K31_COLUMN_C16_CHANNELS) {
                while (localRow + rowCount < inputRows) {
                    const uint32_t nextSourceRow = Reflect101ColumnCoord(
                        static_cast<int32_t>(tileBaseY + localRow + rowCount) - 15, tilingData->h);
                    if (nextSourceRow != sourceRow + rowCount) break;
                    ++rowCount;
                }
            }
            CopyColumnK31C16RowRangeToUb(
                srcGlobal, input, localRow, sourceRow, rowCount, tileBaseX, activeWidth,
                tilingData->h, tilingData->w, channelOffset, outputChannels);
            localRow += rowCount;
        }
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        asc_vf_call<GaussianBlurColumnK31C16SimdVF>(
            reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()), weights, outputRows);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        StoreColumnK31C16ToHwc(
            dstGlobal, input, outputRows, tilingData->w, tilingData->c, channelOffset,
            tileBaseX, tileBaseY, activeWidth, outputChannels);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    }
}
#endif

template <uint32_t ChannelStride, bool DynamicChannels>
__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurRowEdgeKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSize,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __gm__ const float* src,
    __gm__ float* dst)
{
    __ubuf__ float shared[
        GAUSSIAN_BLUR_ROW_TILE_H * ROW_SHARED_W * ChannelStride];
#define EDGE_ARGS height, width, channels, totalTiles, coreIndex, coreCount, tilesX, tilesY, kernelSize, borderType, \
    weights, src, dst, shared
#if GAUSSIAN_BLUR_ENABLE_ROW_HOT_K_SPECIALIZATION
    if (kernelSize == 3U) {
        RunRowTiles<ChannelStride, DynamicChannels, 3U, true>(EDGE_ARGS);
    } else if (kernelSize == 5U) {
        RunRowTiles<ChannelStride, DynamicChannels, 5U, true>(EDGE_ARGS);
    } else if (kernelSize == 11U) {
        RunRowTiles<ChannelStride, DynamicChannels, 11U, true>(EDGE_ARGS);
    } else if (kernelSize == 21U) {
        RunRowTiles<ChannelStride, DynamicChannels, 21U, true>(EDGE_ARGS);
    } else {
        RunRowTiles<ChannelStride, DynamicChannels, 0U, true>(EDGE_ARGS);
    }
#else
    if (kernelSize == 1U) {
        RunRowTiles<ChannelStride, DynamicChannels, 1U, true>(EDGE_ARGS);
    } else if (kernelSize == 3U) {
        RunRowTiles<ChannelStride, DynamicChannels, 3U, true>(EDGE_ARGS);
    } else if (kernelSize == 5U) {
        RunRowTiles<ChannelStride, DynamicChannels, 5U, true>(EDGE_ARGS);
    } else if (kernelSize == 7U) {
        RunRowTiles<ChannelStride, DynamicChannels, 7U, true>(EDGE_ARGS);
    } else if (kernelSize == 9U) {
        RunRowTiles<ChannelStride, DynamicChannels, 9U, true>(EDGE_ARGS);
    } else if (kernelSize == 11U) {
        RunRowTiles<ChannelStride, DynamicChannels, 11U, true>(EDGE_ARGS);
    } else if (kernelSize == 15U) {
        RunRowTiles<ChannelStride, DynamicChannels, 15U, true>(EDGE_ARGS);
    } else if (kernelSize == 21U) {
        RunRowTiles<ChannelStride, DynamicChannels, 21U, true>(EDGE_ARGS);
    } else if (kernelSize == 31U) {
        RunRowTiles<ChannelStride, DynamicChannels, 0U, true>(EDGE_ARGS);
    } else {
        RunRowTiles<ChannelStride, DynamicChannels, 0U, true>(EDGE_ARGS);
    }
#endif
#undef EDGE_ARGS
}

#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
// Wide interior tiles amortize setup cost, but the same width at the right
// boundary makes the VF private UB array exceed its practical limit.
// Process one 32-pixel patch at a time and reuse a small 64-pixel K31 window.
__simt_vf__ __aicore__ __launch_bounds__(512) inline void GaussianBlurRowC8K31SegmentedEdgeKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __gm__ const float* src,
    __gm__ float* dst)
{
    constexpr uint32_t edgeSharedW = 64U;
    constexpr uint32_t anchor = 15U;
    __ubuf__ float shared[GAUSSIAN_BLUR_ROW_TILE_H * edgeSharedW *
                           GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP];
    auto* shared4 = reinterpret_cast<__ubuf__ float4*>(shared);
    const uint32_t spatialTiles = tilesX * tilesY;

    for (uint32_t tileId = coreIndex; tileId < totalTiles; tileId += coreCount) {
        const uint32_t spatialTileId = tileId % spatialTiles;
        const uint32_t channelOffset =
            (tileId / spatialTiles) * GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP;
        const uint32_t outputChannels =
            channelOffset + GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP <= channels ?
                GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP : channels - channelOffset;
        const uint32_t tileX = spatialTileId % tilesX;
        const uint32_t tileY = spatialTileId / tilesX;
        const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_ROW_TILE_W;
        const uint32_t tileBaseY = tileY * GAUSSIAN_BLUR_ROW_TILE_H;
        const bool fullInterior = tileBaseX >= GAUSSIAN_BLUR_ROW_BLOCK_X &&
            tileBaseX + ROW_SHARED_W - GAUSSIAN_BLUR_ROW_BLOCK_X <= width &&
            tileBaseY + GAUSSIAN_BLUR_ROW_TILE_H <= height;
        if (fullInterior) {
            continue;
        }

        for (uint32_t patch = 0U; patch < GAUSSIAN_BLUR_ROW_PATCHES; ++patch) {
            const uint32_t patchBaseX = tileBaseX + patch * GAUSSIAN_BLUR_ROW_BLOCK_X;
#pragma unroll
            for (uint32_t rowIteration = 0U; rowIteration < 2U; ++rowIteration) {
                const uint32_t localRow = threadIdx.y + rowIteration * 4U;
                const uint32_t outputY = tileBaseY + localRow;
                if (threadIdx.z == 0U) {
#pragma unroll
                    for (uint32_t loadHalf = 0U; loadHalf < 2U; ++loadHalf) {
                        const uint32_t sharedX = threadIdx.x + loadHalf * 32U;
                        const int32_t rawX = static_cast<int32_t>(patchBaseX) - 16 +
                            static_cast<int32_t>(sharedX);
                        const int32_t sourceX = outputY < height ?
                            BorderCoord(rawX, static_cast<int32_t>(width), borderType) : -1;
                        const uint32_t sharedBase =
                            (localRow * edgeSharedW + sharedX) * GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP;
                        if (sourceX < 0) {
#pragma unroll
                            for (uint32_t channel = 0U;
                                 channel < GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP; ++channel) {
                                shared[sharedBase + channel] = 0.0f;
                            }
                        } else {
                            const uint64_t sourceBase = ElementOffset(
                                outputY, static_cast<uint32_t>(sourceX), width, channels, channelOffset);
#pragma unroll
                            for (uint32_t channel = 0U;
                                 channel < GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP; ++channel) {
                                shared[sharedBase + channel] = channel < outputChannels ?
                                    src[sourceBase + channel] : 0.0f;
                            }
                        }
                    }
                }
            }
            asc_syncthreads();

#pragma unroll
            for (uint32_t rowIteration = 0U; rowIteration < 2U; ++rowIteration) {
                const uint32_t localRow = threadIdx.y + rowIteration * 4U;
                const uint32_t outputY = tileBaseY + localRow;
                const uint32_t outputX = patchBaseX + threadIdx.x;
                const uint32_t subgroupOffset = threadIdx.z * GAUSSIAN_BLUR_CHANNEL_TILE;
                if (outputY < height && outputX < width && subgroupOffset < outputChannels) {
                    const uint32_t sharedCenter = threadIdx.x + 16U;
                    const uint32_t centerBase =
                        (localRow * edgeSharedW + sharedCenter) * 2U + threadIdx.z;
                    const float4 center = shared4[centerBase];
                    float4 sum = make_float4(
                        center.x * weights[anchor], center.y * weights[anchor],
                        center.z * weights[anchor], center.w * weights[anchor]);
#pragma unroll
                    for (uint32_t offset = 1U; offset <= anchor; ++offset) {
                        const float4 left = shared4[centerBase - offset * 2U];
                        const float4 right = shared4[centerBase + offset * 2U];
                        const float weight = weights[anchor - offset];
                        sum.x += (left.x + right.x) * weight;
                        sum.y += (left.y + right.y) * weight;
                        sum.z += (left.z + right.z) * weight;
                        sum.w += (left.w + right.w) * weight;
                    }
                    const uint32_t active = outputChannels - subgroupOffset <
                            GAUSSIAN_BLUR_CHANNEL_TILE ?
                        outputChannels - subgroupOffset : GAUSSIAN_BLUR_CHANNEL_TILE;
                    const uint64_t outputBase = ChunkMajorOffset(
                        outputY, outputX, height, width,
                        channelOffset + subgroupOffset, active);
                    dst[outputBase] = sum.x;
                    if (active >= 2U) dst[outputBase + 1U] = sum.y;
                    if (active >= 3U) dst[outputBase + 2U] = sum.z;
                    if (active >= 4U) dst[outputBase + 3U] = sum.w;
                }
            }
            asc_syncthreads();
        }
    }
}

// The W288 K31 edge window also exceeds the practical VF stack limit on the
// fixed C4 path. Keep the wide interior pipeline, but process each 32-pixel
// edge patch through a compact 64-pixel window in normal HWC C4 layout.
__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurRowC4K31SegmentedEdgeKernel(
    uint32_t height,
    uint32_t width,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t borderType,
    __ubuf__ const float* weights,
    __gm__ const float* src,
    __gm__ float* dst)
{
    constexpr uint32_t edgeSharedW = 64U;
    constexpr uint32_t anchor = 15U;
    constexpr uint32_t channels = 4U;
    constexpr uint32_t edgeRows = GAUSSIAN_BLUR_ROW_TILE_H / 2U;
    __ubuf__ float shared[edgeRows * edgeSharedW * channels];
    auto* shared4 = reinterpret_cast<__ubuf__ float4*>(shared);

    for (uint32_t tileId = coreIndex; tileId < totalTiles; tileId += coreCount) {
        const uint32_t tileX = tileId % tilesX;
        const uint32_t tileY = tileId / tilesX;
        const uint32_t tileBaseX = tileX * GAUSSIAN_BLUR_ROW_TILE_W;
        const uint32_t tileBaseY = tileY * GAUSSIAN_BLUR_ROW_TILE_H;
        const bool fullInterior = tileBaseX >= GAUSSIAN_BLUR_ROW_BLOCK_X &&
            tileBaseX + ROW_SHARED_W - GAUSSIAN_BLUR_ROW_BLOCK_X <= width &&
            tileBaseY + GAUSSIAN_BLUR_ROW_TILE_H <= height;
        if (fullInterior) {
            continue;
        }

        for (uint32_t patch = 0U; patch < GAUSSIAN_BLUR_ROW_PATCHES; ++patch) {
            const uint32_t patchBaseX = tileBaseX + patch * GAUSSIAN_BLUR_ROW_BLOCK_X;
#pragma unroll
            for (uint32_t rowIteration = 0U; rowIteration < 2U; ++rowIteration) {
                const uint32_t localRow = threadIdx.y;
                const uint32_t outputY = tileBaseY + localRow + rowIteration * edgeRows;
#pragma unroll
                for (uint32_t loadHalf = 0U; loadHalf < 2U; ++loadHalf) {
                    const uint32_t sharedX = threadIdx.x + loadHalf * GAUSSIAN_BLUR_ROW_BLOCK_X;
                    const int32_t rawX = static_cast<int32_t>(patchBaseX) - 16 +
                        static_cast<int32_t>(sharedX);
                    const int32_t sourceX = outputY < height ?
                        BorderCoord(rawX, static_cast<int32_t>(width), borderType) : -1;
                    const uint32_t sharedIndex = localRow * edgeSharedW + sharedX;
                    if (sourceX < 0) {
                        shared4[sharedIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    } else {
                        const uint64_t sourceBase = ElementOffset(
                            outputY, static_cast<uint32_t>(sourceX), width, channels, 0U);
                        shared4[sharedIndex] = *reinterpret_cast<__gm__ const float4*>(src + sourceBase);
                    }
                }
                asc_syncthreads();

                const uint32_t outputX = patchBaseX + threadIdx.x;
                if (outputY < height && outputX < width) {
                    const uint32_t centerIndex = localRow * edgeSharedW + threadIdx.x + 16U;
                    const float4 center = shared4[centerIndex];
                    float4 sum = make_float4(
                        center.x * weights[anchor], center.y * weights[anchor],
                        center.z * weights[anchor], center.w * weights[anchor]);
                    for (uint32_t offset = 1U; offset <= anchor; ++offset) {
                        const float4 left = shared4[centerIndex - offset];
                        const float4 right = shared4[centerIndex + offset];
                        const float weight = weights[anchor - offset];
                        sum.x += (left.x + right.x) * weight;
                        sum.y += (left.y + right.y) * weight;
                        sum.z += (left.z + right.z) * weight;
                        sum.w += (left.w + right.w) * weight;
                    }
                    const uint64_t outputBase = ElementOffset(outputY, outputX, width, channels, 0U);
                    *reinterpret_cast<__gm__ float4*>(dst + outputBase) = sum;
                }
                asc_syncthreads();
            }
        }
    }
}
#endif

template <uint32_t ChannelStride, bool DynamicChannels>
__aicore__ inline void ProcessRowChannels(
    GM_ADDR src,
    GM_ADDR dst,
    const GaussianBlurTilingData* tilingData)
{
    ProcessRowInteriorPipeline<ChannelStride, DynamicChannels>(src, dst, tilingData);
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
    if constexpr (ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP && DynamicChannels) {
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        AscendC::LocalTensor<float> weightTensor = ubAllocator.Alloc<float>(WEIGHT_UB_ELEMENTS);
        __ubuf__ float* weights = reinterpret_cast<__ubuf__ float*>(weightTensor.GetPhyAddr());
#pragma unroll
        for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
            weights[index] = tilingData->weights[index];
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
        asc_vf_call<GaussianBlurRowC8K31SegmentedEdgeKernel>(
            dim3{GAUSSIAN_BLUR_ROW_BLOCK_X, GAUSSIAN_BLUR_ROW_BLOCK_Y / 2U, 2U},
            tilingData->h, tilingData->w, tilingData->c, tilingData->totalTiles,
            AscendC::GetBlockIdx(), AscendC::GetBlockNum(), tilingData->tilesX,
            tilingData->tilesY, tilingData->borderType, weights,
            reinterpret_cast<__gm__ const float*>(src), reinterpret_cast<__gm__ float*>(dst));
        return;
    }
    if constexpr (ChannelStride == GAUSSIAN_BLUR_CHANNEL_TILE && !DynamicChannels) {
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        AscendC::LocalTensor<float> weightTensor = ubAllocator.Alloc<float>(WEIGHT_UB_ELEMENTS);
        __ubuf__ float* weights = reinterpret_cast<__ubuf__ float*>(weightTensor.GetPhyAddr());
#pragma unroll
        for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
            weights[index] = tilingData->weights[index];
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
        asc_vf_call<GaussianBlurRowC4K31SegmentedEdgeKernel>(
            dim3{GAUSSIAN_BLUR_ROW_BLOCK_X, GAUSSIAN_BLUR_ROW_BLOCK_Y / 2U, 1U},
            tilingData->h, tilingData->w, tilingData->totalTiles,
            AscendC::GetBlockIdx(), AscendC::GetBlockNum(), tilingData->tilesX,
            tilingData->tilesY, tilingData->borderType, weights,
            reinterpret_cast<__gm__ const float*>(src), reinterpret_cast<__gm__ float*>(dst));
        return;
    }
#endif
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    AscendC::LocalTensor<float> weightTensor = ubAllocator.Alloc<float>(WEIGHT_UB_ELEMENTS);
    __ubuf__ float* weights = reinterpret_cast<__ubuf__ float*>(weightTensor.GetPhyAddr());
#pragma unroll
    for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
        weights[index] = tilingData->weights[index];
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
    const uint32_t coreIndex = AscendC::GetBlockIdx();
    const uint32_t coreCount = AscendC::GetBlockNum();
    asc_vf_call<GaussianBlurRowEdgeKernel<ChannelStride, DynamicChannels>>(
        dim3{GAUSSIAN_BLUR_ROW_BLOCK_X,
             ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP && DynamicChannels ?
                 GAUSSIAN_BLUR_ROW_BLOCK_Y / 2U : GAUSSIAN_BLUR_ROW_BLOCK_Y,
             ChannelStride >= GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP && DynamicChannels ? 2U : 1U},
        tilingData->h, tilingData->w, tilingData->c, tilingData->totalTiles, coreIndex, coreCount,
        tilingData->tilesX, tilingData->tilesY, tilingData->kernelSize, tilingData->borderType, weights,
        reinterpret_cast<__gm__ const float*>(src), reinterpret_cast<__gm__ float*>(dst));
}

__aicore__ inline void ProcessRowGenericChannels(
    GM_ADDR src,
    GM_ADDR dst,
    const GaussianBlurTilingData* tilingData)
{
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
    ProcessRowK31C16GroupMajorAllTiles(src, dst, tilingData);
    return;
#else
    ProcessRowInteriorPipeline<GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP, true>(src, dst, tilingData);
#endif

    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    AscendC::LocalTensor<float> weightTensor = ubAllocator.Alloc<float>(WEIGHT_UB_ELEMENTS);
    __ubuf__ float* weights = reinterpret_cast<__ubuf__ float*>(weightTensor.GetPhyAddr());
#pragma unroll
    for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
        weights[index] = tilingData->weights[index];
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
    asc_vf_call<GaussianBlurRowEdgeKernel<GAUSSIAN_BLUR_CHANNEL_TILE, true>>(
        dim3{GAUSSIAN_BLUR_ROW_BLOCK_X, GAUSSIAN_BLUR_ROW_BLOCK_Y, 1U},
        tilingData->h, tilingData->w, tilingData->c, tilingData->totalTiles,
        AscendC::GetBlockIdx(), AscendC::GetBlockNum(), tilingData->tilesX, tilingData->tilesY,
        tilingData->kernelSize, tilingData->borderType, weights,
        reinterpret_cast<__gm__ const float*>(src), reinterpret_cast<__gm__ float*>(dst));
}

#ifndef GAUSSIAN_BLUR_COLUMN_ONLY
__aicore__ inline void ProcessRow(
    GM_ADDR src, GM_ADDR dst, const GaussianBlurTilingData* tilingData)
{
    if (tilingData->pathMode == GAUSSIAN_BLUR_PATH_C1_FAST) {
        ProcessRowChannels<1U, false>(src, dst, tilingData);
    } else if (tilingData->pathMode == GAUSSIAN_BLUR_PATH_C3_FAST) {
        ProcessRowChannels<3U, false>(src, dst, tilingData);
    } else if (tilingData->pathMode == GAUSSIAN_BLUR_PATH_C4_FAST) {
        ProcessRowChannels<4U, false>(src, dst, tilingData);
    } else if (tilingData->pathMode == GAUSSIAN_BLUR_PATH_GENERIC_C8) {
        ProcessRowChannels<GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP, true>(src, dst, tilingData);
    } else {
        ProcessRowGenericChannels(src, dst, tilingData);
    }
}
#endif


#ifndef GAUSSIAN_BLUR_ROW_ONLY
__aicore__ inline void ProcessColumn(
    GM_ADDR src, GM_ADDR dst, const GaussianBlurTilingData* tilingData)
{
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    AscendC::LocalTensor<float> weightTensor =
        ubAllocator.Alloc<float>(WEIGHT_UB_ELEMENTS);
    const uint32_t sharedElements =
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
        tilingData->pathMode == GAUSSIAN_BLUR_PATH_GENERIC_C && tilingData->kernelSize == 31U ?
            COLUMN_SHARED_H * GAUSSIAN_BLUR_COLUMN_BLOCK_X * K31_COLUMN_C16_CHANNELS :
#endif
        (tilingData->pathMode == GAUSSIAN_BLUR_PATH_GENERIC_C8 ?
            COLUMN_SHARED_H * GAUSSIAN_BLUR_COLUMN_BLOCK_X * GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP :
            COLUMN_SHARED_ELEMENTS);
    AscendC::LocalTensor<float> sharedTensor = ubAllocator.Alloc<float>(sharedElements);
    __ubuf__ float* weights = reinterpret_cast<__ubuf__ float*>(weightTensor.GetPhyAddr());
    __ubuf__ float* shared = reinterpret_cast<__ubuf__ float*>(sharedTensor.GetPhyAddr());
#pragma unroll
    for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
        weights[index] = tilingData->weights[index];
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
    const uint32_t coreIndex = AscendC::GetBlockIdx();
    const uint32_t coreCount = AscendC::GetBlockNum();
    if (tilingData->pathMode == GAUSSIAN_BLUR_PATH_GENERIC_C8) {
        ProcessColumnC8InteriorMte(src, dst, tilingData, weights, sharedTensor);
        asc_vf_call<GaussianBlurColumnC8EdgeKernel>(
            dim3{GAUSSIAN_BLUR_COLUMN_BLOCK_X, GAUSSIAN_BLUR_COLUMN_BLOCK_Y, 1U},
            tilingData->h, tilingData->w, tilingData->c, tilingData->totalTiles, coreIndex, coreCount,
            tilingData->tilesX, tilingData->tilesY, tilingData->kernelSize, tilingData->borderType,
            weights, shared, reinterpret_cast<__gm__ const float*>(src), reinterpret_cast<__gm__ float*>(dst));
    } else if (tilingData->pathMode == GAUSSIAN_BLUR_PATH_GENERIC_C && tilingData->kernelSize == 21U) {
        ProcessColumnInteriorPipeline(
            src, dst, tilingData, weights, sharedTensor, sharedTensor);
        asc_vf_call<GaussianBlurColumnEdgeKernel>(
            dim3{GAUSSIAN_BLUR_COLUMN_BLOCK_X, GAUSSIAN_BLUR_COLUMN_BLOCK_Y, 1U},
            tilingData->h, tilingData->w, tilingData->c, tilingData->totalTiles, coreIndex, coreCount,
            tilingData->tilesX, tilingData->tilesY, tilingData->kernelSize, tilingData->borderType,
            weights, shared, reinterpret_cast<__gm__ const float*>(src),
            reinterpret_cast<__gm__ float*>(dst));
    } else if (tilingData->pathMode == GAUSSIAN_BLUR_PATH_GENERIC_C && tilingData->kernelSize == 31U) {
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
        ProcessColumnK31C16GroupMajor(src, dst, tilingData, weights, sharedTensor);
#else
        asc_vf_call<GaussianBlurColumnGenericK31Kernel>(
            dim3{GAUSSIAN_BLUR_COLUMN_BLOCK_X, GAUSSIAN_BLUR_COLUMN_BLOCK_Y, 1U},
            tilingData->h, tilingData->w, tilingData->c, tilingData->totalTiles, coreIndex, coreCount,
            tilingData->tilesX, tilingData->tilesY, tilingData->borderType, weights, shared,
            reinterpret_cast<__gm__ const float*>(src), reinterpret_cast<__gm__ float*>(dst));
#endif
    } else {
        asc_vf_call<GaussianBlurColumnKernel>(
            dim3{GAUSSIAN_BLUR_COLUMN_BLOCK_X, GAUSSIAN_BLUR_COLUMN_BLOCK_Y, 1U},
            tilingData->h, tilingData->w, tilingData->c, tilingData->totalTiles, coreIndex, coreCount,
            tilingData->tilesX, tilingData->tilesY, tilingData->kernelSize, tilingData->borderType,
            tilingData->pathMode, weights, shared,
            reinterpret_cast<__gm__ const float*>(src), reinterpret_cast<__gm__ float*>(dst));
    }
}
#endif

#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
static constexpr uint32_t K31_RING_TILE_W = 64U;
static constexpr uint32_t K31_RING_CHANNELS = 8U;
static constexpr uint32_t K31_RING_ROWS = 31U;
static constexpr uint32_t K31_RING_RADIUS = 15U;
static constexpr uint32_t K31_PATCH_W = K31_RING_TILE_W + 2U * K31_RING_RADIUS;

__simt_callee__ inline float ComputeFusedK31HorizontalRuntime(
    int32_t sourceY,
    uint32_t outputX,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t channel,
    uint32_t borderType,
    __ubuf__ const float* weightsX,
    __gm__ const float* src)
{
    if (sourceY < 0) {
        return 0.0f;
    }

    const uint32_t sourceYU32 = static_cast<uint32_t>(sourceY);
    const uint64_t centerOffset =
        ElementOffset(sourceYU32, outputX, width, channels, channelOffset) + channel;
    float sum = src[centerOffset] * weightsX[K31_RING_RADIUS];
#pragma unroll 1
    for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
        const int32_t leftX = BorderCoord(
            static_cast<int32_t>(outputX) - static_cast<int32_t>(offset),
            static_cast<int32_t>(width), borderType);
        const int32_t rightX = BorderCoord(
            static_cast<int32_t>(outputX) + static_cast<int32_t>(offset),
            static_cast<int32_t>(width), borderType);
        float pair = 0.0f;
        if (leftX >= 0) {
            pair += src[ElementOffset(
                sourceYU32, static_cast<uint32_t>(leftX), width, channels, channelOffset) + channel];
        }
        if (rightX >= 0) {
            pair += src[ElementOffset(
                sourceYU32, static_cast<uint32_t>(rightX), width, channels, channelOffset) + channel];
        }
        sum += pair * weightsX[K31_RING_RADIUS - offset];
    }
    return sum;
}



__simt_callee__ inline void LoadFusedK31PatchRowFloat4(
    int32_t sourceY,
    int32_t tileBaseX,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t channelGroup,
    uint32_t borderType,
    __ubuf__ float* patch,
    __gm__ const float* src)
{
    const uint32_t subgroupOffset = channelGroup * 4U;
    const uint32_t channelBase = channelOffset + subgroupOffset;
    auto* patch4 = reinterpret_cast<__ubuf__ float4*>(patch);
#pragma unroll 1
    for (uint32_t patchX = threadIdx.x; patchX < K31_PATCH_W; patchX += K31_RING_TILE_W) {
        const int32_t sourceX = BorderCoord(
            tileBaseX + static_cast<int32_t>(patchX), static_cast<int32_t>(width), borderType);
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (sourceY >= 0 && sourceX >= 0 && channelBase < channels) {
            const uint64_t base = ElementOffset(
                static_cast<uint32_t>(sourceY), static_cast<uint32_t>(sourceX), width,
                channels, channelBase);
            value.x = src[base];
            if (channelBase + 1U < channels) value.y = src[base + 1U];
            if (channelBase + 2U < channels) value.z = src[base + 2U];
            if (channelBase + 3U < channels) value.w = src[base + 3U];
        }
        patch4[patchX * 2U + channelGroup] = value;
    }
}

__simt_callee__ inline float4 ComputeFusedK31HorizontalFromPatchFloat4(
    uint32_t localX,
    uint32_t channelGroup,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* patch)
{
    const auto* patch4 = reinterpret_cast<__ubuf__ const float4*>(patch);
    const uint32_t centerX = localX + K31_RING_RADIUS;
    float4 sum = patch4[centerX * 2U + channelGroup] * weightsX[K31_RING_RADIUS];
#pragma unroll 1
    for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
        const float4 pair = patch4[(centerX - offset) * 2U + channelGroup] +
            patch4[(centerX + offset) * 2U + channelGroup];
        sum += pair * weightsX[K31_RING_RADIUS - offset];
    }
    return sum;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurFusedK31RingKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t tilesX,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t borderType,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* weightsY,
    __ubuf__ float* ring,
    __ubuf__ float* patch,
    __gm__ const float* src,
    __gm__ float* dst)
{
    const uint32_t localX = threadIdx.x;
    const uint32_t channelGroup = threadIdx.y;
    const uint32_t subgroupOffset = channelGroup * 4U;
    constexpr uint32_t rowStride = K31_RING_TILE_W * K31_RING_CHANNELS;
    auto* ring4 = reinterpret_cast<__ubuf__ float4*>(ring);

#if GAUSSIAN_BLUR_K31_RING_DIAGNOSTIC_STAGE == 4
    if (threadIdx.x == 0U && threadIdx.y == 0U && coreIndex == 0U) {
#pragma unroll 1
        for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
            dst[index] = weightsX[index];
            dst[32U + index] = weightsY[index];
        }
        dst[64U] = static_cast<float>(height);
        dst[65U] = static_cast<float>(width);
        dst[66U] = static_cast<float>(channels);
        dst[67U] = static_cast<float>(totalTiles);
        dst[68U] = static_cast<float>(tilesX);
        dst[69U] = static_cast<float>(borderType);
    }
    return;
#endif

#if GAUSSIAN_BLUR_K31_RING_DIAGNOSTIC_STAGE == 1
    const uint32_t markerOffset = localX * K31_RING_CHANNELS + channel;
    ring[markerOffset] = static_cast<float>(coreIndex + 1U);
    asc_syncthreads();
    if (localX == 0U && channel == 0U && coreIndex == 0U) {
        dst[0] = ring[0];
    }
    return;
#endif

    for (uint32_t tileId = coreIndex; tileId < totalTiles; tileId += coreCount) {
        const uint32_t tileX = tileId % tilesX;
        const uint32_t channelOffset = (tileId / tilesX) * K31_RING_CHANNELS;
        const uint32_t outputX = tileX * K31_RING_TILE_W + localX;
        const uint32_t channelBase = channelOffset + subgroupOffset;
        const bool active = outputX < width && channelBase < channels;
        const int32_t tileBaseX =
            static_cast<int32_t>(tileX * K31_RING_TILE_W) - static_cast<int32_t>(K31_RING_RADIUS);

#pragma unroll 1
        for (uint32_t ringRow = 0U; ringRow < K31_RING_ROWS; ++ringRow) {
            const int32_t sourceY = BorderCoord(
                static_cast<int32_t>(ringRow) - static_cast<int32_t>(K31_RING_RADIUS),
                static_cast<int32_t>(height), borderType);
            LoadFusedK31PatchRowFloat4(
                sourceY, tileBaseX, width, channels, channelOffset, channelGroup,
                borderType, patch, src);
            asc_syncthreads();
            const float4 sum = active ? ComputeFusedK31HorizontalFromPatchFloat4(
                localX, channelGroup, weightsX, patch) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            ring4[(ringRow * K31_RING_TILE_W + localX) * 2U + channelGroup] = sum;
            asc_syncthreads();
        }

#if GAUSSIAN_BLUR_K31_RING_DIAGNOSTIC_STAGE == 2
        if (coreIndex == 0U && tileId == 0U && localX == 0U && channel == 0U) {
#pragma unroll 1
            for (uint32_t ringRow = 0U; ringRow < K31_RING_ROWS; ++ringRow) {
                const int32_t sourceY = BorderCoord(
                    static_cast<int32_t>(ringRow) - static_cast<int32_t>(K31_RING_RADIUS),
                    static_cast<int32_t>(height), borderType);
                dst[ringRow] = ring[ringRow * rowStride];
                dst[32U + ringRow] = static_cast<float>(sourceY);
                dst[64U + ringRow] = weightsX[ringRow];
                dst[128U + ringRow] = ComputeFusedK31HorizontalRuntime(
                    sourceY, 0U, width, channels, 0U, 0U, borderType, weightsX, src);
                dst[192U + ringRow] = sourceY >= 0 ?
                    src[ElementOffset(static_cast<uint32_t>(sourceY), 0U, width, channels, 0U)] : 0.0f;
                ring[ringRow * rowStride] = static_cast<float>(ringRow + 1U);
            }
            dst[96U] = static_cast<float>(height);
            dst[97U] = static_cast<float>(width);
            dst[98U] = static_cast<float>(channels);
            dst[99U] = static_cast<float>(borderType);
        }
        asc_syncthreads();
        if (coreIndex == 0U && tileId == 0U && localX == 0U && channel == 0U) {
#pragma unroll 1
            for (uint32_t ringRow = 0U; ringRow < K31_RING_ROWS; ++ringRow) {
                dst[160U + ringRow] = ring[ringRow * rowStride];
            }
        }
        asc_syncthreads();
        continue;
#endif

#if GAUSSIAN_BLUR_K31_RING_DIAGNOSTIC_STAGE == 3
        if (active) {
            constexpr uint32_t centerSlot = K31_RING_RADIUS;
            float4 sum = ring4[(centerSlot * K31_RING_TILE_W + localX) * 2U + channelGroup] *
                weightsY[K31_RING_RADIUS];
#pragma unroll 1
            for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
                const uint32_t upperSlot = centerSlot - offset;
                const uint32_t lowerSlot = centerSlot + offset;
                const float4 pair =
                    ring4[(upperSlot * K31_RING_TILE_W + localX) * 2U + channelGroup] +
                    ring4[(lowerSlot * K31_RING_TILE_W + localX) * 2U + channelGroup];
                sum += pair * weightsY[K31_RING_RADIUS - offset];
            }
            const uint64_t outputBase = outputX * channels + channelBase;
            dst[outputBase] = sum.x;
            if (channelBase + 1U < channels) dst[outputBase + 1U] = sum.y;
            if (channelBase + 2U < channels) dst[outputBase + 2U] = sum.z;
            if (channelBase + 3U < channels) dst[outputBase + 3U] = sum.w;
        }
        asc_syncthreads();
        continue;
#endif

        uint32_t centerSlot = K31_RING_RADIUS;
        uint32_t replaceSlot = 0U;
        for (uint32_t outputY = 0U; outputY < height; ++outputY) {
            if (active) {
                float4 sum = ring4[(centerSlot * K31_RING_TILE_W + localX) * 2U + channelGroup] *
                    weightsY[K31_RING_RADIUS];
#pragma unroll 1
                for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
                    const uint32_t upperSlot = centerSlot >= offset ?
                        centerSlot - offset : centerSlot + K31_RING_ROWS - offset;
                    const uint32_t lowerUnwrapped = centerSlot + offset;
                    const uint32_t lowerSlot = lowerUnwrapped < K31_RING_ROWS ?
                        lowerUnwrapped : lowerUnwrapped - K31_RING_ROWS;
                    const float4 pair =
                        ring4[(upperSlot * K31_RING_TILE_W + localX) * 2U + channelGroup] +
                        ring4[(lowerSlot * K31_RING_TILE_W + localX) * 2U + channelGroup];
                    sum += pair * weightsY[K31_RING_RADIUS - offset];
                }
                const uint64_t outputBase =
                    (static_cast<uint64_t>(outputY) * width + outputX) * channels + channelBase;
                dst[outputBase] = sum.x;
                if (channelBase + 1U < channels) dst[outputBase + 1U] = sum.y;
                if (channelBase + 2U < channels) dst[outputBase + 2U] = sum.z;
                if (channelBase + 3U < channels) dst[outputBase + 3U] = sum.w;
            }
            asc_syncthreads();

            if (outputY + 1U < height) {
                const int32_t sourceY = BorderCoord(
                    static_cast<int32_t>(outputY) + static_cast<int32_t>(K31_RING_RADIUS) + 1,
                    static_cast<int32_t>(height), borderType);
                LoadFusedK31PatchRowFloat4(
                    sourceY, tileBaseX, width, channels, channelOffset, channelGroup,
                    borderType, patch, src);
                asc_syncthreads();
                const float4 horizontal = active ? ComputeFusedK31HorizontalFromPatchFloat4(
                    localX, channelGroup, weightsX, patch) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                ring4[(replaceSlot * K31_RING_TILE_W + localX) * 2U + channelGroup] = horizontal;
            }
            asc_syncthreads();
            centerSlot = centerSlot + 1U < K31_RING_ROWS ? centerSlot + 1U : 0U;
            replaceSlot = replaceSlot + 1U < K31_RING_ROWS ? replaceSlot + 1U : 0U;
        }
    }
}

static constexpr uint32_t K31_CHANNEL_RING_TILE_W = 22U;
static constexpr uint32_t K31_CHANNEL_RING_CHANNELS = 64U;
static constexpr uint32_t K31_CHANNEL_PATCH_W =
    K31_CHANNEL_RING_TILE_W + 2U * K31_RING_RADIUS;
static constexpr uint32_t K31_CHANNEL_RING_ROW_ELEMENTS =
    K31_CHANNEL_RING_TILE_W * K31_CHANNEL_RING_CHANNELS;

__simt_vf__ __aicore__ __launch_bounds__(64) inline void GaussianBlurFusedK31ChannelRingKernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t tilesX,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t borderType,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* weightsY,
    __ubuf__ float* ring,
    __ubuf__ float* patch,
    __gm__ const float* src,
    __gm__ float* dst)
{
    const uint32_t localChannel = threadIdx.x;
    for (uint32_t tileId = coreIndex; tileId < totalTiles; tileId += coreCount) {
        const uint32_t tileX = tileId % tilesX;
        const uint32_t channelOffset =
            (tileId / tilesX) * K31_CHANNEL_RING_CHANNELS;
        const uint32_t channel = channelOffset + localChannel;
        const bool channelActive = channel < channels;
        const int32_t tileBaseX =
            static_cast<int32_t>(tileX * K31_CHANNEL_RING_TILE_W) -
            static_cast<int32_t>(K31_RING_RADIUS);

#pragma unroll 1
        for (uint32_t ringRow = 0U; ringRow < K31_RING_ROWS; ++ringRow) {
            const int32_t sourceY = BorderCoord(
                static_cast<int32_t>(ringRow) - static_cast<int32_t>(K31_RING_RADIUS),
                static_cast<int32_t>(height), borderType);
#pragma unroll 1
            for (uint32_t patchX = 0U; patchX < K31_CHANNEL_PATCH_W; ++patchX) {
                const int32_t sourceX = BorderCoord(
                    tileBaseX + static_cast<int32_t>(patchX),
                    static_cast<int32_t>(width), borderType);
                float value = 0.0f;
                if (channelActive && sourceY >= 0 && sourceX >= 0) {
                    value = src[ElementOffset(
                        static_cast<uint32_t>(sourceY), static_cast<uint32_t>(sourceX),
                        width, channels, channel)];
                }
                patch[patchX * K31_CHANNEL_RING_CHANNELS + localChannel] = value;
            }
            asc_syncthreads();

#pragma unroll 1
            for (uint32_t localX = 0U; localX < K31_CHANNEL_RING_TILE_W; ++localX) {
                const uint32_t centerX = localX + K31_RING_RADIUS;
                float sum = patch[centerX * K31_CHANNEL_RING_CHANNELS + localChannel] *
                    weightsX[K31_RING_RADIUS];
#pragma unroll 1
                for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
                    const float pair =
                        patch[(centerX - offset) * K31_CHANNEL_RING_CHANNELS + localChannel] +
                        patch[(centerX + offset) * K31_CHANNEL_RING_CHANNELS + localChannel];
                    sum += pair * weightsX[K31_RING_RADIUS - offset];
                }
                ring[ringRow * K31_CHANNEL_RING_ROW_ELEMENTS +
                     localX * K31_CHANNEL_RING_CHANNELS + localChannel] = sum;
            }
            asc_syncthreads();
        }

        uint32_t centerSlot = K31_RING_RADIUS;
        uint32_t replaceSlot = 0U;
        for (uint32_t outputY = 0U; outputY < height; ++outputY) {
#pragma unroll 1
            for (uint32_t localX = 0U; localX < K31_CHANNEL_RING_TILE_W; ++localX) {
                const uint32_t outputX = tileX * K31_CHANNEL_RING_TILE_W + localX;
                if (channelActive && outputX < width) {
                    float sum = ring[centerSlot * K31_CHANNEL_RING_ROW_ELEMENTS +
                        localX * K31_CHANNEL_RING_CHANNELS + localChannel] *
                        weightsY[K31_RING_RADIUS];
#pragma unroll 1
                    for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
                        const uint32_t upperSlot = centerSlot >= offset ?
                            centerSlot - offset : centerSlot + K31_RING_ROWS - offset;
                        const uint32_t lowerUnwrapped = centerSlot + offset;
                        const uint32_t lowerSlot = lowerUnwrapped < K31_RING_ROWS ?
                            lowerUnwrapped : lowerUnwrapped - K31_RING_ROWS;
                        const float pair =
                            ring[upperSlot * K31_CHANNEL_RING_ROW_ELEMENTS +
                                localX * K31_CHANNEL_RING_CHANNELS + localChannel] +
                            ring[lowerSlot * K31_CHANNEL_RING_ROW_ELEMENTS +
                                localX * K31_CHANNEL_RING_CHANNELS + localChannel];
                        sum += pair * weightsY[K31_RING_RADIUS - offset];
                    }
                    dst[ElementOffset(outputY, outputX, width, channels, channel)] = sum;
                }
            }
            asc_syncthreads();

            if (outputY + 1U < height) {
                const int32_t sourceY = BorderCoord(
                    static_cast<int32_t>(outputY) + static_cast<int32_t>(K31_RING_RADIUS) + 1,
                    static_cast<int32_t>(height), borderType);
#pragma unroll 1
                for (uint32_t patchX = 0U; patchX < K31_CHANNEL_PATCH_W; ++patchX) {
                    const int32_t sourceX = BorderCoord(
                        tileBaseX + static_cast<int32_t>(patchX),
                        static_cast<int32_t>(width), borderType);
                    float value = 0.0f;
                    if (channelActive && sourceY >= 0 && sourceX >= 0) {
                        value = src[ElementOffset(
                            static_cast<uint32_t>(sourceY), static_cast<uint32_t>(sourceX),
                            width, channels, channel)];
                    }
                    patch[patchX * K31_CHANNEL_RING_CHANNELS + localChannel] = value;
                }
                asc_syncthreads();

#pragma unroll 1
                for (uint32_t localX = 0U; localX < K31_CHANNEL_RING_TILE_W; ++localX) {
                    const uint32_t centerX = localX + K31_RING_RADIUS;
                    float sum = patch[centerX * K31_CHANNEL_RING_CHANNELS + localChannel] *
                        weightsX[K31_RING_RADIUS];
#pragma unroll 1
                    for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
                        const float pair =
                            patch[(centerX - offset) * K31_CHANNEL_RING_CHANNELS + localChannel] +
                            patch[(centerX + offset) * K31_CHANNEL_RING_CHANNELS + localChannel];
                        sum += pair * weightsX[K31_RING_RADIUS - offset];
                    }
                    ring[replaceSlot * K31_CHANNEL_RING_ROW_ELEMENTS +
                         localX * K31_CHANNEL_RING_CHANNELS + localChannel] = sum;
                }
            }
            asc_syncthreads();
            centerSlot = centerSlot + 1U < K31_RING_ROWS ? centerSlot + 1U : 0U;
            replaceSlot = replaceSlot + 1U < K31_RING_ROWS ? replaceSlot + 1U : 0U;
        }
    }
}

static constexpr uint32_t K31_BLOCK_CHANNELS = 2U;
static constexpr uint32_t K31_BLOCK_ROW_ELEMENTS = K31_RING_TILE_W * K31_BLOCK_CHANNELS;

__simt_vf__ __aicore__ __launch_bounds__(128) inline void GaussianBlurK31HorizontalBlockKernel(
    uint32_t sourceBaseY,
    uint32_t inputRows,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t tileX,
    uint32_t channelOffset,
    uint32_t borderType,
    __ubuf__ const float* weightsX,
    __ubuf__ float* horizontal,
    __ubuf__ float* patch,
    __gm__ const float* src)
{
    const uint32_t localX = threadIdx.x;
    const uint32_t outputX = tileX * K31_RING_TILE_W + localX;
    const bool active = outputX < width && channelOffset < channels;
    const int32_t tileBaseX =
        static_cast<int32_t>(tileX * K31_RING_TILE_W) - static_cast<int32_t>(K31_RING_RADIUS);
    for (uint32_t inputRow = 0U; inputRow < inputRows; ++inputRow) {
        const int32_t sourceY = BorderCoord(
            static_cast<int32_t>(sourceBaseY + inputRow) - static_cast<int32_t>(K31_RING_RADIUS),
            static_cast<int32_t>(height), borderType);
#pragma unroll 1
        for (uint32_t patchX = localX; patchX < K31_PATCH_W; patchX += K31_RING_TILE_W) {
            const int32_t sourceX = BorderCoord(
                tileBaseX + static_cast<int32_t>(patchX), static_cast<int32_t>(width), borderType);
            float value0 = 0.0f;
            float value1 = 0.0f;
            if (sourceY >= 0 && sourceX >= 0 && channelOffset < channels) {
                const uint64_t base = ElementOffset(
                    static_cast<uint32_t>(sourceY), static_cast<uint32_t>(sourceX), width,
                    channels, channelOffset);
                value0 = src[base];
                if (channelOffset + 1U < channels) value1 = src[base + 1U];
            }
            patch[patchX * K31_BLOCK_CHANNELS] = value0;
            patch[patchX * K31_BLOCK_CHANNELS + 1U] = value1;
        }
        asc_syncthreads();
        if (active) {
            const uint32_t centerX = localX + K31_RING_RADIUS;
            float sum0 = patch[centerX * K31_BLOCK_CHANNELS] * weightsX[K31_RING_RADIUS];
            float sum1 = patch[centerX * K31_BLOCK_CHANNELS + 1U] * weightsX[K31_RING_RADIUS];
#pragma unroll 1
            for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
                const uint32_t leftBase = (centerX - offset) * K31_BLOCK_CHANNELS;
                const uint32_t rightBase = (centerX + offset) * K31_BLOCK_CHANNELS;
                sum0 += (patch[leftBase] + patch[rightBase]) * weightsX[K31_RING_RADIUS - offset];
                sum1 += (patch[leftBase + 1U] + patch[rightBase + 1U]) *
                    weightsX[K31_RING_RADIUS - offset];
            }
            const uint32_t horizontalBase =
                (inputRow * K31_RING_TILE_W + localX) * K31_BLOCK_CHANNELS;
            horizontal[horizontalBase] = sum0;
            horizontal[horizontalBase + 1U] = sum1;
        }
        asc_syncthreads();
    }
}


__simt_vf__ __aicore__ __launch_bounds__(2048) inline void GaussianBlurK31VerticalBlockSimtVF(
    __ubuf__ const float* horizontal,
    __ubuf__ const float* weightsY,
    __ubuf__ float* output,
    uint32_t outputRows)
{
    const uint32_t totalPixels = outputRows * K31_RING_TILE_W;
    for (uint32_t pixel = threadIdx.x; pixel < totalPixels; pixel += blockDim.x) {
        const uint32_t outputRow = pixel / K31_RING_TILE_W;
        const uint32_t localX = pixel - outputRow * K31_RING_TILE_W;
        const uint32_t centerBase =
            ((outputRow + K31_RING_RADIUS) * K31_RING_TILE_W + localX) *
            K31_BLOCK_CHANNELS;
        float sum0 = horizontal[centerBase] * weightsY[K31_RING_RADIUS];
        float sum1 = horizontal[centerBase + 1U] * weightsY[K31_RING_RADIUS];
#pragma unroll 1
        for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
            const uint32_t elementOffset = offset * K31_BLOCK_ROW_ELEMENTS;
            sum0 += (horizontal[centerBase - elementOffset] +
                horizontal[centerBase + elementOffset]) * weightsY[K31_RING_RADIUS - offset];
            sum1 += (horizontal[centerBase - elementOffset + 1U] +
                horizontal[centerBase + elementOffset + 1U]) * weightsY[K31_RING_RADIUS - offset];
        }
        const uint32_t outputBase = pixel * K31_BLOCK_CHANNELS;
        output[outputBase] = sum0;
        output[outputBase + 1U] = sum1;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(128) inline void GaussianBlurK31StoreBlockKernel(
    uint32_t outputBaseY,
    uint32_t outputRows,
    uint32_t width,
    uint32_t channels,
    uint32_t tileX,
    uint32_t channelOffset,
    __ubuf__ const float* output,
    __gm__ float* dst)
{
    const uint32_t localX = threadIdx.x;
    const uint32_t outputX = tileX * K31_RING_TILE_W + localX;
    if (outputX >= width || channelOffset >= channels) {
        return;
    }
    for (uint32_t outputRow = 0U; outputRow < outputRows; ++outputRow) {
        const uint32_t localBase =
            (outputRow * K31_RING_TILE_W + localX) * K31_BLOCK_CHANNELS;
        const uint64_t outputBase =
            (static_cast<uint64_t>(outputBaseY + outputRow) * width + outputX) * channels + channelOffset;
        dst[outputBase] = output[localBase];
        if (channelOffset + 1U < channels) dst[outputBase + 1U] = output[localBase + 1U];
    }
}

static constexpr uint32_t K31_DUAL_SIMD_CHANNELS = 2U;
static constexpr uint32_t K31_DUAL_SIMD_RAW_ROW_ELEMENTS = 192U;

__simt_vf__ __aicore__ __launch_bounds__(128) inline void GaussianBlurK31GatherBlockKernel(
    uint32_t sourceBaseY,
    uint32_t inputRows,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t tileX,
    uint32_t channelOffset,
    uint32_t borderType,
    __ubuf__ float* raw,
    __gm__ const float* src)
{
    const uint32_t localX = threadIdx.x;
    const int32_t tileBaseX =
        static_cast<int32_t>(tileX * K31_RING_TILE_W) - static_cast<int32_t>(K31_RING_RADIUS);
    for (uint32_t inputRow = 0U; inputRow < inputRows; ++inputRow) {
        const int32_t sourceY = BorderCoord(
            static_cast<int32_t>(sourceBaseY + inputRow) - static_cast<int32_t>(K31_RING_RADIUS),
            static_cast<int32_t>(height), borderType);
#pragma unroll 1
        for (uint32_t patchX = localX; patchX < K31_PATCH_W; patchX += K31_RING_TILE_W) {
            const int32_t sourceX = BorderCoord(
                tileBaseX + static_cast<int32_t>(patchX), static_cast<int32_t>(width), borderType);
            float value0 = 0.0f;
            float value1 = 0.0f;
            if (sourceY >= 0 && sourceX >= 0 && channelOffset < channels) {
                const uint64_t sourceBase = ElementOffset(
                    static_cast<uint32_t>(sourceY), static_cast<uint32_t>(sourceX), width,
                    channels, channelOffset);
                value0 = src[sourceBase];
                if (channelOffset + 1U < channels) value1 = src[sourceBase + 1U];
            }
            const uint32_t rawBase =
                inputRow * K31_DUAL_SIMD_RAW_ROW_ELEMENTS + patchX * K31_DUAL_SIMD_CHANNELS;
            raw[rawBase] = value0;
            raw[rawBase + 1U] = value1;
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(128) inline void GaussianBlurK31GatherHaloKernel(
    uint32_t sourceBaseY,
    uint32_t inputRows,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t tileX,
    uint32_t channelOffset,
    uint32_t borderType,
    __ubuf__ float* raw,
    __gm__ const float* src,
    uint32_t haloBegin,
    uint32_t haloEnd)
{
    const uint32_t patchX = haloBegin + threadIdx.x;
    if (patchX >= haloEnd) {
        return;
    }
    const int32_t tileBaseX =
        static_cast<int32_t>(tileX * K31_RING_TILE_W) - static_cast<int32_t>(K31_RING_RADIUS);
    const int32_t sourceX = BorderCoord(
        tileBaseX + static_cast<int32_t>(patchX), static_cast<int32_t>(width), borderType);
    for (uint32_t inputRow = 0U; inputRow < inputRows; ++inputRow) {
        const int32_t sourceY = BorderCoord(
            static_cast<int32_t>(sourceBaseY + inputRow) - static_cast<int32_t>(K31_RING_RADIUS),
            static_cast<int32_t>(height), borderType);
        float value0 = 0.0f;
        float value1 = 0.0f;
        if (sourceY >= 0 && sourceX >= 0 && channelOffset < channels) {
            const uint64_t sourceBase = ElementOffset(
                static_cast<uint32_t>(sourceY), static_cast<uint32_t>(sourceX), width,
                channels, channelOffset);
            value0 = src[sourceBase];
            if (channelOffset + 1U < channels) value1 = src[sourceBase + 1U];
        }
        const uint32_t rawBase =
            inputRow * K31_DUAL_SIMD_RAW_ROW_ELEMENTS + patchX * K31_DUAL_SIMD_CHANNELS;
        raw[rawBase] = value0;
        raw[rawBase + 1U] = value1;
    }
}


__simt_vf__ __aicore__ __launch_bounds__(2048) inline void GaussianBlurK31HorizontalBlockSimtVF(
    __ubuf__ const float* raw,
    __ubuf__ const float* weightsX,
    __ubuf__ float* horizontal,
    uint32_t inputRows)
{
    const uint32_t totalPixels = inputRows * K31_RING_TILE_W;
    for (uint32_t pixel = threadIdx.x; pixel < totalPixels; pixel += blockDim.x) {
        const uint32_t inputRow = pixel / K31_RING_TILE_W;
        const uint32_t localX = pixel - inputRow * K31_RING_TILE_W;
        const uint32_t centerBase = inputRow * K31_DUAL_SIMD_RAW_ROW_ELEMENTS +
            (localX + K31_RING_RADIUS) * K31_DUAL_SIMD_CHANNELS;
        float sum0 = raw[centerBase] * weightsX[K31_RING_RADIUS];
        float sum1 = raw[centerBase + 1U] * weightsX[K31_RING_RADIUS];
#pragma unroll 1
        for (uint32_t offset = 1U; offset <= K31_RING_RADIUS; ++offset) {
            const uint32_t elementOffset = offset * K31_DUAL_SIMD_CHANNELS;
            sum0 += (raw[centerBase - elementOffset] + raw[centerBase + elementOffset]) *
                weightsX[K31_RING_RADIUS - offset];
            sum1 += (raw[centerBase - elementOffset + 1U] +
                raw[centerBase + elementOffset + 1U]) * weightsX[K31_RING_RADIUS - offset];
        }
        const uint32_t outputBase = pixel * K31_DUAL_SIMD_CHANNELS;
        horizontal[outputBase] = sum0;
        horizontal[outputBase + 1U] = sum1;
    }
}

static constexpr uint32_t K31_STREAM_TILE_W = 128U;
static constexpr uint32_t K31_STREAM_CHANNELS = 8U;
static constexpr uint32_t K31_STREAM_PATCH_W = K31_STREAM_TILE_W + 2U * K31_RING_RADIUS;
static constexpr uint32_t K31_STREAM_RAW_ROW_ELEMENTS = K31_STREAM_PATCH_W * K31_STREAM_CHANNELS;
static constexpr uint32_t K31_STREAM_RING_ROW_ELEMENTS = K31_STREAM_TILE_W * K31_STREAM_CHANNELS;
static constexpr uint32_t K31_STREAM_OUTPUT_BATCH_ROWS = 8U;
static constexpr uint32_t K31_STREAM_RAW_OFFSET = 0U;
static constexpr uint32_t K31_STREAM_RAW1_OFFSET = K31_STREAM_RAW_ROW_ELEMENTS;
static constexpr uint32_t K31_STREAM_PENDING_OFFSET =
    K31_STREAM_RAW1_OFFSET + K31_STREAM_RAW_ROW_ELEMENTS;
static constexpr uint32_t K31_STREAM_RING_OFFSET =
    K31_STREAM_PENDING_OFFSET + K31_STREAM_RING_ROW_ELEMENTS;
static constexpr uint32_t K31_STREAM_OUTPUT_OFFSET =
    K31_STREAM_RING_OFFSET + K31_RING_ROWS * K31_STREAM_RING_ROW_ELEMENTS;
static constexpr uint32_t K31_STREAM_WORKSPACE_ELEMENTS =
    K31_STREAM_OUTPUT_OFFSET +
    K31_STREAM_OUTPUT_BATCH_ROWS * K31_STREAM_RING_ROW_ELEMENTS;
static constexpr uint32_t K31_C1_RING_ROW_ELEMENTS = K31_STREAM_TILE_W;
static constexpr uint32_t K31_C1_RAW_ROW_STRIDE = 512U;
static constexpr uint32_t K31_C1_PENDING_OFFSET = 1024U;
static constexpr uint32_t K31_C1_K31_TILE_W = 128U;
static constexpr uint32_t K31_C1_OUTPUT_BATCH_ROWS =
    K31_STREAM_OUTPUT_BATCH_ROWS * K31_STREAM_CHANNELS;
static constexpr uint32_t K31_C1_SCRATCH_ELEMENTS =
    K31_RING_ROWS * K31_STREAM_RING_ROW_ELEMENTS;

__aicore__ inline uint32_t GaussianBlurC1TileWeight(
    uint32_t width,
    uint32_t kernelSizeX,
    uint32_t kernelSizeY,
    uint32_t tileWidth,
    uint32_t tileX)
{
    const uint32_t outputBaseX = tileX * tileWidth;
    const uint32_t activeWidth = outputBaseX + tileWidth <= width ?
        tileWidth : width - outputBaseX;
    const uint32_t minimumComputeWidth = kernelSizeX == 31U ? 1U : 32U;
    const uint32_t effectiveWidth = activeWidth < minimumComputeWidth ?
        minimumComputeWidth : activeWidth;
    const uint32_t radiusX = kernelSizeX / 2U;
    const uint32_t interiorBegin = outputBaseX < radiusX ?
        (radiusX - outputBaseX < activeWidth ? radiusX - outputBaseX : activeWidth) : 0U;
    const uint32_t interiorGlobalEnd = width > radiusX ? width - radiusX : 0U;
    const uint32_t interiorEnd = interiorGlobalEnd > outputBaseX ?
        (interiorGlobalEnd - outputBaseX < activeWidth ?
            interiorGlobalEnd - outputBaseX : activeWidth) : interiorBegin;
    const uint32_t interiorWidth = interiorEnd > interiorBegin ? interiorEnd - interiorBegin : 0U;
    const uint32_t boundaryPixels = activeWidth - interiorWidth;
    const uint32_t horizontalTaps = kernelSizeX / 2U + 1U;
    const uint32_t verticalTaps = kernelSizeY / 2U + 1U;
    const uint32_t boundaryExtraHorizontalPenalty = kernelSizeX == 31U ? 2U : 6U;
    return effectiveWidth * (verticalTaps + horizontalTaps) +
        boundaryPixels * horizontalTaps * boundaryExtraHorizontalPenalty;
}

__aicore__ inline void MapGaussianBlurC1WeightedTask(
    uint32_t taskId,
    uint32_t taskBudget,
    uint32_t width,
    uint32_t kernelSizeX,
    uint32_t kernelSizeY,
    uint32_t tileWidth,
    uint32_t tilesX,
    uint32_t* selectedTileX,
    uint32_t* selectedTileY,
    uint32_t* selectedTilesY)
{
    uint64_t totalWeight = 0U;
    for (uint32_t tileX = 0U; tileX < tilesX; ++tileX) {
        totalWeight += GaussianBlurC1TileWeight(
            width, kernelSizeX, kernelSizeY, tileWidth, tileX);
    }
    const uint32_t remainingTasks = taskBudget > tilesX ? taskBudget - tilesX : 0U;
    uint64_t prefixWeight = 0U;
    uint32_t taskOffset = 0U;
    for (uint32_t tileX = 0U; tileX < tilesX; ++tileX) {
        const uint32_t weight = GaussianBlurC1TileWeight(
            width, kernelSizeX, kernelSizeY, tileWidth, tileX);
        const uint64_t rounding = kernelSizeX == 31U ? totalWeight / 2U : 0U;
        const uint32_t begin = static_cast<uint32_t>(
            (prefixWeight * remainingTasks + rounding) / totalWeight);
        prefixWeight += weight;
        const uint32_t end = static_cast<uint32_t>(
            (prefixWeight * remainingTasks + rounding) / totalWeight);
        const uint32_t tilesY = 1U + end - begin;
        if (taskId < taskOffset + tilesY) {
            *selectedTileX = tileX;
            *selectedTileY = taskId - taskOffset;
            *selectedTilesY = tilesY;
            return;
        }
        taskOffset += tilesY;
    }
    *selectedTileX = tilesX - 1U;
    *selectedTileY = 0U;
    *selectedTilesY = 1U;
}

__aicore__ inline void MapGaussianBlurEvenSpatialTask(
    uint32_t taskId,
    uint32_t taskBudget,
    uint32_t tilesX,
    uint32_t* selectedTileX,
    uint32_t* selectedTileY,
    uint32_t* selectedTilesY)
{
    const uint32_t remainingTasks = taskBudget > tilesX ? taskBudget - tilesX : 0U;
    uint32_t taskOffset = 0U;
    for (uint32_t tileX = 0U; tileX < tilesX; ++tileX) {
        const uint32_t begin = (tileX * remainingTasks + tilesX / 2U) / tilesX;
        const uint32_t end = ((tileX + 1U) * remainingTasks + tilesX / 2U) / tilesX;
        const uint32_t tilesY = 1U + end - begin;
        if (taskId < taskOffset + tilesY) {
            *selectedTileX = tileX;
            *selectedTileY = taskId - taskOffset;
            *selectedTilesY = tilesY;
            return;
        }
        taskOffset += tilesY;
    }
    *selectedTileX = tilesX - 1U;
    *selectedTileY = 0U;
    *selectedTilesY = 1U;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurK31GatherC16RowKernel(
    int32_t logicalRow,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t tileX,
    uint32_t tileWidth,
    uint32_t channelOffset,
    uint32_t packedChannels,
    uint32_t borderType,
    uint32_t radiusX,
    uint32_t inputWidth,
    uint32_t validLocalStart,
    uint32_t validLocalEnd,
    __ubuf__ float* raw,
    __gm__ const float* src)
{
    const int32_t sourceY = BorderCoord(logicalRow, static_cast<int32_t>(height), borderType);
    const int32_t logicalBaseX =
        static_cast<int32_t>(tileX * tileWidth) - static_cast<int32_t>(radiusX);
    const uint32_t activeElements = inputWidth * packedChannels;
    for (uint32_t task = threadIdx.x; task < activeElements; task += blockDim.x) {
        const uint32_t patchX = task / packedChannels;
        const uint32_t localChannel = task - patchX * packedChannels;
        if (patchX >= validLocalStart && patchX < validLocalEnd) {
            continue;
        }
        const uint32_t channel = channelOffset + localChannel;
        const int32_t sourceX = BorderCoord(
            logicalBaseX + static_cast<int32_t>(patchX), static_cast<int32_t>(width), borderType);
        float value = 0.0f;
        if (sourceY >= 0 && sourceX >= 0 && channel < channels) {
            value = src[ElementOffset(
                static_cast<uint32_t>(sourceY), static_cast<uint32_t>(sourceX),
                width, channels, channel)];
        }
        raw[task] = value;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(128) inline void GaussianBlurK31FillC16HaloKernel(
    int32_t logicalRow,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t tileX,
    uint32_t tileWidth,
    uint32_t channelOffset,
    uint32_t packedChannels,
    uint32_t borderType,
    uint32_t radiusX,
    uint32_t inputWidth,
    uint32_t validLocalStart,
    uint32_t validLocalEnd,
    __ubuf__ float* raw)
{
    const int32_t sourceY = BorderCoord(logicalRow, static_cast<int32_t>(height), borderType);
    const int32_t logicalBaseX =
        static_cast<int32_t>(tileX * tileWidth) - static_cast<int32_t>(radiusX);
    const uint32_t haloPixels = validLocalStart + inputWidth - validLocalEnd;
    const uint32_t haloElements = haloPixels * packedChannels;
    for (uint32_t task = threadIdx.x; task < haloElements; task += blockDim.x) {
        const uint32_t haloX = task / packedChannels;
        const uint32_t localChannel = task - haloX * packedChannels;
        const uint32_t patchX = haloX < validLocalStart ?
            haloX : validLocalEnd + haloX - validLocalStart;
        const int32_t sourceX = BorderCoord(
            logicalBaseX + static_cast<int32_t>(patchX), static_cast<int32_t>(width), borderType);
        float value = 0.0f;
        if (sourceY >= 0 && sourceX >= 0 && channelOffset + localChannel < channels) {
            const uint32_t sourceLocalX =
                static_cast<uint32_t>(sourceX - logicalBaseX);
            value = raw[sourceLocalX * packedChannels + localChannel];
        }
        raw[patchX * packedChannels + localChannel] = value;
    }
}

__simd_vf__ inline void GaussianBlurK31HorizontalC16RowVF(
    __ubuf__ float* raw,
    __ubuf__ const float* weightsX,
    uint32_t channelStride,
    uint32_t outputElements,
    __ubuf__ float* ringRow)
{
    auto mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    const uint32_t vectorsPerRow = (outputElements + 63U) / 64U;
    const uint32_t centerOffset = K31_RING_RADIUS * channelStride;
    uint32_t vectorIndex = 0U;
    for (; vectorIndex + 1U < vectorsPerRow; vectorIndex += 2U) {
        const uint32_t vectorOffset0 = vectorIndex * 64U;
        const uint32_t vectorOffset1 = vectorOffset0 + 64U;
        AscendC::MicroAPI::RegTensor<float> sumEven0;
        AscendC::MicroAPI::RegTensor<float> sumOdd0;
        AscendC::MicroAPI::RegTensor<float> sumEven1;
        AscendC::MicroAPI::RegTensor<float> sumOdd1;
        AscendC::MicroAPI::RegTensor<float> pair;
        AscendC::MicroAPI::RegTensor<float> left;
        AscendC::MicroAPI::RegTensor<float> right;
        AscendC::MicroAPI::RegTensor<float> weight;
        AscendC::MicroAPI::LoadAlign<float>(
            sumEven0, raw + centerOffset + vectorOffset0);
        AscendC::MicroAPI::LoadAlign<float>(
            sumEven1, raw + centerOffset + vectorOffset1);
        AscendC::MicroAPI::Duplicate(weight, weightsX[K31_RING_RADIUS]);
        AscendC::MicroAPI::Mul(sumEven0, sumEven0, weight, mask);
        AscendC::MicroAPI::Mul(sumEven1, sumEven1, weight, mask);
        AscendC::MicroAPI::Duplicate(sumOdd0, 0.0f);
        AscendC::MicroAPI::Duplicate(sumOdd1, 0.0f);
#pragma unroll 1
        for (uint32_t offset = 1U; offset < K31_RING_RADIUS; offset += 2U) {
            const uint32_t elementOffset = offset * channelStride;
            AscendC::MicroAPI::LoadAlign<float>(
                left, raw + centerOffset + vectorOffset0 - elementOffset);
            AscendC::MicroAPI::LoadAlign<float>(
                right, raw + centerOffset + vectorOffset0 + elementOffset);
            AscendC::MicroAPI::Add(pair, left, right, mask);
            AscendC::MicroAPI::Duplicate(weight, weightsX[K31_RING_RADIUS - offset]);
            AscendC::MicroAPI::MulAddDst(sumOdd0, pair, weight, mask);
            AscendC::MicroAPI::LoadAlign<float>(
                left, raw + centerOffset + vectorOffset1 - elementOffset);
            AscendC::MicroAPI::LoadAlign<float>(
                right, raw + centerOffset + vectorOffset1 + elementOffset);
            AscendC::MicroAPI::Add(pair, left, right, mask);
            AscendC::MicroAPI::MulAddDst(sumOdd1, pair, weight, mask);

            const uint32_t evenOffset = offset + 1U;
            const uint32_t evenElementOffset = evenOffset * channelStride;
            AscendC::MicroAPI::LoadAlign<float>(
                left, raw + centerOffset + vectorOffset0 - evenElementOffset);
            AscendC::MicroAPI::LoadAlign<float>(
                right, raw + centerOffset + vectorOffset0 + evenElementOffset);
            AscendC::MicroAPI::Add(pair, left, right, mask);
            AscendC::MicroAPI::Duplicate(weight, weightsX[K31_RING_RADIUS - evenOffset]);
            AscendC::MicroAPI::MulAddDst(sumEven0, pair, weight, mask);
            AscendC::MicroAPI::LoadAlign<float>(
                left, raw + centerOffset + vectorOffset1 - evenElementOffset);
            AscendC::MicroAPI::LoadAlign<float>(
                right, raw + centerOffset + vectorOffset1 + evenElementOffset);
            AscendC::MicroAPI::Add(pair, left, right, mask);
            AscendC::MicroAPI::MulAddDst(sumEven1, pair, weight, mask);
        }
        const uint32_t lastElementOffset = K31_RING_RADIUS * channelStride;
        AscendC::MicroAPI::LoadAlign<float>(
            left, raw + centerOffset + vectorOffset0 - lastElementOffset);
        AscendC::MicroAPI::LoadAlign<float>(
            right, raw + centerOffset + vectorOffset0 + lastElementOffset);
        AscendC::MicroAPI::Add(pair, left, right, mask);
        AscendC::MicroAPI::Duplicate(weight, weightsX[0U]);
        AscendC::MicroAPI::MulAddDst(sumOdd0, pair, weight, mask);
        AscendC::MicroAPI::LoadAlign<float>(
            left, raw + centerOffset + vectorOffset1 - lastElementOffset);
        AscendC::MicroAPI::LoadAlign<float>(
            right, raw + centerOffset + vectorOffset1 + lastElementOffset);
        AscendC::MicroAPI::Add(pair, left, right, mask);
        AscendC::MicroAPI::MulAddDst(sumOdd1, pair, weight, mask);
        AscendC::MicroAPI::Add(sumEven0, sumEven0, sumOdd0, mask);
        AscendC::MicroAPI::Add(sumEven1, sumEven1, sumOdd1, mask);
        AscendC::MicroAPI::StoreAlign<float>(ringRow + vectorOffset0, sumEven0, mask);
        AscendC::MicroAPI::StoreAlign<float>(ringRow + vectorOffset1, sumEven1, mask);
    }
    if (vectorIndex < vectorsPerRow) {
        const uint32_t vectorOffset = vectorIndex * 64U;
        AscendC::MicroAPI::RegTensor<float> sum0;
        AscendC::MicroAPI::RegTensor<float> sum1;
        AscendC::MicroAPI::RegTensor<float> pair;
        AscendC::MicroAPI::RegTensor<float> left;
        AscendC::MicroAPI::RegTensor<float> right;
        AscendC::MicroAPI::RegTensor<float> weight;
        __ubuf__ float* loadAddress = raw + centerOffset + vectorOffset;
        AscendC::MicroAPI::LoadAlign<float>(sum0, loadAddress);
        AscendC::MicroAPI::Duplicate(weight, weightsX[K31_RING_RADIUS]);
        AscendC::MicroAPI::Mul(sum0, sum0, weight, mask);
        AscendC::MicroAPI::Duplicate(sum1, 0.0f);
#pragma unroll 1
        for (uint32_t offset = 1U; offset < K31_RING_RADIUS; offset += 2U) {
            const uint32_t elementOffset = offset * channelStride;
            loadAddress = raw + centerOffset + vectorOffset - elementOffset;
            AscendC::MicroAPI::LoadAlign<float>(left, loadAddress);
            loadAddress = raw + centerOffset + vectorOffset + elementOffset;
            AscendC::MicroAPI::LoadAlign<float>(right, loadAddress);
            AscendC::MicroAPI::Add(pair, left, right, mask);
            AscendC::MicroAPI::Duplicate(weight, weightsX[K31_RING_RADIUS - offset]);
            AscendC::MicroAPI::MulAddDst(sum1, pair, weight, mask);

            const uint32_t evenOffset = offset + 1U;
            const uint32_t evenElementOffset = evenOffset * channelStride;
            loadAddress = raw + centerOffset + vectorOffset - evenElementOffset;
            AscendC::MicroAPI::LoadAlign<float>(left, loadAddress);
            loadAddress = raw + centerOffset + vectorOffset + evenElementOffset;
            AscendC::MicroAPI::LoadAlign<float>(right, loadAddress);
            AscendC::MicroAPI::Add(pair, left, right, mask);
            AscendC::MicroAPI::Duplicate(weight, weightsX[K31_RING_RADIUS - evenOffset]);
            AscendC::MicroAPI::MulAddDst(sum0, pair, weight, mask);
        }
        const uint32_t lastElementOffset = K31_RING_RADIUS * channelStride;
        loadAddress = raw + centerOffset + vectorOffset - lastElementOffset;
        AscendC::MicroAPI::LoadAlign<float>(left, loadAddress);
        loadAddress = raw + centerOffset + vectorOffset + lastElementOffset;
        AscendC::MicroAPI::LoadAlign<float>(right, loadAddress);
        AscendC::MicroAPI::Add(pair, left, right, mask);
        AscendC::MicroAPI::Duplicate(weight, weightsX[0U]);
        AscendC::MicroAPI::MulAddDst(sum1, pair, weight, mask);
        AscendC::MicroAPI::Add(sum0, sum0, sum1, mask);
        AscendC::MicroAPI::StoreAlign<float>(ringRow + vectorOffset, sum0, mask);
    }
}

__simd_vf__ inline void GaussianBlurK31VerticalC16RingVF(
    __ubuf__ float* ring,
    __ubuf__ const float* weightsY,
    uint32_t centerSlot,
    __ubuf__ float* output)
{
    auto mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    constexpr uint32_t vectorsPerRow = K31_STREAM_RING_ROW_ELEMENTS / 64U;
    for (uint32_t vectorIndex = 0U; vectorIndex < vectorsPerRow; ++vectorIndex) {
        const uint32_t vectorOffset = vectorIndex * 64U;
        AscendC::MicroAPI::RegTensor<float> sum0;
        AscendC::MicroAPI::RegTensor<float> sum1;
        AscendC::MicroAPI::RegTensor<float> pair;
        AscendC::MicroAPI::RegTensor<float> upper;
        AscendC::MicroAPI::RegTensor<float> lower;
        AscendC::MicroAPI::RegTensor<float> weight;
        AscendC::MicroAPI::LoadAlign<float>(
            sum0, ring + centerSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
        AscendC::MicroAPI::Duplicate(weight, weightsY[K31_RING_RADIUS]);
        AscendC::MicroAPI::Mul(sum0, sum0, weight, mask);
        AscendC::MicroAPI::Duplicate(sum1, 0.0f);
#pragma unroll 1
        for (uint32_t offset = 1U; offset < K31_RING_RADIUS; offset += 2U) {
            const uint32_t upperSlot = centerSlot >= offset ?
                centerSlot - offset : centerSlot + K31_RING_ROWS - offset;
            const uint32_t lowerUnwrapped = centerSlot + offset;
            const uint32_t lowerSlot = lowerUnwrapped < K31_RING_ROWS ?
                lowerUnwrapped : lowerUnwrapped - K31_RING_ROWS;
            AscendC::MicroAPI::LoadAlign<float>(
                upper, ring + upperSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
            AscendC::MicroAPI::LoadAlign<float>(
                lower, ring + lowerSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
            AscendC::MicroAPI::Add(pair, upper, lower, mask);
            AscendC::MicroAPI::Duplicate(weight, weightsY[K31_RING_RADIUS - offset]);
            AscendC::MicroAPI::MulAddDst(sum1, pair, weight, mask);

            const uint32_t evenOffset = offset + 1U;
            const uint32_t evenUpperSlot = centerSlot >= evenOffset ?
                centerSlot - evenOffset : centerSlot + K31_RING_ROWS - evenOffset;
            const uint32_t evenLowerUnwrapped = centerSlot + evenOffset;
            const uint32_t evenLowerSlot = evenLowerUnwrapped < K31_RING_ROWS ?
                evenLowerUnwrapped : evenLowerUnwrapped - K31_RING_ROWS;
            AscendC::MicroAPI::LoadAlign<float>(
                upper, ring + evenUpperSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
            AscendC::MicroAPI::LoadAlign<float>(
                lower, ring + evenLowerSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
            AscendC::MicroAPI::Add(pair, upper, lower, mask);
            AscendC::MicroAPI::Duplicate(weight, weightsY[K31_RING_RADIUS - evenOffset]);
            AscendC::MicroAPI::MulAddDst(sum0, pair, weight, mask);
        }
        constexpr uint32_t lastOffset = K31_RING_RADIUS;
        const uint32_t lastUpperSlot = centerSlot >= lastOffset ?
            centerSlot - lastOffset : centerSlot + K31_RING_ROWS - lastOffset;
        const uint32_t lastLowerUnwrapped = centerSlot + lastOffset;
        const uint32_t lastLowerSlot = lastLowerUnwrapped < K31_RING_ROWS ?
            lastLowerUnwrapped : lastLowerUnwrapped - K31_RING_ROWS;
        AscendC::MicroAPI::LoadAlign<float>(
            upper, ring + lastUpperSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
        AscendC::MicroAPI::LoadAlign<float>(
            lower, ring + lastLowerSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
        AscendC::MicroAPI::Add(pair, upper, lower, mask);
        AscendC::MicroAPI::Duplicate(weight, weightsY[0U]);
        AscendC::MicroAPI::MulAddDst(sum1, pair, weight, mask);
        AscendC::MicroAPI::Add(sum0, sum0, sum1, mask);
        AscendC::MicroAPI::StoreAlign<float>(output + vectorOffset, sum0, mask);
    }
}

__simd_vf__ inline void GaussianBlurHorizontalC8RuntimeVF(
    __ubuf__ float* raw,
    __ubuf__ const float* weightsX,
    uint32_t radiusX,
    uint32_t outputElements,
    __ubuf__ float* ringRow)
{
    auto mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    const uint32_t vectorsPerRow = (outputElements + 63U) / 64U;
    const uint32_t centerOffset = radiusX * K31_STREAM_CHANNELS;
    for (uint32_t vectorIndex = 0U; vectorIndex < vectorsPerRow; ++vectorIndex) {
        const uint32_t vectorOffset = vectorIndex * 64U;
        AscendC::MicroAPI::RegTensor<float> sum;
        AscendC::MicroAPI::RegTensor<float> pair;
        AscendC::MicroAPI::RegTensor<float> left;
        AscendC::MicroAPI::RegTensor<float> right;
        AscendC::MicroAPI::RegTensor<float> weight;
        AscendC::MicroAPI::LoadAlign<float>(sum, raw + centerOffset + vectorOffset);
        AscendC::MicroAPI::Duplicate(weight, weightsX[radiusX]);
        AscendC::MicroAPI::Mul(sum, sum, weight, mask);
#pragma unroll 1
        for (uint32_t offset = 1U; offset <= radiusX; ++offset) {
            const uint32_t elementOffset = offset * K31_STREAM_CHANNELS;
            AscendC::MicroAPI::LoadAlign<float>(
                left, raw + centerOffset + vectorOffset - elementOffset);
            AscendC::MicroAPI::LoadAlign<float>(
                right, raw + centerOffset + vectorOffset + elementOffset);
            AscendC::MicroAPI::Add(pair, left, right, mask);
            AscendC::MicroAPI::Duplicate(weight, weightsX[radiusX - offset]);
            AscendC::MicroAPI::MulAddDst(sum, pair, weight, mask);
        }
        AscendC::MicroAPI::StoreAlign<float>(ringRow + vectorOffset, sum, mask);
    }
}

__simd_vf__ inline void GaussianBlurVerticalC8RuntimeVF(
    __ubuf__ float* ring,
    __ubuf__ const float* weightsY,
    uint32_t ringRows,
    uint32_t radiusY,
    uint32_t centerSlot,
    uint32_t outputElements,
    __ubuf__ float* output)
{
    auto mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    const uint32_t vectorsPerRow = (outputElements + 63U) / 64U;
    for (uint32_t vectorIndex = 0U; vectorIndex < vectorsPerRow; ++vectorIndex) {
        const uint32_t vectorOffset = vectorIndex * 64U;
        AscendC::MicroAPI::RegTensor<float> sum;
        AscendC::MicroAPI::RegTensor<float> pair;
        AscendC::MicroAPI::RegTensor<float> upper;
        AscendC::MicroAPI::RegTensor<float> lower;
        AscendC::MicroAPI::RegTensor<float> weight;
        AscendC::MicroAPI::LoadAlign<float>(
            sum, ring + centerSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
        AscendC::MicroAPI::Duplicate(weight, weightsY[radiusY]);
        AscendC::MicroAPI::Mul(sum, sum, weight, mask);
#pragma unroll 1
        for (uint32_t offset = 1U; offset <= radiusY; ++offset) {
            const uint32_t upperSlot = centerSlot >= offset ?
                centerSlot - offset : centerSlot + ringRows - offset;
            const uint32_t lowerUnwrapped = centerSlot + offset;
            const uint32_t lowerSlot = lowerUnwrapped < ringRows ?
                lowerUnwrapped : lowerUnwrapped - ringRows;
            AscendC::MicroAPI::LoadAlign<float>(
                upper, ring + upperSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
            AscendC::MicroAPI::LoadAlign<float>(
                lower, ring + lowerSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
            AscendC::MicroAPI::Add(pair, upper, lower, mask);
            AscendC::MicroAPI::Duplicate(weight, weightsY[radiusY - offset]);
            AscendC::MicroAPI::MulAddDst(sum, pair, weight, mask);
        }
        AscendC::MicroAPI::StoreAlign<float>(output + vectorOffset, sum, mask);
    }
}

__simd_vf__ inline void GaussianBlurVerticalC8AdjacentPairVF(
    __ubuf__ float* ring,
    __ubuf__ const float* weightsY,
    uint32_t ringRows,
    uint32_t replaceSlot,
    uint32_t outputElements,
    __ubuf__ float* pending,
    __ubuf__ float* output0,
    __ubuf__ float* output1)
{
    auto mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    const uint32_t vectorsPerRow = (outputElements + 63U) / 64U;
    for (uint32_t vectorIndex = 0U; vectorIndex < vectorsPerRow; ++vectorIndex) {
        const uint32_t vectorOffset = vectorIndex * 64U;
        AscendC::MicroAPI::RegTensor<float> sum0;
        AscendC::MicroAPI::RegTensor<float> sum1;
        AscendC::MicroAPI::RegTensor<float> row;
        AscendC::MicroAPI::RegTensor<float> weight;
        AscendC::MicroAPI::LoadAlign<float>(
            row, ring + replaceSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
        AscendC::MicroAPI::Duplicate(weight, weightsY[0U]);
        AscendC::MicroAPI::Mul(sum0, row, weight, mask);
        AscendC::MicroAPI::Duplicate(sum1, 0.0f);
#pragma unroll 1
        for (uint32_t logicalRow = 1U; logicalRow < ringRows; ++logicalRow) {
            uint32_t slot = replaceSlot + logicalRow;
            slot = slot < ringRows ? slot : slot - ringRows;
            AscendC::MicroAPI::LoadAlign<float>(
                row, ring + slot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset);
            AscendC::MicroAPI::Duplicate(weight, weightsY[logicalRow]);
            AscendC::MicroAPI::MulAddDst(sum0, row, weight, mask);
            AscendC::MicroAPI::Duplicate(weight, weightsY[logicalRow - 1U]);
            AscendC::MicroAPI::MulAddDst(sum1, row, weight, mask);
        }
        AscendC::MicroAPI::LoadAlign<float>(row, pending + vectorOffset);
        AscendC::MicroAPI::Duplicate(weight, weightsY[ringRows - 1U]);
        AscendC::MicroAPI::MulAddDst(sum1, row, weight, mask);
        AscendC::MicroAPI::StoreAlign<float>(output0 + vectorOffset, sum0, mask);
        AscendC::MicroAPI::StoreAlign<float>(output1 + vectorOffset, sum1, mask);
        AscendC::MicroAPI::StoreAlign<float>(
            ring + replaceSlot * K31_STREAM_RING_ROW_ELEMENTS + vectorOffset, row, mask);
    }
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurGatherC1RowKernel(
    int32_t logicalRow,
    uint32_t height,
    uint32_t width,
    uint32_t tileX,
    uint32_t tileWidth,
    uint32_t borderType,
    uint32_t radiusX,
    uint32_t inputWidth,
    uint32_t validLocalStart,
    uint32_t validLocalEnd,
    __ubuf__ float* raw,
    __ubuf__ const float* staging,
    __gm__ const float* src)
{
    const int32_t sourceY = BorderCoord(logicalRow, static_cast<int32_t>(height), borderType);
    const int32_t logicalBaseX =
        static_cast<int32_t>(tileX * tileWidth) - static_cast<int32_t>(radiusX);
    for (uint32_t patchX = threadIdx.x; patchX < inputWidth; patchX += blockDim.x) {
        float value = 0.0f;
        if (sourceY >= 0 && patchX >= validLocalStart && patchX < validLocalEnd) {
            value = staging[patchX - validLocalStart];
        } else {
            const int32_t sourceX = BorderCoord(
                logicalBaseX + static_cast<int32_t>(patchX), static_cast<int32_t>(width), borderType);
            if (sourceY >= 0 && sourceX >= 0) {
                value = src[static_cast<uint64_t>(sourceY) * width + static_cast<uint32_t>(sourceX)];
            }
        }
        raw[patchX] = value;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(64) inline void GaussianBlurFillC1HaloKernel(
    uint32_t width,
    uint32_t tileX,
    uint32_t tileWidth,
    uint32_t borderType,
    uint32_t radiusX,
    uint32_t inputWidth,
    uint32_t validLocalStart,
    uint32_t validLocalEnd,
    __ubuf__ float* raw)
{
    const int32_t logicalBaseX =
        static_cast<int32_t>(tileX * tileWidth) - static_cast<int32_t>(radiusX);
    const uint32_t haloPixels = validLocalStart + inputWidth - validLocalEnd;
    for (uint32_t haloX = threadIdx.x; haloX < haloPixels; haloX += blockDim.x) {
        const uint32_t patchX = haloX < validLocalStart ?
            haloX : validLocalEnd + haloX - validLocalStart;
        const int32_t sourceX = BorderCoord(
            logicalBaseX + static_cast<int32_t>(patchX), static_cast<int32_t>(width), borderType);
        float value = 0.0f;
        if (sourceX >= 0) {
            value = raw[static_cast<uint32_t>(sourceX - logicalBaseX)];
        }
        raw[patchX] = value;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurHorizontalC1RuntimeVF(
    __ubuf__ float* raw,
    __ubuf__ const float* weightsX,
    uint32_t radiusX,
    uint32_t outputWidth,
    __ubuf__ float* ringRow)
{
    for (uint32_t outputX = threadIdx.x; outputX < outputWidth;
         outputX += blockDim.x) {
        const uint32_t center = outputX + radiusX;
        float sum = raw[center] * weightsX[radiusX];
#pragma unroll 1
        for (uint32_t offset = 1U; offset <= radiusX; ++offset) {
            sum += (raw[center - offset] + raw[center + offset]) * weightsX[radiusX - offset];
        }
        ringRow[outputX] = sum;
    }
}

template <uint32_t KernelSize>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurHorizontalC1FixedVF(
    __ubuf__ float* raw,
    __ubuf__ const float* weightsX,
    uint32_t outputWidth,
    __ubuf__ float* ringRow)
{
    constexpr uint32_t radiusX = KernelSize / 2U;
    for (uint32_t outputX = threadIdx.x; outputX < outputWidth;
         outputX += blockDim.x) {
        const uint32_t center = outputX + radiusX;
        float sum = raw[center] * weightsX[radiusX];
#pragma unroll
        for (uint32_t offset = 1U; offset <= radiusX; ++offset) {
            sum += (raw[center - offset] + raw[center + offset]) *
                weightsX[radiusX - offset];
        }
        ringRow[outputX] = sum;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurHorizontalC1K31PairVF(
    __ubuf__ float* raw,
    __ubuf__ const float* weightsX,
    uint32_t outputWidth,
    __ubuf__ float* ringRow)
{
    constexpr uint32_t radiusX = 15U;
    const uint32_t pairsPerRow = (outputWidth + 1U) / 2U;
    for (uint32_t pairIndex = threadIdx.x; pairIndex < pairsPerRow;
         pairIndex += blockDim.x) {
        const uint32_t outputX = pairIndex * 2U;
        const uint32_t center = outputX + radiusX;
        const float center0 = raw[center];
        float sum0 = center0 * weightsX[radiusX];
        if (outputX + 1U < outputWidth) {
            const float center1 = raw[center + 1U];
            float sum1 = center1 * weightsX[radiusX];
            float previousLeft = center0;
            float previousRight = center1;
#pragma unroll
            for (uint32_t offset = 1U; offset <= radiusX; ++offset) {
                const float left = raw[center - offset];
                const float right = raw[center + 1U + offset];
                const float weight = weightsX[radiusX - offset];
                sum0 += (left + previousRight) * weight;
                sum1 += (previousLeft + right) * weight;
                previousLeft = left;
                previousRight = right;
            }
            ringRow[outputX + 1U] = sum1;
        } else {
#pragma unroll
            for (uint32_t offset = 1U; offset <= radiusX; ++offset) {
                sum0 += (raw[center - offset] + raw[center + offset]) *
                    weightsX[radiusX - offset];
            }
        }
        ringRow[outputX] = sum0;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurHorizontalC1K31DualVF(
    __ubuf__ float* raw0,
    __ubuf__ float* raw1,
    __ubuf__ const float* weightsX,
    uint32_t outputWidth,
    __ubuf__ float* ringRow0,
    __ubuf__ float* ringRow1)
{
    constexpr uint32_t radiusX = 15U;
    const uint32_t lane = threadIdx.x;
    __ubuf__ float* rawRow = nullptr;
    __ubuf__ float* ringRow = nullptr;
    uint32_t outputX = 0U;
    if (lane < outputWidth) {
        rawRow = raw0;
        ringRow = ringRow0;
        outputX = lane;
    } else if (lane >= K31_C1_RING_ROW_ELEMENTS &&
               lane < K31_C1_RING_ROW_ELEMENTS + outputWidth) {
        rawRow = raw1;
        ringRow = ringRow1;
        outputX = lane - K31_C1_RING_ROW_ELEMENTS;
    } else {
        return;
    }
    const uint32_t center = outputX + radiusX;
    float sum = rawRow[center] * weightsX[radiusX];
#pragma unroll
    for (uint32_t offset = 1U; offset <= radiusX; ++offset) {
        sum += (rawRow[center - offset] + rawRow[center + offset]) *
            weightsX[radiusX - offset];
    }
    ringRow[outputX] = sum;
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurVerticalC1RuntimeVF(
    __ubuf__ float* ring,
    __ubuf__ const float* weightsY,
    uint32_t ringRows,
    uint32_t radiusY,
    uint32_t centerSlot,
    uint32_t outputWidth,
    __ubuf__ float* output)
{
    for (uint32_t outputX = threadIdx.x; outputX < outputWidth;
         outputX += blockDim.x) {
        float sum = ring[centerSlot * K31_C1_RING_ROW_ELEMENTS + outputX] * weightsY[radiusY];
#pragma unroll 1
        for (uint32_t offset = 1U; offset <= radiusY; ++offset) {
            const uint32_t upperSlot = centerSlot >= offset ?
                centerSlot - offset : centerSlot + ringRows - offset;
            const uint32_t lowerUnwrapped = centerSlot + offset;
            const uint32_t lowerSlot = lowerUnwrapped < ringRows ?
                lowerUnwrapped : lowerUnwrapped - ringRows;
            sum += (ring[upperSlot * K31_C1_RING_ROW_ELEMENTS + outputX] +
                    ring[lowerSlot * K31_C1_RING_ROW_ELEMENTS + outputX]) *
                weightsY[radiusY - offset];
        }
        output[outputX] = sum;
    }
}

template <uint32_t KernelSize>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurVerticalC1FixedVF(
    __ubuf__ float* ring,
    __ubuf__ const float* weightsY,
    uint32_t centerSlot,
    uint32_t outputWidth,
    __ubuf__ float* output)
{
    constexpr uint32_t radiusY = KernelSize / 2U;
    for (uint32_t outputX = threadIdx.x; outputX < outputWidth;
         outputX += blockDim.x) {
        float sum = ring[centerSlot * K31_C1_RING_ROW_ELEMENTS + outputX] * weightsY[radiusY];
#pragma unroll
        for (uint32_t offset = 1U; offset <= radiusY; ++offset) {
            const uint32_t upperSlot = centerSlot >= offset ?
                centerSlot - offset : centerSlot + KernelSize - offset;
            const uint32_t lowerUnwrapped = centerSlot + offset;
            const uint32_t lowerSlot = lowerUnwrapped < KernelSize ?
                lowerUnwrapped : lowerUnwrapped - KernelSize;
            sum += (ring[upperSlot * K31_C1_RING_ROW_ELEMENTS + outputX] +
                    ring[lowerSlot * K31_C1_RING_ROW_ELEMENTS + outputX]) *
                weightsY[radiusY - offset];
        }
        output[outputX] = sum;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(256) inline void CommitGaussianBlurC1RowVF(
    __ubuf__ const float* pendingRow,
    __ubuf__ float* commitRow,
    uint32_t outputWidth)
{
    for (uint32_t outputX = threadIdx.x; outputX < outputWidth;
         outputX += blockDim.x) {
        commitRow[outputX] = pendingRow[outputX];
    }
}



__aicore__ inline void RunGaussianBlurHorizontalC1(
    __ubuf__ float* raw,
    __ubuf__ const float* weightsX,
    uint32_t radiusX,
    uint32_t outputWidth,
    __ubuf__ float* ringRow)
{
    if (radiusX == 15U) {
        asc_vf_call<GaussianBlurHorizontalC1K31PairVF>(
            dim3{256U, 1U, 1U}, raw, weightsX, outputWidth, ringRow);
        return;
    }
    asc_vf_call<GaussianBlurHorizontalC1RuntimeVF>(
        dim3{256U, 1U, 1U}, raw, weightsX, radiusX, outputWidth, ringRow);
}

__aicore__ inline void RunGaussianBlurVerticalC1(
    __ubuf__ float* ring,
    __ubuf__ const float* weightsY,
    uint32_t ringRows,
    uint32_t radiusY,
    uint32_t centerSlot,
    uint32_t outputWidth,
    __ubuf__ float* output)
{
    if (ringRows == 5U) {
        asc_vf_call<GaussianBlurVerticalC1FixedVF<5U>>(
            dim3{256U, 1U, 1U}, ring, weightsY, centerSlot, outputWidth, output);
        return;
    }
    if (ringRows == 15U) {
        asc_vf_call<GaussianBlurVerticalC1FixedVF<15U>>(
            dim3{256U, 1U, 1U}, ring, weightsY, centerSlot, outputWidth, output);
        return;
    }
    if (ringRows == 31U) {
        asc_vf_call<GaussianBlurVerticalC1FixedVF<31U>>(
            dim3{256U, 1U, 1U}, ring, weightsY, centerSlot, outputWidth, output);
        return;
    }
    asc_vf_call<GaussianBlurVerticalC1RuntimeVF>(
        dim3{256U, 1U, 1U}, ring, weightsY, ringRows, radiusY,
        centerSlot, outputWidth, output);
}

template <uint32_t FixedKernelSizeX = 0U, uint32_t FixedKernelSizeY = 0U>
__simt_vf__ __aicore__ __launch_bounds__(256) inline void GaussianBlurC1TileVF(
    __gm__ const float* src,
    __gm__ float* dst,
    __ubuf__ float* horizontal,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* weightsY,
    uint32_t height,
    uint32_t width,
    uint32_t kernelSizeX,
    uint32_t kernelSizeY,
    uint32_t borderType,
    uint32_t outputBaseY,
    uint32_t outputRows,
    uint32_t outputBaseX,
    uint32_t outputWidth)
{
    const uint32_t effectiveKernelSizeX = FixedKernelSizeX != 0U ?
        FixedKernelSizeX : kernelSizeX;
    const uint32_t effectiveKernelSizeY = FixedKernelSizeY != 0U ?
        FixedKernelSizeY : kernelSizeY;
    const uint32_t radiusX = effectiveKernelSizeX / 2U;
    const uint32_t radiusY = effectiveKernelSizeY / 2U;
    const uint32_t horizontalRows = outputRows + effectiveKernelSizeY - 1U;
    const uint32_t rowGroups = blockDim.x / outputWidth;
    const uint32_t activeThreads = rowGroups * outputWidth;
    const bool interiorX = outputBaseX >= radiusX &&
        outputBaseX + outputWidth + radiusX <= width;
    if constexpr (FixedKernelSizeX != 0U) {
        if (interiorX) {
            const uint32_t pairsPerRow = (outputWidth + 1U) / 2U;
            const uint32_t pairElements = horizontalRows * pairsPerRow;
            for (uint32_t pairElement = threadIdx.x; pairElement < pairElements;
                 pairElement += blockDim.x) {
                const uint32_t localRow = pairElement / pairsPerRow;
                const uint32_t localX = (pairElement % pairsPerRow) * 2U;
                const int32_t sourceY = BorderCoord(
                    static_cast<int32_t>(outputBaseY + localRow) - static_cast<int32_t>(radiusY),
                    static_cast<int32_t>(height), borderType);
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                if (sourceY >= 0) {
                    const uint64_t sourceRow = static_cast<uint64_t>(sourceY) * width;
                    const uint32_t centerX = outputBaseX + localX;
                    const float center0 = src[sourceRow + centerX];
                    sum0 = center0 * weightsX[FixedKernelSizeX / 2U];
                    if (localX + 1U < outputWidth) {
                        const float center1 = src[sourceRow + centerX + 1U];
                        sum1 = center1 * weightsX[FixedKernelSizeX / 2U];
                        float previousLeft = center0;
                        float previousRight = center1;
#pragma unroll
                        for (uint32_t offset = 1U; offset <= FixedKernelSizeX / 2U; ++offset) {
                            const float left = src[sourceRow + centerX - offset];
                            const float right = src[sourceRow + centerX + 1U + offset];
                            const float weight = weightsX[FixedKernelSizeX / 2U - offset];
                            sum0 += (left + previousRight) * weight;
                            sum1 += (previousLeft + right) * weight;
                            previousLeft = left;
                            previousRight = right;
                        }
                    } else {
#pragma unroll
                        for (uint32_t offset = 1U; offset <= FixedKernelSizeX / 2U; ++offset) {
                            const float pair = src[sourceRow + centerX - offset] +
                                src[sourceRow + centerX + offset];
                            sum0 += pair * weightsX[FixedKernelSizeX / 2U - offset];
                        }
                    }
                }
                horizontal[localRow * outputWidth + localX] = sum0;
                if (localX + 1U < outputWidth) {
                    horizontal[localRow * outputWidth + localX + 1U] = sum1;
                }
            }
        }
    }
    if (FixedKernelSizeX == 0U || !interiorX) {
        const uint32_t interiorBegin = outputBaseX < radiusX ?
            (radiusX - outputBaseX < outputWidth ? radiusX - outputBaseX : outputWidth) : 0U;
        const uint32_t interiorGlobalEnd = width > radiusX ? width - radiusX : 0U;
        const uint32_t interiorEnd = interiorGlobalEnd > outputBaseX ?
            (interiorGlobalEnd - outputBaseX < outputWidth ?
                interiorGlobalEnd - outputBaseX : outputWidth) : interiorBegin;
        const uint32_t interiorWidth = interiorEnd > interiorBegin ?
            interiorEnd - interiorBegin : 0U;
        if constexpr (FixedKernelSizeX != 0U) {
            const uint32_t pairsPerRow = (interiorWidth + 1U) / 2U;
            const uint32_t pairElements = horizontalRows * pairsPerRow;
            for (uint32_t pairElement = threadIdx.x; pairElement < pairElements;
                 pairElement += blockDim.x) {
                const uint32_t localRow = pairElement / pairsPerRow;
                const uint32_t localX = interiorBegin + (pairElement % pairsPerRow) * 2U;
                const int32_t sourceY = BorderCoord(
                    static_cast<int32_t>(outputBaseY + localRow) - static_cast<int32_t>(radiusY),
                    static_cast<int32_t>(height), borderType);
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                if (sourceY >= 0) {
                    const uint64_t sourceRow = static_cast<uint64_t>(sourceY) * width;
                    const uint32_t centerX = outputBaseX + localX;
                    const float center0 = src[sourceRow + centerX];
                    sum0 = center0 * weightsX[FixedKernelSizeX / 2U];
                    if (localX + 1U < interiorEnd) {
                        const float center1 = src[sourceRow + centerX + 1U];
                        sum1 = center1 * weightsX[FixedKernelSizeX / 2U];
                        float previousLeft = center0;
                        float previousRight = center1;
#pragma unroll
                        for (uint32_t offset = 1U; offset <= FixedKernelSizeX / 2U; ++offset) {
                            const float left = src[sourceRow + centerX - offset];
                            const float right = src[sourceRow + centerX + 1U + offset];
                            const float weight = weightsX[FixedKernelSizeX / 2U - offset];
                            sum0 += (left + previousRight) * weight;
                            sum1 += (previousLeft + right) * weight;
                            previousLeft = left;
                            previousRight = right;
                        }
                    } else {
#pragma unroll
                        for (uint32_t offset = 1U; offset <= FixedKernelSizeX / 2U; ++offset) {
                            const float pair = src[sourceRow + centerX - offset] +
                                src[sourceRow + centerX + offset];
                            sum0 += pair * weightsX[FixedKernelSizeX / 2U - offset];
                        }
                    }
                }
                horizontal[localRow * outputWidth + localX] = sum0;
                if (localX + 1U < interiorEnd) {
                    horizontal[localRow * outputWidth + localX + 1U] = sum1;
                }
            }
        } else {
            const uint32_t interiorElements = horizontalRows * interiorWidth;
            for (uint32_t element = threadIdx.x; element < interiorElements; element += blockDim.x) {
                const uint32_t localRow = element / interiorWidth;
                const uint32_t localX = interiorBegin + element % interiorWidth;
                const int32_t sourceY = BorderCoord(
                    static_cast<int32_t>(outputBaseY + localRow) - static_cast<int32_t>(radiusY),
                    static_cast<int32_t>(height), borderType);
                float sum = 0.0f;
                if (sourceY >= 0) {
                    const uint64_t sourceRow = static_cast<uint64_t>(sourceY) * width;
                    const uint32_t centerX = outputBaseX + localX;
                    sum = src[sourceRow + centerX] * weightsX[radiusX];
#pragma unroll 1
                    for (uint32_t offset = 1U; offset <= radiusX; ++offset) {
                        const float pair = src[sourceRow + centerX - offset] +
                            src[sourceRow + centerX + offset];
                        sum += pair * weightsX[radiusX - offset];
                    }
                }
                horizontal[localRow * outputWidth + localX] = sum;
            }
        }

        const uint32_t boundaryWidth = interiorBegin + outputWidth - interiorEnd;
        const uint32_t boundaryElements = horizontalRows * boundaryWidth;
        for (uint32_t element = threadIdx.x; element < boundaryElements; element += blockDim.x) {
            const uint32_t localRow = element / boundaryWidth;
            const uint32_t boundaryX = element % boundaryWidth;
            const uint32_t localX = boundaryX < interiorBegin ?
                boundaryX : interiorEnd + boundaryX - interiorBegin;
            const int32_t logicalCenterX = static_cast<int32_t>(outputBaseX + localX);
            const int32_t sourceY = BorderCoord(
                static_cast<int32_t>(outputBaseY + localRow) - static_cast<int32_t>(radiusY),
                static_cast<int32_t>(height), borderType);
            float sum = 0.0f;
            if (sourceY >= 0) {
                const uint64_t sourceRow = static_cast<uint64_t>(sourceY) * width;
                const int32_t centerX = BorderCoord(
                    logicalCenterX, static_cast<int32_t>(width), borderType);
                if (centerX >= 0) {
                    sum = src[sourceRow + static_cast<uint32_t>(centerX)] * weightsX[radiusX];
                }
                if constexpr (FixedKernelSizeX != 0U) {
#pragma unroll
                    for (uint32_t offset = 1U; offset <= FixedKernelSizeX / 2U; ++offset) {
                        const int32_t leftX = BorderCoord(
                            logicalCenterX - static_cast<int32_t>(offset),
                            static_cast<int32_t>(width), borderType);
                        const int32_t rightX = BorderCoord(
                            logicalCenterX + static_cast<int32_t>(offset),
                            static_cast<int32_t>(width), borderType);
                        float pair = 0.0f;
                        if (leftX >= 0) {
                            pair += src[sourceRow + static_cast<uint32_t>(leftX)];
                        }
                        if (rightX >= 0) {
                            pair += src[sourceRow + static_cast<uint32_t>(rightX)];
                        }
                        sum += pair * weightsX[FixedKernelSizeX / 2U - offset];
                    }
                } else {
#pragma unroll 1
                    for (uint32_t offset = 1U; offset <= radiusX; ++offset) {
                        const int32_t leftX = BorderCoord(
                            logicalCenterX - static_cast<int32_t>(offset),
                            static_cast<int32_t>(width), borderType);
                        const int32_t rightX = BorderCoord(
                            logicalCenterX + static_cast<int32_t>(offset),
                            static_cast<int32_t>(width), borderType);
                        float pair = 0.0f;
                        if (leftX >= 0) {
                            pair += src[sourceRow + static_cast<uint32_t>(leftX)];
                        }
                        if (rightX >= 0) {
                            pair += src[sourceRow + static_cast<uint32_t>(rightX)];
                        }
                        sum += pair * weightsX[radiusX - offset];
                    }
                }
            }
            horizontal[localRow * outputWidth + localX] = sum;
        }
    }
    asc_syncthreads();

    if constexpr (FixedKernelSizeY != 0U) {
        const uint32_t outputPairs = (outputRows + 1U) / 2U;
        const uint32_t pairElements = outputPairs * outputWidth;
        for (uint32_t pairElement = threadIdx.x; pairElement < pairElements;
             pairElement += blockDim.x) {
            const uint32_t localX = pairElement % outputWidth;
            const uint32_t localRow = (pairElement / outputWidth) * 2U;
            const uint32_t centerRow = localRow + FixedKernelSizeY / 2U;
            const float center0 = horizontal[centerRow * outputWidth + localX];
            float sum0 = center0 * weightsY[FixedKernelSizeY / 2U];
            if (localRow + 1U < outputRows) {
                const float center1 = horizontal[(centerRow + 1U) * outputWidth + localX];
                float sum1 = center1 * weightsY[FixedKernelSizeY / 2U];
                float previousUpper = center0;
                float previousLower = center1;
#pragma unroll
                for (uint32_t offset = 1U; offset <= FixedKernelSizeY / 2U; ++offset) {
                    const float upper =
                        horizontal[(centerRow - offset) * outputWidth + localX];
                    const float lower =
                        horizontal[(centerRow + 1U + offset) * outputWidth + localX];
                    const float weight = weightsY[FixedKernelSizeY / 2U - offset];
                    sum0 += (upper + previousLower) * weight;
                    sum1 += (previousUpper + lower) * weight;
                    previousUpper = upper;
                    previousLower = lower;
                }
                dst[static_cast<uint64_t>(outputBaseY + localRow) * width +
                    outputBaseX + localX] = sum0;
                dst[static_cast<uint64_t>(outputBaseY + localRow + 1U) * width +
                    outputBaseX + localX] = sum1;
            } else {
#pragma unroll
                for (uint32_t offset = 1U; offset <= FixedKernelSizeY / 2U; ++offset) {
                    const float pair =
                        horizontal[(centerRow - offset) * outputWidth + localX] +
                        horizontal[(centerRow + offset) * outputWidth + localX];
                    sum0 += pair * weightsY[FixedKernelSizeY / 2U - offset];
                }
                dst[static_cast<uint64_t>(outputBaseY + localRow) * width +
                    outputBaseX + localX] = sum0;
            }
        }
    } else if (threadIdx.x < activeThreads) {
        const uint32_t localX = threadIdx.x % outputWidth;
        const uint32_t rowGroup = threadIdx.x / outputWidth;
        for (uint32_t localRow = rowGroup; localRow < outputRows; localRow += rowGroups) {
            float sum = horizontal[(localRow + radiusY) * outputWidth + localX] * weightsY[radiusY];
#pragma unroll 1
            for (uint32_t offset = 1U; offset <= radiusY; ++offset) {
                const float pair = horizontal[(localRow + radiusY - offset) * outputWidth + localX] +
                    horizontal[(localRow + radiusY + offset) * outputWidth + localX];
                sum += pair * weightsY[radiusY - offset];
            }
            dst[static_cast<uint64_t>(outputBaseY + localRow) * width + outputBaseX + localX] = sum;
        }
    }
}

__aicore__ inline void CopyC1RowToUb(
    const AscendC::GlobalTensor<float>& srcGlobal,
    AscendC::LocalTensor<float>& rawTensor,
    int32_t logicalRow,
    uint32_t height,
    uint32_t width,
    uint32_t tileX,
    uint32_t tileWidth,
    uint32_t borderType,
    uint32_t radiusX,
    uint32_t inputWidth,
    uint32_t rawBaseOffset,
    int32_t eventMte2ToV,
    __ubuf__ float* raw,
    __gm__ const float* src)
{
    const uint32_t outputBaseX = tileX * tileWidth;
    const int32_t sourceY = BorderCoordAicore(logicalRow, static_cast<int32_t>(height), borderType);
    const int32_t logicalBaseX =
        static_cast<int32_t>(outputBaseX) - static_cast<int32_t>(radiusX);
    const int32_t logicalEndX = logicalBaseX + static_cast<int32_t>(inputWidth);
    const uint32_t validSourceStart = logicalBaseX > 0 ? static_cast<uint32_t>(logicalBaseX) : 0U;
    const uint32_t validSourceEnd = logicalEndX < static_cast<int32_t>(width) ?
        static_cast<uint32_t>(logicalEndX) : width;
    const uint32_t validColumns = validSourceEnd > validSourceStart ?
        validSourceEnd - validSourceStart : 0U;
    const uint32_t validLocalStart = logicalBaseX < 0 ?
        static_cast<uint32_t>(-logicalBaseX) : 0U;
    const uint32_t validLocalEnd = validLocalStart + validColumns;
    __ubuf__ float* rawBase = raw + rawBaseOffset;
    if (sourceY >= 0 && validColumns != 0U) {
        const uint64_t sourceOffset = static_cast<uint64_t>(sourceY) * width + validSourceStart;
        AscendC::DataCopyExtParams params{
            1U, validColumns * static_cast<uint32_t>(sizeof(float)), 0, 0, 0U};
        AscendC::DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0f};
        AscendC::DataCopyPad(
            rawTensor[rawBaseOffset + validLocalStart], srcGlobal[sourceOffset], params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventMte2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMte2ToV);
        if (validLocalStart != 0U || validLocalEnd != inputWidth) {
            asc_vf_call<GaussianBlurFillC1HaloKernel>(
                dim3{64U, 1U, 1U}, width, tileX, tileWidth, borderType,
                radiusX, inputWidth, validLocalStart, validLocalEnd, rawBase);
        }
        return;
    }
    asc_vf_call<GaussianBlurGatherC1RowKernel>(
        dim3{256U, 1U, 1U}, logicalRow, height, width, tileX, tileWidth, borderType,
        radiusX, inputWidth, validLocalStart, validLocalEnd, rawBase, rawBase, src);
}

__aicore__ inline void StoreC1RowToGm(
    AscendC::GlobalTensor<float>& dstGlobal,
    AscendC::LocalTensor<float>& outputTensor,
    uint32_t outputY,
    uint32_t tileX,
    uint32_t tileWidth,
    uint32_t width,
    uint32_t localOutputOffset)
{
    const uint32_t outputBaseX = tileX * tileWidth;
    const uint32_t activeWidth = outputBaseX + tileWidth <= width ?
        tileWidth : width - outputBaseX;
    AscendC::DataCopyExtParams params{
        1U, activeWidth * static_cast<uint32_t>(sizeof(float)), 0, 0, 0U};
    const uint64_t destinationOffset = static_cast<uint64_t>(outputY) * width + outputBaseX;
    AscendC::DataCopyPad(dstGlobal[destinationOffset], outputTensor[localOutputOffset], params);
}

__aicore__ inline void CopyK31C16RowToUb(
    const AscendC::GlobalTensor<float>& srcGlobal,
    AscendC::LocalTensor<float>& rawTensor,
    int32_t logicalRow,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t tileX,
    uint32_t tileWidth,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t packedChannels,
    uint32_t borderType,
    uint32_t radiusX,
    uint32_t inputWidth,
    int32_t eventMte2ToV,
    __ubuf__ float* raw,
    __gm__ const float* src)
{
    const uint32_t outputBaseX = tileX * tileWidth;
    const int32_t sourceY = BorderCoordAicore(logicalRow, static_cast<int32_t>(height), borderType);
    const int32_t logicalBaseX =
        static_cast<int32_t>(outputBaseX) - static_cast<int32_t>(radiusX);
    const int32_t logicalEndX = logicalBaseX + static_cast<int32_t>(inputWidth);
    const uint32_t validSourceStart = logicalBaseX > 0 ? static_cast<uint32_t>(logicalBaseX) : 0U;
    const uint32_t validSourceEnd = logicalEndX < static_cast<int32_t>(width) ?
        static_cast<uint32_t>(logicalEndX) : width;
    const uint32_t validColumns = validSourceEnd > validSourceStart ?
        validSourceEnd - validSourceStart : 0U;
    const uint32_t validLocalStart = logicalBaseX < 0 ?
        static_cast<uint32_t>(-logicalBaseX) : 0U;
    const uint32_t validLocalEnd = validLocalStart + validColumns;
    if (sourceY < 0 || validColumns == 0U) {
        asc_vf_call<GaussianBlurK31GatherC16RowKernel>(
            dim3{256U, 1U, 1U}, logicalRow, height, width, channels, tileX,
            tileWidth, channelOffset, packedChannels, borderType, radiusX,
            inputWidth, 0U, 0U, raw, src);
        return;
    }

    const uint64_t sourceOffset = ElementOffsetAicore(
        static_cast<uint32_t>(sourceY), validSourceStart, width, channels, channelOffset);
    if (channels == packedChannels) {
        AscendC::DataCopyExtParams params{
            1U, validColumns * packedChannels * static_cast<uint32_t>(sizeof(float)),
            0, 0, 0U};
        AscendC::DataCopyPadExtParams<float> pad{true, 0U, 0U, 0.0f};
        AscendC::DataCopyPad(
            rawTensor[validLocalStart * packedChannels], srcGlobal[sourceOffset], params, pad);
    } else {
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(validColumns),
            static_cast<uint32_t>(outputChannels * sizeof(float)),
            static_cast<int64_t>(channels - outputChannels) * static_cast<int64_t>(sizeof(float)),
            0, 0U};
        AscendC::DataCopyPadExtParams<float> pad{
            true, 0U, static_cast<uint8_t>(packedChannels - outputChannels), 0.0f};
        AscendC::DataCopyPad(
            rawTensor[validLocalStart * packedChannels], srcGlobal[sourceOffset], params, pad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventMte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMte2ToV);
    if (validLocalStart != 0U || validLocalEnd != inputWidth) {
        asc_vf_call<GaussianBlurK31FillC16HaloKernel>(
            dim3{128U, 1U, 1U}, logicalRow, height, width, channels, tileX,
            tileWidth, channelOffset, packedChannels, borderType, radiusX, inputWidth,
            validLocalStart, validLocalEnd, raw);
    }
}

__aicore__ inline void StoreK31C16RowToGm(
    AscendC::GlobalTensor<float>& dstGlobal,
    AscendC::LocalTensor<float>& outputTensor,
    uint32_t outputY,
    uint32_t tileX,
    uint32_t tileWidth,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t outputChannels,
    uint32_t packedChannels,
    uint32_t localOutputOffset)
{
    const uint32_t outputBaseX = tileX * tileWidth;
    const uint32_t activeWidth = outputBaseX + tileWidth <= width ?
        tileWidth : width - outputBaseX;
    const uint64_t destinationOffset =
        ElementOffsetAicore(outputY, outputBaseX, width, channels, channelOffset);
    if (channels == packedChannels) {
        AscendC::DataCopyExtParams params{
            1U, activeWidth * packedChannels * static_cast<uint32_t>(sizeof(float)),
            0, 0, 0U};
        AscendC::DataCopyPad(dstGlobal[destinationOffset], outputTensor[localOutputOffset], params);
    } else {
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(activeWidth),
            static_cast<uint32_t>(outputChannels * sizeof(float)), 0,
            static_cast<int64_t>(channels - outputChannels) * static_cast<int64_t>(sizeof(float)), 0U};
        AscendC::DataCopyPad(dstGlobal[destinationOffset], outputTensor[localOutputOffset], params);
    }
}

static constexpr uint32_t FUSED_DIRECT_SIMT_THREADS = 256U;
static constexpr uint64_t FUSED_DIRECT_SIMT_MAX_OUTPUTS = 512U;
static constexpr uint64_t FUSED_DIRECT_SIMT_MAX_WORK = 262144U;

__simt_vf__ __aicore__ __launch_bounds__(FUSED_DIRECT_SIMT_THREADS) inline void
GaussianBlurFusedDirectSimtVF(
    __gm__ const float* src,
    __gm__ float* dst,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* weightsY,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t kernelSizeX,
    uint32_t kernelSizeY,
    uint32_t borderType,
    uint32_t coreIndex,
    uint32_t coreCount)
{
    const uint64_t total = static_cast<uint64_t>(height) * width * channels;
    const uint64_t threadBase =
        static_cast<uint64_t>(coreIndex) * blockDim.x + threadIdx.x;
    const uint64_t threadStride = static_cast<uint64_t>(coreCount) * blockDim.x;
    const int32_t radiusX = static_cast<int32_t>(kernelSizeX / 2U);
    const int32_t radiusY = static_cast<int32_t>(kernelSizeY / 2U);
    for (uint64_t outputIndex = threadBase; outputIndex < total; outputIndex += threadStride) {
        const uint32_t channel = static_cast<uint32_t>(outputIndex % channels);
        const uint64_t pixelIndex = outputIndex / channels;
        const uint32_t outputX = static_cast<uint32_t>(pixelIndex % width);
        const uint32_t outputY = static_cast<uint32_t>(pixelIndex / width);
        float sum = 0.0f;
#pragma unroll 1
        for (uint32_t kernelY = 0U; kernelY < kernelSizeY; ++kernelY) {
            const int32_t sourceY = BorderCoord(
                static_cast<int32_t>(outputY) + static_cast<int32_t>(kernelY) - radiusY,
                static_cast<int32_t>(height), borderType);
            if (sourceY < 0) {
                continue;
            }
            const float weightY = weightsY[kernelY];
#pragma unroll 1
            for (uint32_t kernelX = 0U; kernelX < kernelSizeX; ++kernelX) {
                const int32_t sourceX = BorderCoord(
                    static_cast<int32_t>(outputX) + static_cast<int32_t>(kernelX) - radiusX,
                    static_cast<int32_t>(width), borderType);
                if (sourceX < 0) {
                    continue;
                }
                const uint64_t sourceOffset = ElementOffset(
                    static_cast<uint32_t>(sourceY), static_cast<uint32_t>(sourceX),
                    width, channels, channel);
                sum += src[sourceOffset] * weightsX[kernelX] * weightY;
            }
        }
        dst[outputIndex] = sum;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(FUSED_DIRECT_SIMT_THREADS) inline void
GaussianBlurFusedSeparableSimtVF(
    __gm__ const float* src,
    __gm__ float* dst,
    __ubuf__ float* horizontal,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* weightsY,
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t kernelSizeX,
    uint32_t kernelSizeY,
    uint32_t borderType)
{
    const uint64_t total = static_cast<uint64_t>(height) * width * channels;
    const int32_t radiusX = static_cast<int32_t>(kernelSizeX / 2U);
    const int32_t radiusY = static_cast<int32_t>(kernelSizeY / 2U);
    for (uint64_t outputIndex = threadIdx.x; outputIndex < total; outputIndex += blockDim.x) {
        const uint32_t channel = static_cast<uint32_t>(outputIndex % channels);
        const uint64_t pixelIndex = outputIndex / channels;
        const uint32_t outputX = static_cast<uint32_t>(pixelIndex % width);
        const uint32_t outputY = static_cast<uint32_t>(pixelIndex / width);
        float sum = 0.0f;
#pragma unroll 1
        for (uint32_t kernelX = 0U; kernelX < kernelSizeX; ++kernelX) {
            const int32_t sourceX = BorderCoord(
                static_cast<int32_t>(outputX) + static_cast<int32_t>(kernelX) - radiusX,
                static_cast<int32_t>(width), borderType);
            if (sourceX < 0) {
                continue;
            }
            const uint64_t sourceOffset = ElementOffset(
                outputY, static_cast<uint32_t>(sourceX), width, channels, channel);
            sum += src[sourceOffset] * weightsX[kernelX];
        }
        horizontal[outputIndex] = sum;
    }
    asc_syncthreads();
    for (uint64_t outputIndex = threadIdx.x; outputIndex < total; outputIndex += blockDim.x) {
        const uint32_t channel = static_cast<uint32_t>(outputIndex % channels);
        const uint64_t pixelIndex = outputIndex / channels;
        const uint32_t outputX = static_cast<uint32_t>(pixelIndex % width);
        const uint32_t outputY = static_cast<uint32_t>(pixelIndex / width);
        float sum = 0.0f;
#pragma unroll 1
        for (uint32_t kernelY = 0U; kernelY < kernelSizeY; ++kernelY) {
            const int32_t sourceY = BorderCoord(
                static_cast<int32_t>(outputY) + static_cast<int32_t>(kernelY) - radiusY,
                static_cast<int32_t>(height), borderType);
            if (sourceY < 0) {
                continue;
            }
            const uint64_t horizontalOffset = ElementOffset(
                static_cast<uint32_t>(sourceY), outputX, width, channels, channel);
            sum += horizontal[horizontalOffset] * weightsY[kernelY];
        }
        dst[outputIndex] = sum;
    }
}

__aicore__ inline void ProcessFusedK31DualSimd(
    GM_ADDR src, GM_ADDR dst, const GaussianBlurTilingData* tilingData)
{
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    AscendC::LocalTensor<float> weightXTensor = ubAllocator.Alloc<float>(WEIGHT_UB_ELEMENTS);
    AscendC::LocalTensor<float> weightYTensor = ubAllocator.Alloc<float>(WEIGHT_UB_ELEMENTS);
    AscendC::LocalTensor<float> workspaceTensor =
        ubAllocator.Alloc<float>(K31_STREAM_WORKSPACE_ELEMENTS);
    AscendC::LocalTensor<float> rawTensor = workspaceTensor[K31_STREAM_RAW_OFFSET];
    AscendC::LocalTensor<float> raw1Tensor = workspaceTensor[K31_STREAM_RAW1_OFFSET];
    AscendC::LocalTensor<float> pendingTensor = workspaceTensor[K31_STREAM_PENDING_OFFSET];
    AscendC::LocalTensor<float> ringTensor = workspaceTensor[K31_STREAM_RING_OFFSET];
    AscendC::LocalTensor<float> outputTensor = workspaceTensor[K31_STREAM_OUTPUT_OFFSET];
    __ubuf__ float* weightsX = reinterpret_cast<__ubuf__ float*>(weightXTensor.GetPhyAddr());
    __ubuf__ float* weightsY = reinterpret_cast<__ubuf__ float*>(weightYTensor.GetPhyAddr());
    __ubuf__ float* workspace =
        reinterpret_cast<__ubuf__ float*>(workspaceTensor.GetPhyAddr());
    __ubuf__ float* raw = workspace + K31_STREAM_RAW_OFFSET;
    __ubuf__ float* raw1 = workspace + K31_STREAM_RAW1_OFFSET;
    __ubuf__ float* c8Pending = workspace + K31_STREAM_PENDING_OFFSET;
    __ubuf__ float* ring = workspace + K31_STREAM_RING_OFFSET;
    __ubuf__ float* output = workspace + K31_STREAM_OUTPUT_OFFSET;
    __ubuf__ float* pending = raw + K31_C1_PENDING_OFFSET;
#pragma unroll
    for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
        weightsX[index] = tilingData->weights[index];
        weightsY[index] = tilingData->weightsY[index];
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();

    const uint64_t directOutputs =
        static_cast<uint64_t>(tilingData->h) * tilingData->w * tilingData->c;
    const uint64_t directWork =
        directOutputs * tilingData->kernelSize * tilingData->kernelSizeY;
    if (tilingData->w < K31_STREAM_TILE_W &&
        directOutputs <= FUSED_DIRECT_SIMT_MAX_OUTPUTS &&
        directWork <= FUSED_DIRECT_SIMT_MAX_WORK) {
        if (AscendC::GetBlockNum() == 1U) {
            asc_vf_call<GaussianBlurFusedSeparableSimtVF>(
                dim3{FUSED_DIRECT_SIMT_THREADS, 1U, 1U},
                reinterpret_cast<__gm__ const float*>(src),
                reinterpret_cast<__gm__ float*>(dst),
                output, weightsX, weightsY,
                tilingData->h, tilingData->w, tilingData->c,
                tilingData->kernelSize, tilingData->kernelSizeY,
                tilingData->borderType);
            return;
        }
        asc_vf_call<GaussianBlurFusedDirectSimtVF>(
            dim3{FUSED_DIRECT_SIMT_THREADS, 1U, 1U},
            reinterpret_cast<__gm__ const float*>(src),
            reinterpret_cast<__gm__ float*>(dst),
            weightsX, weightsY,
            tilingData->h, tilingData->w, tilingData->c,
            tilingData->kernelSize, tilingData->kernelSizeY,
            tilingData->borderType,
            AscendC::GetBlockIdx(), AscendC::GetBlockNum());
        return;
    }

    AscendC::GlobalTensor<float> srcGlobal;
    AscendC::GlobalTensor<float> dstGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst));
    AscendC::TPipe pipe;
    const int32_t eventMte2ToV =
        static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    const int32_t eventVToMte2 =
        static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE2));
    const int32_t eventVToMte3 =
        static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
    const int32_t eventMte3ToV =
        static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::MTE3_V));

    const uint32_t kernelSizeX = tilingData->kernelSize;
    const uint32_t kernelSizeY = tilingData->kernelSizeY;
    const uint32_t radiusX = kernelSizeX / 2U;
    const uint32_t radiusY = kernelSizeY / 2U;
    const bool weightedSpatialTiling = tilingData->reserved[0] == 0U;
    const bool c1WeightedTiling = tilingData->c == 1U && weightedSpatialTiling;
    const bool c8FullCoreTiling = tilingData->c > 1U && tilingData->c <= K31_STREAM_CHANNELS &&
        weightedSpatialTiling;
    const uint32_t spatialTiles = weightedSpatialTiling ? tilingData->totalTiles :
        tilingData->tilesX * tilingData->tilesY;
    if (tilingData->c == 1U) {
        const uint32_t c1TileWidth = kernelSizeX == 31U ?
            K31_C1_K31_TILE_W : K31_STREAM_TILE_W;
        for (uint32_t tileId = AscendC::GetBlockIdx(); tileId < spatialTiles;
             tileId += AscendC::GetBlockNum()) {
            uint32_t tileX = tileId % tilingData->tilesX;
            uint32_t tileY = tileId / tilingData->tilesX;
            uint32_t tilesYForX = tilingData->tilesY;
            if (c1WeightedTiling) {
                MapGaussianBlurC1WeightedTask(
                    tileId, tilingData->totalTiles, tilingData->w, kernelSizeX, kernelSizeY,
                    c1TileWidth, tilingData->tilesX, &tileX, &tileY, &tilesYForX);
            }
            const uint32_t outputBaseY = c1WeightedTiling ?
                static_cast<uint32_t>(static_cast<uint64_t>(tilingData->h) * tileY / tilesYForX) :
                tileY * tilingData->reserved[0];
            const uint32_t outputEndY = c1WeightedTiling ?
                static_cast<uint32_t>(
                    static_cast<uint64_t>(tilingData->h) * (tileY + 1U) / tilesYForX) :
                (outputBaseY + tilingData->reserved[0] < tilingData->h ?
                    outputBaseY + tilingData->reserved[0] : tilingData->h);
            const uint32_t outputRows = outputEndY - outputBaseY;
            const uint32_t outputBaseX = tileX * c1TileWidth;
            const uint32_t activeWidth = outputBaseX + c1TileWidth <= tilingData->w ?
                c1TileWidth : tilingData->w - outputBaseX;
            const uint32_t c1InputWidth = activeWidth + kernelSizeX - 1U;
            const uint32_t c1ValidLocalStart = tileX == 0U ? radiusX : 0U;
            const uint32_t c1RawAlignment =
                (8U - c1ValidLocalStart % 8U) % 8U;
            __ubuf__ float* alignedRaw = raw + c1RawAlignment;
            __ubuf__ float* alignedRaw1 = raw + K31_C1_RAW_ROW_STRIDE + c1RawAlignment;
            const uint32_t tileScratchElements =
                (outputRows + kernelSizeY - 1U) * activeWidth;
            const bool c1TilePreferred = kernelSizeX != 31U || activeWidth <= 16U;
            if (c1TilePreferred && tileScratchElements <= K31_C1_SCRATCH_ELEMENTS) {
                if (kernelSizeX == 7U && kernelSizeY == 31U) {
                    asc_vf_call<GaussianBlurC1TileVF<7U, 31U>>(
                        dim3{256U, 1U, 1U},
                        reinterpret_cast<__gm__ const float*>(src),
                        reinterpret_cast<__gm__ float*>(dst),
                        ring, weightsX, weightsY,
                        tilingData->h, tilingData->w, kernelSizeX, kernelSizeY,
                        tilingData->borderType, outputBaseY, outputRows, outputBaseX, activeWidth);
                } else if (kernelSizeX == 21U && kernelSizeY == 15U) {
                    asc_vf_call<GaussianBlurC1TileVF<21U, 15U>>(
                        dim3{256U, 1U, 1U},
                        reinterpret_cast<__gm__ const float*>(src),
                        reinterpret_cast<__gm__ float*>(dst),
                        ring, weightsX, weightsY,
                        tilingData->h, tilingData->w, kernelSizeX, kernelSizeY,
                        tilingData->borderType, outputBaseY, outputRows, outputBaseX, activeWidth);
                } else if (kernelSizeX == 3U && kernelSizeY == 21U) {
                    asc_vf_call<GaussianBlurC1TileVF<3U, 21U>>(
                        dim3{256U, 1U, 1U},
                        reinterpret_cast<__gm__ const float*>(src),
                        reinterpret_cast<__gm__ float*>(dst),
                        ring, weightsX, weightsY,
                        tilingData->h, tilingData->w, kernelSizeX, kernelSizeY,
                        tilingData->borderType, outputBaseY, outputRows, outputBaseX, activeWidth);
                } else if (kernelSizeX == 15U && kernelSizeY == 3U) {
                    asc_vf_call<GaussianBlurC1TileVF<15U, 3U>>(
                        dim3{256U, 1U, 1U},
                        reinterpret_cast<__gm__ const float*>(src),
                        reinterpret_cast<__gm__ float*>(dst),
                        ring, weightsX, weightsY,
                        tilingData->h, tilingData->w, kernelSizeX, kernelSizeY,
                        tilingData->borderType, outputBaseY, outputRows, outputBaseX, activeWidth);
                } else if (kernelSizeX == 11U && kernelSizeY == 3U) {
                    asc_vf_call<GaussianBlurC1TileVF<11U, 3U>>(
                        dim3{256U, 1U, 1U},
                        reinterpret_cast<__gm__ const float*>(src),
                        reinterpret_cast<__gm__ float*>(dst),
                        ring, weightsX, weightsY,
                        tilingData->h, tilingData->w, kernelSizeX, kernelSizeY,
                        tilingData->borderType, outputBaseY, outputRows, outputBaseX, activeWidth);
                } else {
                    asc_vf_call<GaussianBlurC1TileVF<>>(
                        dim3{256U, 1U, 1U},
                        reinterpret_cast<__gm__ const float*>(src),
                        reinterpret_cast<__gm__ float*>(dst),
                        ring, weightsX, weightsY,
                        tilingData->h, tilingData->w, kernelSizeX, kernelSizeY,
                        tilingData->borderType, outputBaseY, outputRows, outputBaseX, activeWidth);
                }
                continue;
            }
            uint32_t ringRow = 0U;
            if (kernelSizeX == 31U) {
                for (; ringRow + 1U < kernelSizeY; ringRow += 2U) {
                    CopyC1RowToUb(
                        srcGlobal, rawTensor,
                        static_cast<int32_t>(outputBaseY + ringRow) - static_cast<int32_t>(radiusY),
                        tilingData->h, tilingData->w, tileX, c1TileWidth, tilingData->borderType,
                        radiusX, c1InputWidth, c1RawAlignment, eventMte2ToV, raw,
                        reinterpret_cast<__gm__ const float*>(src));
                    CopyC1RowToUb(
                        srcGlobal, rawTensor,
                        static_cast<int32_t>(outputBaseY + ringRow + 1U) - static_cast<int32_t>(radiusY),
                        tilingData->h, tilingData->w, tileX, c1TileWidth, tilingData->borderType,
                        radiusX, c1InputWidth, K31_C1_RAW_ROW_STRIDE + c1RawAlignment,
                        eventMte2ToV, raw,
                        reinterpret_cast<__gm__ const float*>(src));
                    asc_vf_call<GaussianBlurHorizontalC1K31DualVF>(
                        dim3{256U, 1U, 1U}, alignedRaw, alignedRaw1,
                        weightsX, activeWidth,
                        ring + ringRow * K31_C1_RING_ROW_ELEMENTS,
                        ring + (ringRow + 1U) * K31_C1_RING_ROW_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                }
            }
            for (; ringRow < kernelSizeY; ++ringRow) {
                CopyC1RowToUb(
                    srcGlobal, rawTensor,
                    static_cast<int32_t>(outputBaseY + ringRow) - static_cast<int32_t>(radiusY),
                    tilingData->h, tilingData->w, tileX, c1TileWidth, tilingData->borderType,
                    radiusX, c1InputWidth, c1RawAlignment, eventMte2ToV, raw,
                    reinterpret_cast<__gm__ const float*>(src));
                RunGaussianBlurHorizontalC1(
                    alignedRaw, weightsX, radiusX, activeWidth,
                    ring + ringRow * K31_C1_RING_ROW_ELEMENTS);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
            }

            uint32_t centerSlot = radiusY;
            uint32_t replaceSlot = 0U;
            // The pending horizontal row lives in the unused aligned tail of rawTensor. It is
            // disjoint from both raw/staging rows and from outputTensor, which remains the
            // exclusive source of the MTE3 store until MTE3_V completes.
            const bool dualRolling = kernelSizeY == 5U && outputRows >= 3U;
            bool pendingHorizontalRow = false;
            for (uint32_t batchBaseY = 0U; batchBaseY < outputRows;
                 batchBaseY += K31_C1_OUTPUT_BATCH_ROWS) {
                const uint32_t batchRows = batchBaseY + K31_C1_OUTPUT_BATCH_ROWS <= outputRows ?
                    K31_C1_OUTPUT_BATCH_ROWS : outputRows - batchBaseY;
                uint32_t batchRow = 0U;
                while (batchRow < batchRows) {
                    const uint32_t localRow = batchBaseY + batchRow;
                    RunGaussianBlurVerticalC1(
                        ring, weightsY, kernelSizeY, radiusY, centerSlot, activeWidth,
                        output + batchRow * K31_C1_RING_ROW_ELEMENTS);
                    if (localRow + 1U < outputRows) {
                        if (dualRolling && pendingHorizontalRow) {
                            asc_vf_call<CommitGaussianBlurC1RowVF>(
                                dim3{256U, 1U, 1U}, pending,
                                ring + replaceSlot * K31_C1_RING_ROW_ELEMENTS, activeWidth);
                            pendingHorizontalRow = false;
                        } else {
                            CopyC1RowToUb(
                                srcGlobal, rawTensor,
                                static_cast<int32_t>(outputBaseY + localRow + radiusY + 1U),
                                tilingData->h, tilingData->w, tileX, c1TileWidth,
                                tilingData->borderType,
                                radiusX, c1InputWidth, c1RawAlignment, eventMte2ToV, raw,
                                reinterpret_cast<__gm__ const float*>(src));
                            if (dualRolling && localRow + 2U < outputRows) {
                                CopyC1RowToUb(
                                    srcGlobal, rawTensor,
                                    static_cast<int32_t>(outputBaseY + localRow + radiusY + 2U),
                                    tilingData->h, tilingData->w, tileX, c1TileWidth,
                                    tilingData->borderType,
                                    radiusX, c1InputWidth,
                                    K31_C1_RAW_ROW_STRIDE + c1RawAlignment,
                                    eventMte2ToV, raw,
                                    reinterpret_cast<__gm__ const float*>(src));
                                asc_vf_call<GaussianBlurHorizontalC1K31DualVF>(
                                    dim3{256U, 1U, 1U}, alignedRaw, alignedRaw1,
                                    weightsX, activeWidth,
                                    ring + replaceSlot * K31_C1_RING_ROW_ELEMENTS,
                                    pending);
                                pendingHorizontalRow = true;
                            } else {
                                RunGaussianBlurHorizontalC1(
                                    alignedRaw, weightsX, radiusX, activeWidth,
                                    ring + replaceSlot * K31_C1_RING_ROW_ELEMENTS);
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                    }
                    centerSlot = centerSlot + 1U < kernelSizeY ? centerSlot + 1U : 0U;
                    replaceSlot = replaceSlot + 1U < kernelSizeY ? replaceSlot + 1U : 0U;
                    ++batchRow;
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
                for (uint32_t batchRow = 0U; batchRow < batchRows; ++batchRow) {
                    StoreC1RowToGm(
                        dstGlobal, outputTensor, outputBaseY + batchBaseY + batchRow,
                        tileX, c1TileWidth, tilingData->w,
                        batchRow * K31_C1_RING_ROW_ELEMENTS);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMte3ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMte3ToV);
            }
        }
        return;
    }
    for (uint32_t tileId = AscendC::GetBlockIdx(); tileId < tilingData->totalTiles;
         tileId += AscendC::GetBlockNum()) {
        const uint32_t spatialTile = tileId % spatialTiles;
        uint32_t tileX = spatialTile % tilingData->tilesX;
        uint32_t tileY = spatialTile / tilingData->tilesX;
        uint32_t tilesYForX = tilingData->tilesY;
        if (c8FullCoreTiling) {
            MapGaussianBlurEvenSpatialTask(
                spatialTile, spatialTiles, tilingData->tilesX,
                &tileX, &tileY, &tilesYForX);
        }
        const uint32_t tileHeight = tilingData->reserved[0];
        const uint32_t outputBaseY = c8FullCoreTiling ?
            static_cast<uint32_t>(static_cast<uint64_t>(tilingData->h) * tileY / tilesYForX) :
            tileY * tileHeight;
        const uint32_t outputEndY = c8FullCoreTiling ?
            static_cast<uint32_t>(
                static_cast<uint64_t>(tilingData->h) * (tileY + 1U) / tilesYForX) :
            (outputBaseY + tileHeight < tilingData->h ? outputBaseY + tileHeight : tilingData->h);
        const uint32_t outputRows = outputEndY - outputBaseY;
        const uint32_t channelOffset = (tileId / spatialTiles) * K31_STREAM_CHANNELS;
        if (channelOffset >= tilingData->c) continue;
        const uint32_t outputChannels = channelOffset + K31_STREAM_CHANNELS <= tilingData->c ?
            K31_STREAM_CHANNELS : tilingData->c - channelOffset;
        const uint32_t packedChannels = K31_STREAM_CHANNELS;
        const uint32_t streamTileWidth = tilingData->reserved[1] == 0U ?
            K31_STREAM_TILE_W : tilingData->reserved[1];
        const uint32_t outputBaseX = tileX * streamTileWidth;
        const uint32_t activeWidth = outputBaseX + streamTileWidth <= tilingData->w ?
            streamTileWidth : tilingData->w - outputBaseX;
        const uint32_t activeElements = activeWidth * packedChannels;
        const uint32_t inputWidth = activeWidth + kernelSizeX - 1U;
        const bool fullVectorTile = activeElements + 63U >= K31_STREAM_RING_ROW_ELEMENTS;
        for (uint32_t ringRow = 0U; ringRow < kernelSizeY; ++ringRow) {
            CopyK31C16RowToUb(
                srcGlobal, rawTensor,
                static_cast<int32_t>(outputBaseY + ringRow) - static_cast<int32_t>(radiusY),
                tilingData->h, tilingData->w, tilingData->c, tileX, streamTileWidth, channelOffset,
                outputChannels, packedChannels, tilingData->borderType, radiusX, inputWidth,
                eventMte2ToV, raw,
                reinterpret_cast<__gm__ const float*>(src));
            if (kernelSizeX == 31U) {
                asc_vf_call<GaussianBlurK31HorizontalC16RowVF>(
                    raw, weightsX, packedChannels, activeElements,
                    ring + ringRow * K31_STREAM_RING_ROW_ELEMENTS);
            } else {
                asc_vf_call<GaussianBlurHorizontalC8RuntimeVF>(
                    raw, weightsX, radiusX, activeElements,
                    ring + ringRow * K31_STREAM_RING_ROW_ELEMENTS);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
        }

        uint32_t centerSlot = radiusY;
        uint32_t replaceSlot = 0U;
        for (uint32_t batchBaseY = 0U; batchBaseY < outputRows;
             batchBaseY += K31_STREAM_OUTPUT_BATCH_ROWS) {
            const uint32_t batchRows = batchBaseY + K31_STREAM_OUTPUT_BATCH_ROWS <= outputRows ?
                K31_STREAM_OUTPUT_BATCH_ROWS : outputRows - batchBaseY;
            uint32_t batchRow = 0U;
            while (batchRow < batchRows) {
                const uint32_t localRow = batchBaseY + batchRow;
                __ubuf__ float* batchOutput =
                    output + batchRow * K31_STREAM_RING_ROW_ELEMENTS;
                const bool adjacentPair = kernelSizeY == 11U && batchRow + 1U < batchRows;
                if (adjacentPair) {
                    CopyK31C16RowToUb(
                        srcGlobal, raw1Tensor,
                        static_cast<int32_t>(outputBaseY + localRow + radiusY + 1U),
                        tilingData->h, tilingData->w, tilingData->c,
                        tileX, streamTileWidth, channelOffset,
                        outputChannels, packedChannels, tilingData->borderType, radiusX, inputWidth,
                        eventMte2ToV, raw1,
                        reinterpret_cast<__gm__ const float*>(src));
                    if (kernelSizeX == 31U) {
                        asc_vf_call<GaussianBlurK31HorizontalC16RowVF>(
                            raw1, weightsX, packedChannels, activeElements, c8Pending);
                    } else {
                        asc_vf_call<GaussianBlurHorizontalC8RuntimeVF>(
                            raw1, weightsX, radiusX, activeElements, c8Pending);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                    asc_vf_call<GaussianBlurVerticalC8AdjacentPairVF>(
                        ring, weightsY, kernelSizeY, replaceSlot, activeElements,
                        c8Pending, batchOutput,
                        batchOutput + K31_STREAM_RING_ROW_ELEMENTS);
                    centerSlot = centerSlot + 1U < kernelSizeY ? centerSlot + 1U : 0U;
                    replaceSlot = replaceSlot + 1U < kernelSizeY ? replaceSlot + 1U : 0U;

                    if (localRow + 2U < outputRows) {
                        CopyK31C16RowToUb(
                            srcGlobal, rawTensor,
                            static_cast<int32_t>(outputBaseY + localRow + radiusY + 2U),
                            tilingData->h, tilingData->w, tilingData->c,
                            tileX, streamTileWidth, channelOffset,
                            outputChannels, packedChannels, tilingData->borderType,
                            radiusX, inputWidth,
                            eventMte2ToV, raw,
                            reinterpret_cast<__gm__ const float*>(src));
                        if (kernelSizeX == 31U) {
                            asc_vf_call<GaussianBlurK31HorizontalC16RowVF>(
                                raw, weightsX, packedChannels, activeElements,
                                ring + replaceSlot * K31_STREAM_RING_ROW_ELEMENTS);
                        } else {
                            asc_vf_call<GaussianBlurHorizontalC8RuntimeVF>(
                                raw, weightsX, radiusX, activeElements,
                                ring + replaceSlot * K31_STREAM_RING_ROW_ELEMENTS);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                        centerSlot = centerSlot + 1U < kernelSizeY ? centerSlot + 1U : 0U;
                        replaceSlot = replaceSlot + 1U < kernelSizeY ? replaceSlot + 1U : 0U;
                    }
                    batchRow += 2U;
                    continue;
                }
                if (kernelSizeY == 31U && fullVectorTile) {
                    asc_vf_call<GaussianBlurK31VerticalC16RingVF>(
                        ring, weightsY, centerSlot, batchOutput);
                } else {
                    asc_vf_call<GaussianBlurVerticalC8RuntimeVF>(
                        ring, weightsY, kernelSizeY, radiusY, centerSlot,
                        activeElements, batchOutput);
                }

                if (localRow + 1U < outputRows) {
                    CopyK31C16RowToUb(
                        srcGlobal, rawTensor,
                        static_cast<int32_t>(outputBaseY + localRow + radiusY + 1U),
                        tilingData->h, tilingData->w, tilingData->c,
                        tileX, streamTileWidth, channelOffset,
                        outputChannels, packedChannels, tilingData->borderType,
                        radiusX, inputWidth,
                        eventMte2ToV, raw,
                        reinterpret_cast<__gm__ const float*>(src));
                    if (kernelSizeX == 31U) {
                        asc_vf_call<GaussianBlurK31HorizontalC16RowVF>(
                            raw, weightsX, packedChannels, activeElements,
                            ring + replaceSlot * K31_STREAM_RING_ROW_ELEMENTS);
                    } else {
                        asc_vf_call<GaussianBlurHorizontalC8RuntimeVF>(
                            raw, weightsX, radiusX, activeElements,
                            ring + replaceSlot * K31_STREAM_RING_ROW_ELEMENTS);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
                }
                centerSlot = centerSlot + 1U < kernelSizeY ? centerSlot + 1U : 0U;
                replaceSlot = replaceSlot + 1U < kernelSizeY ? replaceSlot + 1U : 0U;
                ++batchRow;
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
            for (uint32_t batchRow = 0U; batchRow < batchRows; ++batchRow) {
                StoreK31C16RowToGm(
                    dstGlobal, outputTensor, outputBaseY + batchBaseY + batchRow,
                    tileX, streamTileWidth, tilingData->w, tilingData->c,
                    channelOffset, outputChannels, packedChannels,
                    batchRow * K31_STREAM_RING_ROW_ELEMENTS);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMte3ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMte3ToV);
        }
    }
}
#endif

#if !GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS && !defined(GAUSSIAN_BLUR_ROW_ONLY) && !defined(GAUSSIAN_BLUR_COLUMN_ONLY)
static constexpr uint32_t FUSED_TILE_W = 32U;
static constexpr uint32_t FUSED_TILE_H = 20U;
static constexpr uint32_t FUSED_VERTICAL_KERNEL = 5U;
static constexpr uint32_t FUSED_INPUT_H = FUSED_TILE_H + FUSED_VERTICAL_KERNEL - 1U;
static constexpr uint32_t FUSED_MAX_KERNEL_X = 21U;
static constexpr uint32_t FUSED_MAX_INPUT_W = FUSED_TILE_W + FUSED_MAX_KERNEL_X - 1U;
static constexpr uint32_t FUSED_CHANNELS = 16U;

__simt_callee__ inline float ComputeFusedHorizontalSymmetric(
    uint32_t kernelSizeX,
    uint32_t inputWidth,
    uint32_t localX,
    uint32_t localRow,
    uint32_t channel,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* inputTile)
{
    const uint32_t radiusX = kernelSizeX / 2U;
    const uint32_t inputBase =
        (localRow * inputWidth + localX) * FUSED_CHANNELS + channel;
    float sum = inputTile[inputBase + radiusX * FUSED_CHANNELS] * weightsX[radiusX];
    for (uint32_t kernelX = 0U; kernelX < radiusX; ++kernelX) {
        const float pair = inputTile[inputBase + kernelX * FUSED_CHANNELS] +
            inputTile[inputBase + (kernelSizeX - 1U - kernelX) * FUSED_CHANNELS];
        sum += pair * weightsX[kernelX];
    }
    return sum;
}

template <uint32_t KernelSizeX>
__simt_callee__ inline float ComputeFusedHorizontalSymmetricFixed(
    uint32_t localX,
    uint32_t localRow,
    uint32_t channel,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* inputTile)
{
    constexpr uint32_t radiusX = KernelSizeX / 2U;
    constexpr uint32_t inputWidth = FUSED_TILE_W + KernelSizeX - 1U;
    const uint32_t inputBase =
        (localRow * inputWidth + localX) * FUSED_CHANNELS + channel;
    float sum = inputTile[inputBase + radiusX * FUSED_CHANNELS] * weightsX[radiusX];
#pragma unroll
    for (uint32_t kernelX = 0U; kernelX < radiusX; ++kernelX) {
        const float pair = inputTile[inputBase + kernelX * FUSED_CHANNELS] +
            inputTile[inputBase + (KernelSizeX - 1U - kernelX) * FUSED_CHANNELS];
        sum += pair * weightsX[kernelX];
    }
    return sum;
}

__simt_callee__ inline float ComputeFusedVerticalSymmetric(
    uint32_t localX,
    uint32_t localRow,
    uint32_t channel,
    __ubuf__ const float* weightsY,
    __ubuf__ const float* horizontalTile)
{
    const uint32_t rowStride = FUSED_TILE_W * FUSED_CHANNELS;
    const uint32_t base = (localRow * FUSED_TILE_W + localX) * FUSED_CHANNELS + channel;
    const float outer = horizontalTile[base] + horizontalTile[base + 4U * rowStride];
    const float inner = horizontalTile[base + rowStride] + horizontalTile[base + 3U * rowStride];
    return outer * weightsY[0] + inner * weightsY[1] +
        horizontalTile[base + 2U * rowStride] * weightsY[2];
}

__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurFusedGenericC8Kernel(
    uint32_t height,
    uint32_t width,
    uint32_t channels,
    uint32_t totalTiles,
    uint32_t coreIndex,
    uint32_t coreCount,
    uint32_t tilesX,
    uint32_t tilesY,
    uint32_t kernelSizeX,
    uint32_t borderType,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* weightsY,
    __ubuf__ float* inputTile,
    __ubuf__ float* horizontalTile,
    __gm__ const float* src,
    __gm__ float* dst,
    bool edgeOnly)
{
    const uint32_t thread = threadIdx.x;
    const uint32_t spatialTiles = tilesX * tilesY;
    const uint32_t radiusX = (kernelSizeX - 1U) / 2U;
    constexpr uint32_t radiusY = FUSED_VERTICAL_KERNEL / 2U;

    for (uint32_t tileId = coreIndex; tileId < totalTiles; tileId += coreCount) {
        const uint32_t spatialTile = tileId % spatialTiles;
        const uint32_t channelOffset = (tileId / spatialTiles) * FUSED_CHANNELS;
        const uint32_t activeChannels = channelOffset + FUSED_CHANNELS <= channels ?
            FUSED_CHANNELS : channels - channelOffset;
        const uint32_t tileX = spatialTile % tilesX;
        const uint32_t tileY = spatialTile / tilesX;
        const uint32_t outputBaseX = tileX * FUSED_TILE_W;
        const uint32_t outputBaseY = tileY * FUSED_TILE_H;
        const uint32_t inputWidth = FUSED_TILE_W + kernelSizeX - 1U;
        const uint32_t inputElements = FUSED_INPUT_H * inputWidth * FUSED_CHANNELS;
        const bool fullInterior = activeChannels == FUSED_CHANNELS &&
            outputBaseX >= radiusX && outputBaseX + FUSED_TILE_W + radiusX <= width &&
            outputBaseY >= radiusY && outputBaseY + FUSED_TILE_H + radiusY <= height;
        if (edgeOnly && fullInterior) {
            continue;
        }

        for (uint32_t index = thread; index < inputElements; index += GAUSSIAN_BLUR_THREADS) {
            const uint32_t channel = index % FUSED_CHANNELS;
            const uint32_t pixel = index / FUSED_CHANNELS;
            const uint32_t localX = pixel % inputWidth;
            const uint32_t localRow = pixel / inputWidth;
            const int32_t rawX = static_cast<int32_t>(outputBaseX + localX) - static_cast<int32_t>(radiusX);
            const int32_t rawY = static_cast<int32_t>(outputBaseY + localRow) - static_cast<int32_t>(radiusY);
            const int32_t sourceX = BorderCoord(rawX, static_cast<int32_t>(width), borderType);
            const int32_t sourceY = BorderCoord(rawY, static_cast<int32_t>(height), borderType);
            float value = 0.0f;
            if (channel < activeChannels && sourceX >= 0 && sourceY >= 0) {
                const uint64_t sourceOffset =
                    (static_cast<uint64_t>(sourceY) * width + static_cast<uint32_t>(sourceX)) * channels +
                    channelOffset + channel;
                value = src[sourceOffset];
            }
            inputTile[index] = value;
        }
        asc_syncthreads();

        constexpr uint32_t horizontalElements = FUSED_INPUT_H * FUSED_TILE_W * FUSED_CHANNELS;
        for (uint32_t index = thread; index < horizontalElements; index += GAUSSIAN_BLUR_THREADS) {
            const uint32_t channel = index % FUSED_CHANNELS;
            const uint32_t pixel = index / FUSED_CHANNELS;
            const uint32_t localX = pixel % FUSED_TILE_W;
            const uint32_t localRow = pixel / FUSED_TILE_W;
            horizontalTile[index] = ComputeFusedHorizontalSymmetric(
                kernelSizeX, inputWidth, localX, localRow, channel, weightsX, inputTile);
        }
        asc_syncthreads();

        constexpr uint32_t outputElements = FUSED_TILE_H * FUSED_TILE_W * FUSED_CHANNELS;
        for (uint32_t index = thread; index < outputElements; index += GAUSSIAN_BLUR_THREADS) {
            const uint32_t channel = index % FUSED_CHANNELS;
            const uint32_t pixel = index / FUSED_CHANNELS;
            const uint32_t localX = pixel % FUSED_TILE_W;
            const uint32_t localRow = pixel / FUSED_TILE_W;
            const uint32_t outputX = outputBaseX + localX;
            const uint32_t outputY = outputBaseY + localRow;
            if (channel < activeChannels && outputX < width && outputY < height) {
                const float sum = ComputeFusedVerticalSymmetric(
                    localX, localRow, channel, weightsY, horizontalTile);
                const uint64_t outputOffset =
                    (static_cast<uint64_t>(outputY) * width + outputX) * channels + channelOffset + channel;
                dst[outputOffset] = sum;
            }
        }
        asc_syncthreads();
    }
}

__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurFusedInteriorKernel(
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t outputBaseX,
    uint32_t outputBaseY,
    uint32_t kernelSizeX,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* weightsY,
    __ubuf__ const float* inputTile,
    __ubuf__ float* horizontalTile,
    __gm__ float* dst)
{
    const uint32_t thread = threadIdx.x;
    const uint32_t inputWidth = FUSED_TILE_W + kernelSizeX - 1U;
    constexpr uint32_t horizontalElements = FUSED_INPUT_H * FUSED_TILE_W * FUSED_CHANNELS;
    for (uint32_t index = thread; index < horizontalElements; index += GAUSSIAN_BLUR_THREADS) {
        const uint32_t channel = index % FUSED_CHANNELS;
        const uint32_t pixel = index / FUSED_CHANNELS;
        const uint32_t localX = pixel % FUSED_TILE_W;
        const uint32_t localRow = pixel / FUSED_TILE_W;
        horizontalTile[index] = ComputeFusedHorizontalSymmetric(
            kernelSizeX, inputWidth, localX, localRow, channel, weightsX, inputTile);
    }
    asc_syncthreads();

    constexpr uint32_t outputElements = FUSED_TILE_H * FUSED_TILE_W * FUSED_CHANNELS;
    for (uint32_t index = thread; index < outputElements; index += GAUSSIAN_BLUR_THREADS) {
        const uint32_t channel = index % FUSED_CHANNELS;
        const uint32_t pixel = index / FUSED_CHANNELS;
        const uint32_t localX = pixel % FUSED_TILE_W;
        const uint32_t localRow = pixel / FUSED_TILE_W;
        const float sum = ComputeFusedVerticalSymmetric(
            localX, localRow, channel, weightsY, horizontalTile);
        const uint64_t outputOffset =
            (static_cast<uint64_t>(outputBaseY + localRow) * width + outputBaseX + localX) * channels +
            channelOffset + channel;
        dst[outputOffset] = sum;
    }
}

template <uint32_t KernelSizeX>
__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurFusedHorizontalKernel(
    uint32_t rowCount,
    uint32_t horizontalRowOffset,
    __ubuf__ const float* weightsX,
    __ubuf__ const float* inputTile,
    __ubuf__ float* horizontalTile)
{
    const uint32_t thread = threadIdx.x;
    const uint32_t horizontalElements = rowCount * FUSED_TILE_W * FUSED_CHANNELS;
    for (uint32_t index = thread; index < horizontalElements; index += GAUSSIAN_BLUR_THREADS) {
        const uint32_t channel = index % FUSED_CHANNELS;
        const uint32_t pixel = index / FUSED_CHANNELS;
        const uint32_t localX = pixel % FUSED_TILE_W;
        const uint32_t localRow = pixel / FUSED_TILE_W;
        const float sum = ComputeFusedHorizontalSymmetricFixed<KernelSizeX>(
            localX, localRow, channel, weightsX, inputTile);
        const uint32_t horizontalOffset =
            ((horizontalRowOffset + localRow) * FUSED_TILE_W + localX) * FUSED_CHANNELS + channel;
        horizontalTile[horizontalOffset] = sum;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(GAUSSIAN_BLUR_THREADS) inline void GaussianBlurFusedVerticalKernel(
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t outputBaseX,
    uint32_t outputBaseY,
    __ubuf__ const float* weightsY,
    __ubuf__ const float* horizontalTile,
    __gm__ float* dst)
{
    const uint32_t thread = threadIdx.x;
    constexpr uint32_t outputElements = FUSED_TILE_H * FUSED_TILE_W * FUSED_CHANNELS;
    for (uint32_t index = thread; index < outputElements; index += GAUSSIAN_BLUR_THREADS) {
        const uint32_t channel = index % FUSED_CHANNELS;
        const uint32_t pixel = index / FUSED_CHANNELS;
        const uint32_t localX = pixel % FUSED_TILE_W;
        const uint32_t localRow = pixel / FUSED_TILE_W;
        const float sum = ComputeFusedVerticalSymmetric(
            localX, localRow, channel, weightsY, horizontalTile);
        const uint64_t outputOffset =
            (static_cast<uint64_t>(outputBaseY + localRow) * width + outputBaseX + localX) * channels +
            channelOffset + channel;
        dst[outputOffset] = sum;
    }
}

__aicore__ inline bool IsFusedInteriorTile(
    uint32_t tileId, uint32_t spatialTiles, uint32_t height, uint32_t width,
    uint32_t channels, uint32_t tilesX, uint32_t kernelSizeX)
{
    const uint32_t spatialTile = tileId % spatialTiles;
    const uint32_t channelOffset = (tileId / spatialTiles) * FUSED_CHANNELS;
    const uint32_t tileX = spatialTile % tilesX;
    const uint32_t tileY = spatialTile / tilesX;
    const uint32_t outputBaseX = tileX * FUSED_TILE_W;
    const uint32_t outputBaseY = tileY * FUSED_TILE_H;
    const uint32_t radiusX = (kernelSizeX - 1U) / 2U;
    constexpr uint32_t radiusY = FUSED_VERTICAL_KERNEL / 2U;
    return channelOffset + FUSED_CHANNELS <= channels &&
        outputBaseX >= radiusX && outputBaseX + FUSED_TILE_W + radiusX <= width &&
        outputBaseY >= radiusY && outputBaseY + FUSED_TILE_H + radiusY <= height;
}

__aicore__ inline uint32_t FindNextFusedInteriorTile(
    uint32_t candidate, uint32_t totalTiles, uint32_t spatialTiles, uint32_t coreCount,
    uint32_t height, uint32_t width, uint32_t channels, uint32_t tilesX, uint32_t kernelSizeX)
{
    while (candidate < totalTiles &&
           !IsFusedInteriorTile(candidate, spatialTiles, height, width, channels, tilesX, kernelSizeX)) {
        candidate += coreCount;
    }
    return candidate;
}

__aicore__ inline void CopyFusedInteriorToUb(
    const AscendC::GlobalTensor<float>& srcGlobal,
    AscendC::LocalTensor<float>& inputTensor,
    uint32_t sourceBaseX,
    uint32_t sourceBaseY,
    uint32_t width,
    uint32_t channels,
    uint32_t channelOffset,
    uint32_t inputWidth,
    uint32_t sourceRowOffset,
    uint32_t rowCount)
{
    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(inputWidth), FUSED_CHANNELS * sizeof(float),
        static_cast<int64_t>(channels - FUSED_CHANNELS) * static_cast<int64_t>(sizeof(float)),
        static_cast<int64_t>(0), 0U};
    AscendC::DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
    for (uint32_t row = 0U; row < rowCount; ++row) {
        const uint64_t sourceOffset =
            (static_cast<uint64_t>(sourceBaseY + sourceRowOffset + row) * width + sourceBaseX) * channels +
            channelOffset;
        AscendC::DataCopyPad<float, AscendC::PaddingMode::Compact>(
            inputTensor[row * inputWidth * FUSED_CHANNELS], srcGlobal[sourceOffset], params, pad);
    }
}

__aicore__ inline void ProcessFusedGenericC8(
    GM_ADDR src, GM_ADDR dst, const GaussianBlurTilingData* tilingData)
{
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    AscendC::LocalTensor<float> inputTensor = ubAllocator.Alloc<float>(
        FUSED_INPUT_H * FUSED_MAX_INPUT_W * FUSED_CHANNELS);
    AscendC::LocalTensor<float> horizontalTensor = ubAllocator.Alloc<float>(
        FUSED_INPUT_H * FUSED_TILE_W * FUSED_CHANNELS);
    AscendC::LocalTensor<float> weightXTensor = ubAllocator.Alloc<float>(32U);
    AscendC::LocalTensor<float> weightYTensor = ubAllocator.Alloc<float>(32U);
    __ubuf__ float* weightsX = reinterpret_cast<__ubuf__ float*>(weightXTensor.GetPhyAddr());
    __ubuf__ float* weightsY = reinterpret_cast<__ubuf__ float*>(weightYTensor.GetPhyAddr());
    __ubuf__ float* inputTile = reinterpret_cast<__ubuf__ float*>(inputTensor.GetPhyAddr());
    __ubuf__ float* horizontalTile = reinterpret_cast<__ubuf__ float*>(horizontalTensor.GetPhyAddr());
#pragma unroll
    for (uint32_t index = 0U; index < GAUSSIAN_BLUR_KERNEL_MAX_SIZE; ++index) {
        weightsX[index] = tilingData->weights[index];
        weightsY[index] = tilingData->weightsY[index];
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::UB>();
    AscendC::GlobalTensor<float> srcGlobal;
    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    AscendC::TPipe pipe;
    const int32_t eventMte2ToV = static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    const int32_t eventVToMte2 = static_cast<int32_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE2));
    const uint32_t spatialTiles = tilingData->tilesX * tilingData->tilesY;
    const uint32_t coreIndex = AscendC::GetBlockIdx();
    const uint32_t coreCount = AscendC::GetBlockNum();
    uint32_t tileId = FindNextFusedInteriorTile(
        coreIndex, tilingData->totalTiles, spatialTiles, coreCount, tilingData->h, tilingData->w,
        tilingData->c, tilingData->tilesX, tilingData->kernelSize);
    while (tileId < tilingData->totalTiles) {
        const uint32_t spatialTile = tileId % spatialTiles;
        const uint32_t channelOffset = (tileId / spatialTiles) * FUSED_CHANNELS;
        const uint32_t outputBaseX = (spatialTile % tilingData->tilesX) * FUSED_TILE_W;
        const uint32_t outputBaseY = (spatialTile / tilingData->tilesX) * FUSED_TILE_H;
        const uint32_t radiusX = (tilingData->kernelSize - 1U) / 2U;
        constexpr uint32_t radiusY = FUSED_VERTICAL_KERNEL / 2U;
        const uint32_t inputWidth = FUSED_TILE_W + tilingData->kernelSize - 1U;
        // A Compact destination cannot be reused before V_MTE2. Keep every
        // MTE batch inside the experimentally observed 8 KiB window while
        // minimizing the number of MTE/VF synchronization boundaries.
        constexpr uint32_t compactWindowBytes = 8192U;
        const uint32_t compactRowBytes = inputWidth * FUSED_CHANNELS * sizeof(float);
        uint32_t rowsPerGroup = compactWindowBytes / compactRowBytes;
        rowsPerGroup = rowsPerGroup == 0U ? 1U : rowsPerGroup;
        rowsPerGroup = rowsPerGroup < FUSED_INPUT_H ? rowsPerGroup : FUSED_INPUT_H;
        for (uint32_t rowOffset = 0U; rowOffset < FUSED_INPUT_H; rowOffset += rowsPerGroup) {
            const uint32_t rowCount = rowOffset + rowsPerGroup <= FUSED_INPUT_H ?
                rowsPerGroup : FUSED_INPUT_H - rowOffset;
            CopyFusedInteriorToUb(
                srcGlobal, inputTensor, outputBaseX - radiusX, outputBaseY - radiusY,
                tilingData->w, tilingData->c, channelOffset, inputWidth, rowOffset, rowCount);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMte2ToV);
            if (tilingData->kernelSize == 3U) {
                asc_vf_call<GaussianBlurFusedHorizontalKernel<3U>>(
                    dim3{GAUSSIAN_BLUR_THREADS, 1U, 1U}, rowCount, rowOffset,
                    weightsX, inputTile, horizontalTile);
            } else if (tilingData->kernelSize == 5U) {
                asc_vf_call<GaussianBlurFusedHorizontalKernel<5U>>(
                    dim3{GAUSSIAN_BLUR_THREADS, 1U, 1U}, rowCount, rowOffset,
                    weightsX, inputTile, horizontalTile);
            } else if (tilingData->kernelSize == 11U) {
                asc_vf_call<GaussianBlurFusedHorizontalKernel<11U>>(
                    dim3{GAUSSIAN_BLUR_THREADS, 1U, 1U}, rowCount, rowOffset,
                    weightsX, inputTile, horizontalTile);
            } else {
                asc_vf_call<GaussianBlurFusedHorizontalKernel<21U>>(
                    dim3{GAUSSIAN_BLUR_THREADS, 1U, 1U}, rowCount, rowOffset,
                    weightsX, inputTile, horizontalTile);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVToMte2);
        }
        asc_vf_call<GaussianBlurFusedVerticalKernel>(
            dim3{GAUSSIAN_BLUR_THREADS, 1U, 1U}, tilingData->w, tilingData->c, channelOffset,
            outputBaseX, outputBaseY, weightsY, horizontalTile, reinterpret_cast<__gm__ float*>(dst));
        tileId = FindNextFusedInteriorTile(
            tileId + coreCount, tilingData->totalTiles, spatialTiles, coreCount, tilingData->h,
            tilingData->w, tilingData->c, tilingData->tilesX, tilingData->kernelSize);
    }
    asc_vf_call<GaussianBlurFusedGenericC8Kernel>(
        dim3{GAUSSIAN_BLUR_THREADS, 1U, 1U},
        tilingData->h, tilingData->w, tilingData->c, tilingData->totalTiles,
        AscendC::GetBlockIdx(), AscendC::GetBlockNum(), tilingData->tilesX, tilingData->tilesY,
        tilingData->kernelSize, tilingData->borderType, weightsX, weightsY, inputTile, horizontalTile,
        reinterpret_cast<__gm__ const float*>(src), reinterpret_cast<__gm__ float*>(dst), true);
}
#endif

} // namespace GAUSSIAN_BLUR_IMPL_NAMESPACE
