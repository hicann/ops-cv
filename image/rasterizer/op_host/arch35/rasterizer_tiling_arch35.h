/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RASTERIZER_TILING_ARCH35_H
#define RASTERIZER_TILING_ARCH35_H

#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"
#include "../../op_kernel/arch35/rasterizer_tiling_struct.h"

namespace optiling {

// Public, context-independent contract used by the Task 19 white-box tests.
// Task 20 will make TilingFunc adapt gert::TilingContext to this input and map
// the result back to blockDim, tiling data and workspace dispatch state.
enum class RasterizerPublicTilingStatus : int32_t {
    kSuccess = 0,
    kNullInput = 1,
    kDtypeNotSupported = 2,
    kShapeMismatch = 3,
    kAttributeValueOutOfRange = 4,
    kPlatformInvalid = 5,
    kOutOfMemory = 6,
    kArithmeticOverflow = 7,
    kNotImplemented = INT32_MIN,
};

enum class RasterizerPublicDataType : uint8_t {
    kFloat32 = 0,
    kInt32 = 1,
    kFloat16 = 2,
    kUnknown = 255,
};

struct RasterizerPublicTensorDesc {
    bool present = false;
    RasterizerPublicDataType dtype = RasterizerPublicDataType::kUnknown;
    uint32_t rank = 0;
    uint64_t dims[3] = {0, 0, 0};
    bool ndContiguous = false;
};

struct RasterizerPublicTilingInputs {
    RasterizerPublicTensorDesc v;
    RasterizerPublicTensorDesc f;
    RasterizerPublicTensorDesc d;
    RasterizerPublicTensorDesc findices;
    RasterizerPublicTensorDesc barycentric;

    int64_t width = 0;
    int64_t height = 0;
    double occlusionTruncation = 0.0;
    int64_t useDepthPrior = 0;

    bool aivCoreQuerySucceeded = false;
    bool ubQuerySucceeded = false;
    bool blockBytesQuerySucceeded = false;
    bool vectorBytesQuerySucceeded = false;
    bool systemWorkspaceQuerySucceeded = false;
    uint64_t availableAivCores = 0;
    uint64_t ubBytes = 0;
    uint64_t blockBytes = 0;
    uint64_t vectorBytes = 0;
    uint64_t systemWorkspaceBytes = 0;

    bool outputAllocationSucceeded = false;
    bool workspaceAllocationSucceeded = false;
    uint64_t runtimeSizeMax = UINT64_MAX;
};

struct RasterizerPublicTilingResult {
    RasterizerPublicTilingStatus status = RasterizerPublicTilingStatus::kNotImplemented;
    bool kernelDispatchAllowed = false;
    uint32_t tilingKey = 0;
    uint64_t totalPixels = 0;
    uint64_t alignedPixels = 0;
    uint64_t pixelAlignment = 0;
    uint32_t vectorAlignment = 0;
    uint64_t serializedBytes = 0;
    uint64_t totalWorkspaceBytes = 0;
    RasterizerTilingData tilingData{};
};

// Testable public computation boundary shared by Host glue and white-box UT.
void ComputeRasterizerPublicTiling(const RasterizerPublicTilingInputs& inputs, RasterizerPublicTilingResult& result);

// Context-independent DESIGN-BRANCH-0 §2/§8 computation boundary. The public
// Host contract validates Tensor descriptors and attributes before adapting the
// dimensions and queried resources consumed by PIXEL_DIRECT.
struct RasterizerBranch0TilingInputs {
    uint64_t numFaces = 0;
    uint64_t height = 0;
    uint64_t width = 0;

    // Ascend 950 public tiling dispatches only the low-stack PixelDirect kernel.
    // Keep false by default so independent branch white-box semantics remain unchanged.
    bool forcePixelDirect = false;

    bool aivCoreQuerySucceeded = false;
    bool ubQuerySucceeded = false;
    bool blockBytesQuerySucceeded = false;
    bool vectorBytesQuerySucceeded = false;
    bool systemWorkspaceQuerySucceeded = false;
    uint64_t availableAivCores = 0;
    uint64_t ubBytes = 0;
    uint64_t blockBytes = 0;
    uint64_t vectorBytes = 0;
    uint64_t systemWorkspaceBytes = 0;

    bool outputAllocationSucceeded = false;
    bool workspaceAllocationSucceeded = false;
    uint64_t runtimeSizeMax = UINT64_MAX;
};

struct RasterizerBranch0TilingResult {
    RasterizerPublicTilingStatus status = RasterizerPublicTilingStatus::kNotImplemented;
    bool branchSelected = false;
    bool kernelDispatchAllowed = false;
    uint32_t tilingKey = 0;
    uint64_t totalPixels = 0;
    uint64_t alignedPixels = 0;
    uint64_t pixelAlignment = 0;
    uint32_t vectorAlignment = 0;
    uint64_t targetCoreNum = 0;
    uint32_t blockDim = 0;
    uint64_t serializedBytes = 0;
    uint64_t totalWorkspaceBytes = 0;
    RasterizerTilingData tilingData{};
};

// Production PIXEL_DIRECT formula used by the public Host path and white-box UT.
void ComputeRasterizerBranch0Tiling(const RasterizerBranch0TilingInputs& inputs, RasterizerBranch0TilingResult& result);

// Context-independent DESIGN-BRANCH-1 §2/§8 computation boundary used by
// both the public Host path and the independent FACE_GROUP white-box UT.
struct RasterizerBranch1TilingInputs {
    uint64_t numFaces = 0;
    uint64_t height = 0;
    uint64_t width = 0;

    bool aivCoreQuerySucceeded = false;
    bool ubQuerySucceeded = false;
    bool blockBytesQuerySucceeded = false;
    bool vectorBytesQuerySucceeded = false;
    bool systemWorkspaceQuerySucceeded = false;
    uint64_t availableAivCores = 0;
    uint64_t ubBytes = 0;
    uint64_t blockBytes = 0;
    uint64_t vectorBytes = 0;
    uint64_t systemWorkspaceBytes = 0;

    bool outputAllocationSucceeded = false;
    bool workspaceAllocationSucceeded = false;
    bool scheduleModeSetSucceeded = false;
    uint64_t runtimeSizeMax = UINT64_MAX;
};

struct RasterizerBranch1TilingResult {
    RasterizerPublicTilingStatus status = RasterizerPublicTilingStatus::kNotImplemented;
    bool branchSelected = false;
    bool kernelDispatchAllowed = false;
    uint32_t tilingKey = 0;
    uint64_t totalPixels = 0;
    uint64_t alignedPixels = 0;
    uint64_t pixelAlignment = 0;
    uint32_t vectorAlignment = 0;
    uint64_t faceCoreNum = 0;
    uint64_t faceQuotient = 0;
    uint64_t faceRemainder = 0;
    uint64_t mergeCoreNum = 0;
    uint32_t blockDim = 0;
    uint32_t scheduleMode = 0;
    uint64_t ubFaceBytes = 0;
    uint64_t maxTilePixels = 0;
    uint64_t serializedBytes = 0;
    uint64_t totalWorkspaceBytes = 0;
    RasterizerTilingData tilingData{};
};

void ComputeRasterizerBranch1Tiling(const RasterizerBranch1TilingInputs& inputs, RasterizerBranch1TilingResult& result);

} // namespace optiling

#endif // RASTERIZER_TILING_ARCH35_H
