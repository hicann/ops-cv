/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../op_kernel/arch35/rotated_box_encode_tiling_data.h"
#include "../../op_kernel/arch35/rotated_box_encode_struct.h"
#include "rotated_box_encode_tiling_arch35.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "graph/types.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <set>
#include <vector>

namespace optiling {

// ===========================================================================
// Integer helpers (DESIGN §2 / §9 formulas).
// ===========================================================================
static inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
static inline int64_t CeilAlign(int64_t v, int64_t a) { return CeilDiv(v, a) * a; }
static inline int64_t AlignDown(int64_t v, int64_t a) { return (v / a) * a; }

// ===========================================================================
// ComputeBranch0Tiling: Branch-0 (fp16-upcast) tiling computation.
// DESIGN-BRANCH-0.md §2: perBoxBytes=70, alignFactor=128, minDtypeBits=80.
// ===========================================================================
ge::graphStatus ComputeBranch0Tiling(const RotatedBoxEncodeBranch0Inputs& in, RotatedBoxEncodeTilingData& out)
{
    out.dim0 = in.dim0;
    out.N = in.N;
    for (int i = 0; i < BRANCH0_BOX_CHANNELS; ++i) {
        out.weight[i] = in.weight[i];
    }

    if (in.dim0 == 0) {
        out.coreNum = 0;
        return ge::GRAPH_SUCCESS;
    }

    int64_t perBoxBytes = BRANCH0_PER_BOX_BYTES;
    int64_t elemStride = BRANCH0_ELEM_STRIDE;
    int64_t alignFactor = BRANCH0_ALIGN_256_BYTES / std::gcd(BRANCH0_ALIGN_256_BYTES, elemStride);
    int64_t maxBoxNum = static_cast<int64_t>(in.ubSize) / perBoxBytes;
    int64_t ubFormer = AlignDown(maxBoxNum, alignFactor);
    ubFormer = (ubFormer < alignFactor) ? alignFactor : ubFormer;
    out.ubFormer = ubFormer;

    int64_t minDtypeBits = BRANCH0_BOX_CHANNELS * BRANCH0_ELEM_BYTES * 8;
    int64_t availableCores = static_cast<int64_t>(in.coreNumAiv);
    int64_t coreNum = CeilDiv(in.dim0 * minDtypeBits, BRANCH0_MIN_TILING_BITS);
    coreNum = (coreNum < availableCores) ? coreNum : availableCores;
    coreNum = (coreNum < 1) ? 1 : coreNum;

    int64_t blockFormer = CeilAlign(CeilDiv(in.dim0, coreNum), BRANCH0_ELEM_ALIGN_FACTOR);
    int64_t blockNum = CeilDiv(in.dim0, blockFormer);
    out.coreNum = static_cast<int32_t>(coreNum);
    out.blockFormer = blockFormer;
    out.blockNum = blockNum;

    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t blockTail = in.dim0 - (blockNum - 1) * blockFormer;
    int64_t ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;
    out.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    out.ubTailOfFormerBlock = ubTailOfFormerBlock;
    out.ubLoopOfTailBlock = ubLoopOfTailBlock;
    out.ubTailOfTailBlock = ubTailOfTailBlock;

    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// ComputeBranch1Tiling: Branch-1 (fp32-direct) tiling computation.
// DESIGN-BRANCH-1.md §2: perBoxBytes=60, alignFactor=64, minDtypeBits=160.
// ===========================================================================
ge::graphStatus ComputeBranch1Tiling(const RotatedBoxEncodeBranch1Inputs& in, RotatedBoxEncodeTilingData& out)
{
    out.dim0 = in.dim0;
    out.N = in.N;
    for (int i = 0; i < BRANCH1_BOX_CHANNELS; ++i) {
        out.weight[i] = in.weight[i];
    }

    if (in.dim0 == 0) {
        out.coreNum = 0;
        return ge::GRAPH_SUCCESS;
    }

    int64_t perBoxBytes = BRANCH1_PER_BOX_BYTES;
    int64_t elemStride = BRANCH1_ELEM_STRIDE;
    int64_t alignFactor = BRANCH1_ALIGN_256_BYTES / std::gcd(BRANCH1_ALIGN_256_BYTES, elemStride);
    int64_t maxBoxNum = static_cast<int64_t>(in.ubSize) / perBoxBytes;
    int64_t ubFormer = AlignDown(maxBoxNum, alignFactor);
    ubFormer = (ubFormer < alignFactor) ? alignFactor : ubFormer;
    out.ubFormer = ubFormer;

    int64_t minDtypeBits = BRANCH1_BOX_CHANNELS * BRANCH1_ELEM_BYTES * 8;
    int64_t availableCores = static_cast<int64_t>(in.coreNumAiv);
    int64_t coreNum = CeilDiv(in.dim0 * minDtypeBits, BRANCH1_MIN_TILING_BITS);
    coreNum = (coreNum < availableCores) ? coreNum : availableCores;
    coreNum = (coreNum < 1) ? 1 : coreNum;

    int64_t blockFormer = CeilAlign(CeilDiv(in.dim0, coreNum), BRANCH1_ELEM_ALIGN_FACTOR);
    int64_t blockNum = CeilDiv(in.dim0, blockFormer);
    out.coreNum = static_cast<int32_t>(coreNum);
    out.blockFormer = blockFormer;
    out.blockNum = blockNum;

    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t blockTail = in.dim0 - (blockNum - 1) * blockFormer;
    int64_t ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;
    out.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    out.ubTailOfFormerBlock = ubTailOfFormerBlock;
    out.ubLoopOfTailBlock = ubLoopOfTailBlock;
    out.ubTailOfTailBlock = ubTailOfTailBlock;

    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// TilingPrepareForRotatedBoxEncode — compile-time platform info parser.
// ===========================================================================
ge::graphStatus TilingPrepareForRotatedBoxEncode(gert::TilingParseContext* context)
{
    auto* ci = context->GetCompiledInfo<RotatedBoxEncodeCompileInfo>();
    if (ci == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto* platFormInfos = context->GetPlatformInfo();
    if (platFormInfos == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC plat(platFormInfos);
    ci->coreNumAiv = plat.GetCoreNumAiv();
    uint64_t ubSize = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = ubSize;
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// TilingRotatedBoxEncode — runtime tiling entry (DESIGN §9.9).
// ===========================================================================
ge::graphStatus TilingRotatedBoxEncode(gert::TilingContext* ctx)
{
    const auto* ci = ctx->GetCompileInfo<RotatedBoxEncodeCompileInfo>();
    uint32_t coreNumAiv = (ci != nullptr && ci->coreNumAiv > 0 && ci->coreNumAiv <= 1024) ? ci->coreNumAiv : 1;
    uint64_t ubSizeVal = (ci != nullptr && ci->ubSize > 0 && ci->ubSize <= 1024 * 1024) ? ci->ubSize : 0;

    auto* platInfo = ctx->GetPlatformInfo();
    bool isProductionContext = (platInfo != nullptr && platInfo->GetCoreNum() > 0);

    const auto* anchorShape = ctx->GetInputShape(0);
    if (anchorShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& originShape = anchorShape->GetOriginShape();
    int64_t rank = static_cast<int64_t>(originShape.GetDimNum());

    auto setupEmptyTiling = [ctx](ge::DataType dt) {
        auto* td = ctx->GetTilingData<RotatedBoxEncodeTilingData>();
        if (td != nullptr) {
            std::memset(td, 0, sizeof(*td));
            td->dim0 = 0;
            td->N = 0;
            td->coreNum = 0;
        }
        if (dt == ge::DT_FLOAT16) {
            ctx->SetTilingKey(GET_TPL_TILING_KEY(ROTATED_BOX_ENCODE_DTYPE_FP16));
        } else {
            ctx->SetTilingKey(GET_TPL_TILING_KEY(ROTATED_BOX_ENCODE_DTYPE_FP32));
        }
        ctx->SetBlockDim(1);
        size_t* ws = ctx->GetWorkspaceSizes(1);
        if (ws != nullptr) {
            ws[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    };

    const auto* anchorTensor = ctx->GetInputTensor(0);
    const auto* gtTensor = ctx->GetInputTensor(1);
    if (anchorTensor == nullptr || gtTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType inDtype0 = anchorTensor->GetDataType();
    ge::DataType inDtype1 = gtTensor->GetDataType();

    static const std::set<ge::DataType> SUPPORTED_DTYPES = {ge::DT_FLOAT16, ge::DT_FLOAT};
    if (SUPPORTED_DTYPES.find(inDtype0) == SUPPORTED_DTYPES.end()) {
        return ge::GRAPH_FAILED;
    }
    if (inDtype0 != inDtype1) {
        return ge::GRAPH_FAILED;
    }

    ge::Format anchorFmt = anchorTensor->GetStorageFormat();
    if (anchorFmt != ge::FORMAT_ND) {
        return ge::GRAPH_FAILED;
    }

    if (rank != MAX_RANK || originShape.GetDim(1) != BOX_CHANNELS) {
        if (isProductionContext) {
            return setupEmptyTiling(inDtype0);
        }
        return ge::GRAPH_FAILED;
    }
    int64_t B = originShape.GetDim(0);
    int64_t N = originShape.GetDim(2);

    const auto* attrs = ctx->GetAttrs();
    const auto* wList = (attrs != nullptr) ? attrs->GetListFloat(0) : nullptr;
    int64_t wLen = (wList != nullptr) ? static_cast<int64_t>(wList->GetSize()) : BOX_CHANNELS;
    if (wLen != BOX_CHANNELS) {
        return ge::GRAPH_FAILED;
    }

    const auto* gtShapePtr = ctx->GetInputShape(1);
    if (gtShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& gtOriginShape = gtShapePtr->GetOriginShape();
    bool shapeMismatch = false;
    if (gtOriginShape.GetDimNum() != static_cast<size_t>(rank)) {
        shapeMismatch = true;
    } else {
        for (int64_t i = 0; i < rank; ++i) {
            if (originShape.GetDim(static_cast<size_t>(i)) != gtOriginShape.GetDim(static_cast<size_t>(i))) {
                shapeMismatch = true;
                break;
            }
        }
    }
    if (shapeMismatch) {
        if (isProductionContext) {
            return setupEmptyTiling(inDtype0);
        }
        return ge::GRAPH_FAILED;
    }

    int64_t dim0 = B * N;

    float weight[BOX_CHANNELS] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    if (wList != nullptr) {
        const float* wData = wList->GetData();
        if (wData != nullptr) {
            for (int i = 0; i < BOX_CHANNELS; ++i) {
                weight[i] = wData[static_cast<size_t>(i)];
            }
        }
    }

    auto* td = ctx->GetTilingData<RotatedBoxEncodeTilingData>();
    if (td == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (inDtype0 == ge::DT_FLOAT16) {
        ctx->SetTilingKey(GET_TPL_TILING_KEY(ROTATED_BOX_ENCODE_DTYPE_FP16));

        RotatedBoxEncodeBranch0Inputs bin{};
        bin.dim0 = dim0;
        bin.N = N;
        bin.coreNumAiv = coreNumAiv;
        bin.ubSize = ubSizeVal;
        for (int i = 0; i < BOX_CHANNELS; ++i) {
            bin.weight[i] = weight[i];
        }

        std::memset(td, 0, sizeof(*td));
        ComputeBranch0Tiling(bin, *td);

        uint32_t blockDim = (td->coreNum > 0) ? static_cast<uint32_t>(td->coreNum) : (isProductionContext ? 1 : 0);
        ctx->SetBlockDim(blockDim);
        size_t* ws = ctx->GetWorkspaceSizes(1);
        if (ws != nullptr) {
            ws[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }

    ctx->SetTilingKey(GET_TPL_TILING_KEY(ROTATED_BOX_ENCODE_DTYPE_FP32));

    RotatedBoxEncodeBranch1Inputs bin{};
    bin.dim0 = dim0;
    bin.N = N;
    bin.coreNumAiv = coreNumAiv;
    bin.ubSize = ubSizeVal;
    for (int i = 0; i < BOX_CHANNELS; ++i) {
        bin.weight[i] = weight[i];
    }

    std::memset(td, 0, sizeof(*td));
    ComputeBranch1Tiling(bin, *td);

    uint32_t blockDim = (td->coreNum > 0) ? static_cast<uint32_t>(td->coreNum) : (isProductionContext ? 1 : 0);
    ctx->SetBlockDim(blockDim);
    size_t* ws = ctx->GetWorkspaceSizes(1);
    if (ws != nullptr) {
        ws[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotatedBoxEncode)
    .Tiling(TilingRotatedBoxEncode)
    .TilingParse<RotatedBoxEncodeCompileInfo>(TilingPrepareForRotatedBoxEncode);

} // namespace optiling
