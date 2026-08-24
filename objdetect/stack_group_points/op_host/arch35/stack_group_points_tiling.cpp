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
 * \file stack_group_points_tiling.cpp
 * \brief Tiling implementation for stack_group_points operator
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "stack_group_points_tiling.h"

namespace optiling {

// ========== 常量 ==========
constexpr int64_t PER_CORE_MIN = 1024;               // 每核最小元素数，对齐到 32
constexpr uint32_t DCACHE_SIZE = 32 * 1024;          // DCache 预留 32KB
constexpr uint32_t STATIC_UB_ESTIMATE = 0;           // 无静态 UB
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024; // 系统工作空间 16MB

// ========== 输入索引常量（SE §2.2） ==========
static constexpr int32_t kFeaturesIdx = 0;
static constexpr int32_t kFeaturesBatchCntIdx = 1;
static constexpr int32_t kIndicesIdx = 2;
static constexpr int32_t kIndicesBatchCntIdx = 3;

// ========== 维度索引常量 ==========
static constexpr int32_t kFeaturesCIdx = 1;      // features dim[1] = C
static constexpr int32_t kIndicesNsampleIdx = 1; // indices dim[1] = nsample
static constexpr int32_t kBatchCntBIdx = 0;      // batch_cnt dim[0] = B

// ========== Tiling Key 常量（与 op_kernel/arch35/stack_group_points_tiling_key.h 一致） ==========
static constexpr int32_t kTilingKeyFp32 = 0;
static constexpr int32_t kTilingKeyFp16 = 1;

struct StackGroupPointsCompileInfo {};

// ========== ValidateDtype（SE §2.2 + §2.6） ==========
static ge::graphStatus ValidateDtype(gert::TilingContext* context)
{
    auto opName = context->GetNodeName();
    auto featuresDesc = context->GetInputDesc(kFeaturesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresDesc);
    auto featuresDtype = featuresDesc->GetDataType();
    OP_CHECK_IF(featuresDtype != ge::DT_FLOAT16 && featuresDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(
                    opName, "features", std::to_string(static_cast<int32_t>(featuresDtype)).c_str(), "float16/float32"),
                return ge::GRAPH_FAILED);

    auto fbcDesc = context->GetInputDesc(kFeaturesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, fbcDesc);
    OP_CHECK_IF(
        fbcDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(opName, "features_batch_cnt",
                                  std::to_string(static_cast<int32_t>(fbcDesc->GetDataType())).c_str(), "int32"),
        return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(kIndicesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_IF(
        indicesDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(opName, "indices",
                                  std::to_string(static_cast<int32_t>(indicesDesc->GetDataType())).c_str(), "int32"),
        return ge::GRAPH_FAILED);

    auto ibcDesc = context->GetInputDesc(kIndicesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, ibcDesc);
    OP_CHECK_IF(
        ibcDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(opName, "indices_batch_cnt",
                                  std::to_string(static_cast<int32_t>(ibcDesc->GetDataType())).c_str(), "int32"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ========== ValidateShape（SE §2.5） ==========
static ge::graphStatus ValidateShape(gert::TilingContext* context)
{
    auto opName = context->GetNodeName();
    auto featuresShape = context->GetInputShape(kFeaturesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);
    OP_CHECK_IF(
        featuresShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "features", (std::to_string(featuresShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    auto fbcShape = context->GetInputShape(kFeaturesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, fbcShape);
    OP_CHECK_IF(
        fbcShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "features_batch_cnt",
                                     (std::to_string(fbcShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);

    auto indicesShape = context->GetInputShape(kIndicesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    OP_CHECK_IF(
        indicesShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "indices",
                                     (std::to_string(indicesShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    auto ibcShape = context->GetInputShape(kIndicesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, ibcShape);
    OP_CHECK_IF(
        ibcShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "indices_batch_cnt",
                                     (std::to_string(ibcShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);

    // 维度值边界校验（防止 kernel 中除零：index % nsample, index / nsample % c）
    OP_CHECK_IF(
        featuresShape->GetStorageShape().GetDim(kFeaturesCIdx) <= 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "features",
            ("features dim[1]=" + std::to_string(featuresShape->GetStorageShape().GetDim(kFeaturesCIdx))).c_str(),
            "features dim[1] (C) must be > 0 to avoid division by zero in kernel"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        indicesShape->GetStorageShape().GetDim(kIndicesNsampleIdx) <= 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "indices",
            ("indices dim[1]=" + std::to_string(indicesShape->GetStorageShape().GetDim(kIndicesNsampleIdx))).c_str(),
            "indices dim[1] (nsample) must be > 0 to avoid division by zero in kernel"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ibcShape->GetStorageShape().GetDim(kBatchCntBIdx) <= 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "indices_batch_cnt",
            ("indices_batch_cnt dim[0]=" + std::to_string(ibcShape->GetStorageShape().GetDim(kBatchCntBIdx))).c_str(),
            "indices_batch_cnt dim[0] (B) must be > 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ========== ValidateInputs 调度器 ==========
static ge::graphStatus ValidateInputs(gert::TilingContext* context)
{
    // SE §2.2 + §2.6
    OP_CHECK_IF(ValidateDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    // SE §2.5
    OP_CHECK_IF(ValidateShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateShape failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== GetPlatformInfo ==========
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== FillTilingData ==========
static void FillTilingData(StackGroupPointsTilingData* tiling, int64_t m, int64_t c, int64_t nsample, int64_t b,
                           int64_t n, int64_t totalElements, int64_t needCoreNum)
{
    tiling->m = m;
    tiling->c = c;
    tiling->nsample = nsample;
    tiling->b = b;
    tiling->n = n;
    tiling->totalElements = totalElements;
    tiling->needCoreNum = needCoreNum;
}

// ========== DumpTilingData (DFX) ==========
static void DumpTilingData(gert::TilingContext* context, const StackGroupPointsTilingData* tiling)
{
    OP_LOGD(context,
            "StackGroupPointsTilingData: m=%ld, c=%ld, nsample=%ld, b=%ld, n=%ld, totalElements=%ld, needCoreNum=%ld",
            tiling->m, tiling->c, tiling->nsample, tiling->b, tiling->n, tiling->totalElements, tiling->needCoreNum);
}

// ========== GetWorkspaceSize ==========
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    // 仅分配系统 workspace（无用户 workspace）
    currentWorkspace[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

// ========== StackGroupPointsTilingFunc 主函数 ==========
static ge::graphStatus StackGroupPointsTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    // 2. 校验输入（SE §2.2 + §2.5 + §2.6）
    OP_CHECK_IF(ValidateInputs(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateInputs failed"),
                return ge::GRAPH_FAILED);

    // 3. 提取 shape 维度
    auto featuresShape = context->GetInputShape(kFeaturesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);
    auto indicesShape = context->GetInputShape(kIndicesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    auto ibcShape = context->GetInputShape(kIndicesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, ibcShape);

    int64_t n = featuresShape->GetStorageShape().GetDim(0);      // N
    int64_t c = featuresShape->GetStorageShape().GetDim(1);      // C
    int64_t m = indicesShape->GetStorageShape().GetDim(0);       // M
    int64_t nsample = indicesShape->GetStorageShape().GetDim(1); // nsample
    int64_t b = ibcShape->GetStorageShape().GetDim(0);           // B
    auto fbcShape = context->GetInputShape(kFeaturesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, fbcShape);
    int64_t fbcB = fbcShape->GetStorageShape().GetDim(0);
    std::string fbcShapeMsg = "features_batch_cnt dim0=" + std::to_string(fbcB) +
                              ", indices_batch_cnt dim0=" + std::to_string(b);
    OP_CHECK_IF(fbcB != b,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(),
                                                       "features_batch_cnt and indices_batch_cnt", fbcShapeMsg.c_str(),
                                                       "features_batch_cnt dim0 must equal indices_batch_cnt dim0"),
                return ge::GRAPH_FAILED);

    int64_t totalElements = m * c * nsample;
    int64_t perCoreElements = (totalElements + coreNum - 1) / coreNum;
    if (perCoreElements < PER_CORE_MIN) {
        perCoreElements = PER_CORE_MIN;
    }
    int64_t needCoreNum = (totalElements + perCoreElements - 1) / perCoreElements;
    if (needCoreNum > coreNum) {
        needCoreNum = coreNum;
    }
    if (needCoreNum <= 0) {
        needCoreNum = 1;
    }

    auto tiling = context->GetTilingData<StackGroupPointsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(StackGroupPointsTilingData), 0, sizeof(StackGroupPointsTilingData)) != EOK,
                OP_LOGE(context, "memset_s failed"), return ge::GRAPH_FAILED);
    FillTilingData(tiling, m, c, nsample, b, n, totalElements, needCoreNum);

    // 6. DFX 日志
    DumpTilingData(context, tiling);

    // 7. 设置 workspace（无条件调用 GetWorkspaceSizes(1)）
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize failed"),
                return ge::GRAPH_FAILED);

    // 8. 设置 BlockDim、LocalMemory
    context->SetBlockDim(static_cast<int32_t>(needCoreNum));
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE),
                OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize), return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu, DCACHE_SIZE=%u, STATIC_UB_ESTIMATE=%u", ubSize,
                        DCACHE_SIZE, STATIC_UB_ESTIMATE),
                return ge::GRAPH_FAILED);

    // 9. 设置 TilingKey（FP32=0, FP16=1，与 tiling_key.h 定义一致）
    auto dtypeDesc = context->GetInputDesc(kFeaturesIdx);
    auto dtype = dtypeDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(kTilingKeyFp16);
    } else if (dtype == ge::DT_FLOAT) {
        context->SetTilingKey(kTilingKeyFp32);
    }

    return ge::GRAPH_SUCCESS;
}

// ========== TilingParse 回调 ==========
static ge::graphStatus TilingParseForStackGroupPoints([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// ========== 注册三段式 ==========
IMPL_OP_OPTILING(StackGroupPoints)
    .Tiling(StackGroupPointsTilingFunc)
    .TilingParse<StackGroupPointsCompileInfo>(TilingParseForStackGroupPoints);

} // namespace optiling
