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
 * \file blend_face_bg_part_two_tiling.cpp
 * \brief BlendFaceBgPartTwo arch35 host tiling 实现。
 */

#include <algorithm>
#include <cstring>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "graph/utils/type_utils.h"
#include "../../op_kernel/arch35/blend_face_bg_part_two_tiling_data.h"
#include "../../op_kernel/arch35/blend_face_bg_part_two_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

namespace {
constexpr size_t kInputAccFaceIdx = 0;
constexpr size_t kInputAccMaskIdx = 1;
constexpr size_t kInputMaxMaskIdx = 2;
constexpr size_t kInputBgImgIdx = 3; // bg_img 输入下标（acc_face/acc_mask/max_mask/bg_img）
constexpr size_t kAttrEpsilonIdx = 0;
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;

// EleWise 切分常量（承接 regbase elewise tiling.md）
constexpr int64_t MIN_TILING_BITS = 32768; // 每核至少 4KB（bits）
constexpr int64_t ELEM_ALIGN_FACTOR = 512; // 多核切分元素对齐因子
constexpr int64_t ALIGN_256 = 256;         // UB 对齐字节数
constexpr int64_t FP32_BYTES = 4;          // fp32 element size
constexpr int64_t UINT8_BYTES = 1;         // uint8 element size
constexpr int64_t BUFFER_NUM_DB = 2;
// bufferDivisor：双缓冲下每元素 UB 占用（3 路 fp32 + bg + 1 输出 fp32）
//   KEY_FP32 ：2*(4+4+4+4+4) = 40 bytes/元素
//   KEY_UINT8：2*(4+4+4+1+4) = 34 bytes/元素
constexpr int64_t BUFFER_DIVISOR_FP32 = BUFFER_NUM_DB * (4 * FP32_BYTES + FP32_BYTES);                // 40
constexpr int64_t BUFFER_DIVISOR_UINT8 = BUFFER_NUM_DB * (3 * FP32_BYTES + UINT8_BYTES + FP32_BYTES); // 34
constexpr float DEFAULT_EPSILON = 1.0e-12f;
constexpr size_t kExpectRank = 3; // (H,W,C)
} // namespace

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // 纯 vector 逐元素算子 → 取 VectorCore 核数（DAV_3510 CubeCore:VectorCore = 1:2）
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static float GetEpsilonAttr(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return DEFAULT_EPSILON;
    }
    const float* epsP = attrs->GetFloat(kAttrEpsilonIdx);
    return (epsP != nullptr) ? *epsP : DEFAULT_EPSILON;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BlendFaceBgPartTwoTilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter BlendFaceBgPartTwoTilingFunc");

    // 0) 拦截防线（前置至平台信息之前，确保 tiling 被调用时第一时间拦截）
    //    GEIR 通路中 CANN 框架不调用自定义 InferShape/VerifyFunc，tiling 是唯一算子侧防线。
    //    框架会将所有输入的 storage_shape 归一化为 acc_face shape，
    //    导致 GetStorageShape() 无法检测 shape 不一致；此处改用 GetOriginShape() 获取
    //    用户构建图时指定的原始 shape 进行校验。
    auto accFaceShape = context->GetInputShape(kInputAccFaceIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, accFaceShape);
    const auto& accFaceOrigin = accFaceShape->GetOriginShape();

    // 0.0) dtype 校验（四输入 dtype 必须符合约束）
    //   acc_face / acc_mask / max_mask 恒 DT_FLOAT
    //   bg_img ∈ {DT_FLOAT, DT_UINT8}
    {
        const size_t kDtypeIdxes[] = {kInputAccFaceIdx, kInputAccMaskIdx, kInputMaxMaskIdx, kInputBgImgIdx};
        const char* kDtypeNames[] = {"acc_face", "acc_mask", "max_mask", "bg_img"};
        for (size_t i = 0; i < 4; ++i) {
            auto desc = context->GetInputDesc(kDtypeIdxes[i]);
            OP_CHECK_NULL_WITH_CONTEXT(context, desc);
            ge::DataType dt = desc->GetDataType();
            if (i < 3) {
                OP_CHECK_IF(dt != ge::DT_FLOAT,
                            OP_LOGE(context->GetNodeName(), "BlendFaceBgPartTwo: %s dtype must be FLOAT, got=%d",
                                    kDtypeNames[i], static_cast<int>(dt)),
                            return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(
                    dt != ge::DT_FLOAT && dt != ge::DT_UINT8,
                    OP_LOGE(context->GetNodeName(), "BlendFaceBgPartTwo: %s dtype must be FLOAT or UINT8, got=%d",
                            kDtypeNames[i], static_cast<int>(dt)),
                    return ge::GRAPH_FAILED);
            }
        }
    }

    // 0.1) rank=3 校验
    OP_CHECK_IF(accFaceOrigin.GetDimNum() != kExpectRank,
                OP_LOGE(context->GetNodeName(), "BlendFaceBgPartTwo: acc_face rank must be 3 (H,W,C), got rank=%zu",
                        accFaceOrigin.GetDimNum()),
                return ge::GRAPH_FAILED);

    // 0.2) 空 tensor 拦截（逐维检查 origin shape，任意 dim==0 即拒绝）
    for (size_t d = 0; d < accFaceOrigin.GetDimNum(); ++d) {
        OP_CHECK_IF(accFaceOrigin.GetDim(d) == 0,
                    OP_LOGE(context->GetNodeName(),
                            "BlendFaceBgPartTwo: acc_face origin dim[%zu]=0, empty tensor is not supported", d),
                    return ge::GRAPH_FAILED);
    }

    // 0.3) 四输入 shape 一致性校验（逐维比较 origin shape，防框架 storage shape 归一化绕过）
    {
        const size_t kOtherIdxes[] = {kInputAccMaskIdx, kInputMaxMaskIdx, kInputBgImgIdx};
        const char* kOtherNames[] = {"acc_mask", "max_mask", "bg_img"};
        for (size_t i = 0; i < 3; ++i) {
            auto otherShape = context->GetInputShape(kOtherIdxes[i]);
            OP_CHECK_NULL_WITH_CONTEXT(context, otherShape);
            const auto& otherOrigin = otherShape->GetOriginShape();
            OP_CHECK_IF(
                otherOrigin.GetDimNum() != accFaceOrigin.GetDimNum(),
                OP_LOGE(context->GetNodeName(), "BlendFaceBgPartTwo: %s origin rank=%zu must equal acc_face rank=%zu",
                        kOtherNames[i], otherOrigin.GetDimNum(), accFaceOrigin.GetDimNum()),
                return ge::GRAPH_FAILED);
            for (size_t d = 0; d < accFaceOrigin.GetDimNum(); ++d) {
                OP_CHECK_IF(otherOrigin.GetDim(d) != accFaceOrigin.GetDim(d),
                            OP_LOGE(context->GetNodeName(),
                                    "BlendFaceBgPartTwo: %s origin dim[%zu]=%ld must equal acc_face dim[%zu]=%ld",
                                    kOtherNames[i], d, otherOrigin.GetDim(d), d, accFaceOrigin.GetDim(d)),
                            return ge::GRAPH_FAILED);
            }
        }
    }

    // 1) 平台运行信息（核数 / UB，运行时获取，禁止硬编码）
    uint64_t ubSize = 0;
    int64_t availableCoreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &availableCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2) shape：四输入同 shape，展平 dim0 = H*W*C（以 acc_face 为准）
    auto accFace = context->GetInputShape(kInputAccFaceIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, accFace);
    int64_t dim0 = accFace->GetStorageShape().GetShapeSize();

    // 2.1) bg_img dtype：据此选择 TilingKey 分支参数（bufferDivisor / minDtypeBits）
    auto bgImgDesc = context->GetInputDesc(kInputBgImgIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, bgImgDesc);
    ge::DataType bgDtype = bgImgDesc->GetDataType();
    bool isBgUint8 = (bgDtype == ge::DT_UINT8);
    // KEY_UINT8：bg 参与量最小 dtype 位宽取 uint8=8 用于最小粒度判定；bufferDivisor=34。
    // KEY_FP32 ：minDtypeBits=32，bufferDivisor=40。
    int64_t minDtypeBits = isBgUint8 ? (UINT8_BYTES * 8) : (FP32_BYTES * 8);
    int64_t bufferDivisor = isBgUint8 ? BUFFER_DIVISOR_UINT8 : BUFFER_DIVISOR_FP32;

    // 2.2) epsilon 值域校验（必须 >= 0）
    float epsilonVal = GetEpsilonAttr(context);
    OP_CHECK_IF(epsilonVal < 0.0f,
                OP_LOGE(context->GetNodeName(), "BlendFaceBgPartTwo: epsilon must be >= 0, got=%f", epsilonVal),
                return ge::GRAPH_FAILED);

    // 3) workspace（即使为 0 也须声明 slot）
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 4) TilingData
    BlendFaceBgPartTwoTilingData* tiling = context->GetTilingData<BlendFaceBgPartTwoTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BlendFaceBgPartTwoTilingData), 0, sizeof(BlendFaceBgPartTwoTilingData)) != EOK,
                OP_LOGE(context, "Failed to set tiling data"), return ge::GRAPH_FAILED);
    tiling->epsilon = GetEpsilonAttr(context);

    // 5) 多核切分（每核至少 4KB，blockFormer 512 元素对齐）
    int64_t coreNum = CeilDiv(dim0 * minDtypeBits, MIN_TILING_BITS);
    coreNum = std::min(coreNum, availableCoreNum);
    if (coreNum < 1) {
        coreNum = 1;
    }
    int64_t blockFormer = CeilAlign(CeilDiv(dim0, coreNum), ELEM_ALIGN_FACTOR);
    OP_CHECK_IF(blockFormer <= 0, OP_LOGE(context, "blockFormer=%ld invalid", blockFormer), return ge::GRAPH_FAILED);
    int64_t blockNum = CeilDiv(dim0, blockFormer);

    // 6) UB 切分（256B 对齐，bufferDivisor 按 bg dtype：KEY_FP32=40 / KEY_UINT8=34 bytes/元素）
    int64_t maxElemNum = static_cast<int64_t>(ubSize) / bufferDivisor;
    int64_t alignFactor = ALIGN_256 / FP32_BYTES; // fp32 → 64 元素
    int64_t ubFormer = FloorAlign(maxElemNum, alignFactor);
    OP_CHECK_IF(
        ubFormer <= 0,
        OP_LOGE(context, "ubFormer=%ld invalid (ubSize=%lu, bufferDivisor=%ld)", ubFormer, ubSize, bufferDivisor),
        return ge::GRAPH_FAILED);

    // 7) 首/尾 block 循环次数与尾块大小
    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t blockTail = dim0 - (blockNum - 1) * blockFormer;
    int64_t ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

    tiling->dim0 = dim0;
    tiling->coreNum = coreNum;
    tiling->blockFormer = blockFormer;
    tiling->blockNum = blockNum;
    tiling->ubFormer = ubFormer;
    tiling->ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    tiling->ubTailOfFormerBlock = ubTailOfFormerBlock;
    tiling->ubLoopOfTailBlock = ubLoopOfTailBlock;
    tiling->ubTailOfTailBlock = ubTailOfTailBlock;

    // blockDim = 实际使用的 block 数（每 block 一核）
    context->SetBlockDim(static_cast<uint32_t>(blockNum));

    // 8) TilingKey：数据量足够时启用双缓冲
    uint64_t useDoubleBuffer = 1;
    ASCENDC_TPL_SEL_PARAM(context, useDoubleBuffer);
    OP_LOGD(context,
            "BlendFaceBgPartTwo tiling: bgUint8=%d, bufferDivisor=%ld, dim0=%ld, coreNum=%ld, "
            "blockFormer=%ld, blockNum=%ld, ubFormer=%ld, ubLoopFormer=%ld, ubTailFormer=%ld, "
            "ubLoopTail=%ld, ubTailTail=%ld, eps=%e",
            static_cast<int>(isBgUint8), bufferDivisor, dim0, coreNum, blockFormer, blockNum, ubFormer,
            ubLoopOfFormerBlock, ubTailOfFormerBlock, ubLoopOfTailBlock, ubTailOfTailBlock, tiling->epsilon);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBlendFaceBgPartTwo([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BlendFaceBgPartTwoCompileInfo {}; // 入图场景依赖

IMPL_OP_OPTILING(BlendFaceBgPartTwo)
    .Tiling(BlendFaceBgPartTwoTilingFunc)
    .TilingParse<BlendFaceBgPartTwoCompileInfo>(TilingParseForBlendFaceBgPartTwo);

} // namespace optiling
