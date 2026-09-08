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
 * \file iou3d_infershape.cpp
 * \brief Iou3D 算子形状推导实现
 *
 * shape_rule:
 *   iou.shape = (bboxes.shape[0], bboxes.shape[2], gtboxes.shape[2])  == (B, N, K)
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/operator_reg.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// iou3d 输出形状推导：iou[B, N, K]
static ge::graphStatus InferShape4Iou3D(gert::InferShapeContext* context)
{
    const gert::Shape* bboxesShape = context->GetInputShape(0); // [B, 7, N]
    OP_CHECK_NULL_WITH_CONTEXT(context, bboxesShape);
    const gert::Shape* gtboxesShape = context->GetInputShape(1); // [B, 7, K]
    OP_CHECK_NULL_WITH_CONTEXT(context, gtboxesShape);

    gert::Shape* iouShape = context->GetOutputShape(0); // [B, N, K]
    OP_CHECK_NULL_WITH_CONTEXT(context, iouShape);

    // 处理 unknownRank 场景：输入 rank 未知时，输出也设为 unknownRank
    if (bboxesShape->GetDimNum() == ge::UNKNOWN_RANK.size() || gtboxesShape->GetDimNum() == ge::UNKNOWN_RANK.size()) {
        iouShape->SetDimNum(ge::UNKNOWN_RANK.size());
        return ge::GRAPH_SUCCESS;
    }

    // 输入必须为 rank-3（[B, 7, N] / [B, 7, K]）
    OP_CHECK_IF(bboxesShape->GetDimNum() != 3 || gtboxesShape->GetDimNum() != 3,
                OP_LOGE(context, "Iou3D: bboxes/gtboxes must be rank-3, got %zu/%zu", bboxesShape->GetDimNum(),
                        gtboxesShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    // channel 维度必须为 7（7-DoF）
    const int64_t bboxesChannel = bboxesShape->GetDim(1);
    const int64_t gtboxesChannel = gtboxesShape->GetDim(1);
    OP_CHECK_IF((bboxesChannel >= 0 && bboxesChannel != 7) || (gtboxesChannel >= 0 && gtboxesChannel != 7),
                OP_LOGE(context, "Iou3D: channel dimension must be 7, got bboxes=%ld, gtboxes=%ld", bboxesChannel,
                        gtboxesChannel),
                return ge::GRAPH_FAILED);

    const int64_t batch = bboxesShape->GetDim(0); // B
    const int64_t numN = bboxesShape->GetDim(2);  // N
    const int64_t numK = gtboxesShape->GetDim(2); // K

    // batch 一致性校验
    const int64_t gtboxesBatch = gtboxesShape->GetDim(0);
    OP_CHECK_IF(
        batch >= 0 && gtboxesBatch >= 0 && batch != gtboxesBatch,
        OP_LOGE(context, "Iou3D: batch dimension must be consistent, got bboxes=%ld, gtboxes=%ld", batch, gtboxesBatch),
        return ge::GRAPH_FAILED);

    // D5 对标 mmcv：移除 K≤2000 上限（mmcv 无 K 限制）。逐对计算的 UB/Sort32 缓冲仅按
    //   tileLen 与固定 32 元素多边形排序规模分配，与 K 无耦合，任意 K 成立。

    iouShape->SetDimNum(3);
    iouShape->SetDim(0, batch);
    iouShape->SetDim(1, numN);
    iouShape->SetDim(2, numK);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Iou3D).InferShape(InferShape4Iou3D);

// GE V1 图构建链不会桥接到上面的 gert InferShape。注册同语义的 V1
// InferShapeAndType，覆盖 canndev 中只校验 batch 的历史实现，使 dtype、format、
// rank、channel 和 batch 在进入 Tiling/Kernel 前完成校验。
static ge::graphStatus InferShapeAndTypeForIou3D(ge::Operator& op)
{
    const ge::TensorDesc bboxesDesc = op.GetInputDesc("bboxes");
    const ge::TensorDesc gtboxesDesc = op.GetInputDesc("gtboxes");

    if (bboxesDesc.GetDataType() != ge::DT_FLOAT || gtboxesDesc.GetDataType() != ge::DT_FLOAT) {
        OP_LOGE("Iou3D", "bboxes and gtboxes must both be float32");
        return ge::GRAPH_FAILED;
    }

    if (bboxesDesc.GetFormat() != ge::FORMAT_ND || gtboxesDesc.GetFormat() != ge::FORMAT_ND) {
        OP_LOGE("Iou3D", "bboxes and gtboxes must both use ND format");
        return ge::GRAPH_FAILED;
    }

    const ge::Shape bboxesShape = bboxesDesc.GetShape();
    const ge::Shape gtboxesShape = gtboxesDesc.GetShape();
    const bool bboxesUnknownRank = (bboxesShape.GetDims() == ge::UNKNOWN_RANK);
    const bool gtboxesUnknownRank = (gtboxesShape.GetDims() == ge::UNKNOWN_RANK);

    ge::TensorDesc outputDesc = op.GetOutputDesc("iou");
    outputDesc.SetDataType(ge::DT_FLOAT);
    outputDesc.SetFormat(ge::FORMAT_ND);
    outputDesc.SetOriginFormat(ge::FORMAT_ND);

    if (bboxesUnknownRank || gtboxesUnknownRank) {
        outputDesc.SetShape(ge::Shape(ge::UNKNOWN_RANK));
        outputDesc.SetOriginShape(ge::Shape(ge::UNKNOWN_RANK));
        return op.UpdateOutputDesc("iou", outputDesc);
    }

    const std::vector<int64_t> bboxesDims = bboxesShape.GetDims();
    const std::vector<int64_t> gtboxesDims = gtboxesShape.GetDims();
    if (bboxesDims.size() != 3U || gtboxesDims.size() != 3U) {
        OP_LOGE("Iou3D", "bboxes and gtboxes must both be rank-3");
        return ge::GRAPH_FAILED;
    }

    if ((bboxesDims[1] >= 0 && bboxesDims[1] != 7) || (gtboxesDims[1] >= 0 && gtboxesDims[1] != 7)) {
        OP_LOGE("Iou3D", "channel dimension must be 7");
        return ge::GRAPH_FAILED;
    }

    if (bboxesDims[0] >= 0 && gtboxesDims[0] >= 0 && bboxesDims[0] != gtboxesDims[0]) {
        OP_LOGE("Iou3D", "bboxes and gtboxes must have the same batch dimension");
        return ge::GRAPH_FAILED;
    }

    const ge::Shape outputShape({bboxesDims[0], bboxesDims[2], gtboxesDims[2]});
    outputDesc.SetShape(outputShape);
    outputDesc.SetOriginShape(outputShape);
    return op.UpdateOutputDesc("iou", outputDesc);
}

COMMON_INFER_FUNC_REG(Iou3D, InferShapeAndTypeForIou3D);

} // namespace ops
