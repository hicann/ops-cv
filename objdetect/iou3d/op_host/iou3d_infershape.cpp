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
    OP_CHECK_IF(bboxesChannel != 7 || gtboxesChannel != 7,
                OP_LOGE(context, "Iou3D: channel dimension must be 7, got bboxes=%ld, gtboxes=%ld", bboxesChannel,
                        gtboxesChannel),
                return ge::GRAPH_FAILED);

    const int64_t batch = bboxesShape->GetDim(0); // B
    const int64_t numN = bboxesShape->GetDim(2);  // N
    const int64_t numK = gtboxesShape->GetDim(2); // K

    // batch 一致性校验
    const int64_t gtboxesBatch = gtboxesShape->GetDim(0);
    OP_CHECK_IF(
        batch != gtboxesBatch,
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

} // namespace ops
