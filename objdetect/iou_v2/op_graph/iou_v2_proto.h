/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file iou_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_

#include "graph/operator_reg.h"
namespace ge
{
    /**
    * @brief Computes the intersection over union (iou) or the intersection over
    * foreground (iof) based on the ground-truth and predicted regions . \n

    * @par Inputs:
    * Two inputs, including:
    * @li bboxes: Bounding boxes, a 2D Tensor of type float16 or float32 with
    * shape (N, 4). "N" indicates the number of bounding boxes, and the value
    * "4" refers to "x0", "x1", "y0", and "y1".
    * @li gtboxes: Ground-truth boxes, a 2D Tensor of type float16 or float32
    * with shape (M, 4). It's dtype should be same as bboxes.
    * "M" indicates the number of ground truth boxes, and
    * the value "4" refers to "x0", "x1", "y0", and "y1" . \n

    * @par Attributes:
    * @li those attribute is valid only for Ascend910B AI Processors and later products.
    * @li mode: Computation mode, a character string with the value range of [iou, iof].
    * default value is iou .
    * @li eps: An optional float, prevent division by 0, default value is 1.0 .
    * The value can only choose one of those values: [0, 0.01, 1] when Soc Version is : \n
    * Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component. \n
    * Atlas A3 Training Series Product/Atlas A3 Inference Series Product. \n
    * Atlas Training Series Product. \n
    * Atlas Inference Series Product. \n
    * @li aligned: A bool value, if aligned is true, calculate the ious between each aligned pair of bboxes and gtboxes.
    * default value is false . \n
    * @par Outputs:
    * overlap: A 2D Tensor of type float16 or float32 with shape [M, N] or [M, 1], specifying
    * the IoU or IoF ratio . It's dtype should be same as bboxes. \n

    * @attention Constraints:
    * Computation of float16 and float32 data are supported. To avoid overflow, the input
    * length and width are scaled by 0.2 internally.
    */
    REG_OP(IouV2)
        .INPUT(bboxes, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
        .INPUT(gtboxes, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
        .OUTPUT(overlap, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
        .ATTR(mode, String, "iou")
        .ATTR(eps, Float, 1.0)
        .ATTR(aligned, Bool, false)
        .OP_END_FACTORY_REG(IouV2)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_