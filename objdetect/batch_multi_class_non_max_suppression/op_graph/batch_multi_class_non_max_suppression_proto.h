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
 * \file batch_multi_class_non_max_suppression_proto.h
 * \brief Operator prototype definition for BatchMultiClassNonMaxSuppression.
 */
#ifndef OPS_OP_PROTO_INC_BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_OPS_H_
#define OPS_OP_PROTO_INC_BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Selects high-scoring bounding boxes independently for each class and suppresses boxes with excessive IoU.
 *
 * @par Inputs:
 * @li boxes: A float16 or float32 tensor. When transpose_box is false, its shape is [B, N, q, 4]; when true, its
 * shape is [B, q, 4, N]. B is the batch size, N is the number of boxes, and q is 1 or the number of classes.
 * @li scores: A float16 or float32 tensor of shape [B, N, C], where C is the number of classes. Its type must equal
 * the type of boxes.
 * @li clip_window: An optional float16 or float32 tensor of shape [B, 4]. Its type must equal the type of boxes.
 * @li num_valid_boxes: An optional int32 tensor of shape [B].
 *
 * @par Attributes:
 * @li score_threshold: A required float, the score filtering threshold.
 * @li iou_threshold: A required float in [0, 1], the IoU suppression threshold.
 * @li max_size_per_class: A required int, the maximum selected boxes per class.
 * @li max_total_size: A required int, the maximum selected boxes per batch.
 * @li change_coordinate_frame: An optional bool. When true, clip_window must be provided. Defaults to false.
 * @li transpose_box: An optional bool indicating the layout of boxes. Defaults to false.
 * @li image_size: An optional list of int values reserved for graph compatibility.
 *
 * @par Outputs:
 * @li nmsed_boxes: A float16 or float32 tensor of shape [B, max_total_size, 4].
 * @li nmsed_scores: A float16 or float32 tensor of shape [B, max_total_size].
 * @li nmsed_classes: A float16 or float32 tensor of shape [B, max_total_size].
 * @li nmsed_num: An int32 tensor of shape [B].
 */
REG_OP(BatchMultiClassNonMaxSuppression)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(clip_window, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(num_valid_boxes, TensorType({DT_INT32}))
    .OUTPUT(nmsed_boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(nmsed_scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(nmsed_classes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(nmsed_num, TensorType({DT_INT32}))
    .REQUIRED_ATTR(score_threshold, Float)
    .REQUIRED_ATTR(iou_threshold, Float)
    .REQUIRED_ATTR(max_size_per_class, Int)
    .REQUIRED_ATTR(max_total_size, Int)
    .ATTR(change_coordinate_frame, Bool, false)
    .ATTR(transpose_box, Bool, false)
    .ATTR(image_size, ListInt, {})
    .OP_END_FACTORY_REG(BatchMultiClassNonMaxSuppression)
} // namespace ge

#endif // OPS_OP_PROTO_INC_BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_OPS_H_
