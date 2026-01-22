/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_NON_MAX_SUPPRESSION_V6_OPS_H
#define OPS_OP_PROTO_INC_NON_MAX_SUPPRESSION_V6_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Greedily selects a subset of bounding boxes in descending order of
score . \n

*@par Inputs:
*Input boxes and  scores must be float16 type. Inputs include:
*@li boxes: A input tensor with shape [num_batches,spatial_dimension,4].
The single box data format is indicated by center_point_box.
*@li scores: A input tensor with shape [num_batches,num_classes,spatial_dimension]
*@li max_output_size: A scalar integer tensor representing the maximum number
of boxes to be selected by non max suppression.
*@li iou_threshold: A 0-D float tensor representing the threshold for deciding
whether boxes overlap too much with respect to IOU.
*@li score_threshold: A 0-D float tensor representing the threshold for
deciding when to remove boxes based on score . \n

*@par Attributes:
*center_point_box:Integer indicate the format of the box data.
The default is 0. 0 - the box data is supplied as [y1, x1, y2, x2]
where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair
of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval [0, 1]) or absolute.Mostly used for TF models.
1 - the box data is supplied as [x_center, y_center, width, height].
 Mostly used for Pytorch models. \n

*@par Outputs:
*@li selected_indices: A 2-D integer tensor of shape [M] representing the
selected indices from the boxes tensor, where M <= max_output_size. \n

*@attention Constraints:
*Input boxes and  scores must be float16 type . \n

*@par Third-party framework compatibility
*Compatible with onnx NonMaxSuppression operator.

*@par Restrictions:
*Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/

REG_OP(NonMaxSuppressionV6)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(max_output_size, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(iou_threshold, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(score_threshold, TensorType({DT_FLOAT}))
    .OUTPUT(selected_indices, TensorType({DT_INT32}))
    .ATTR(center_point_box, Int, 0)
    .ATTR(max_boxes_size, Int, 0)
    .OP_END_FACTORY_REG(NonMaxSuppressionV6)
} // namespace ge

#endif // OPS_OP_PROTO_INC_NON_MAX_SUPPRESSION_V6_OPS_H