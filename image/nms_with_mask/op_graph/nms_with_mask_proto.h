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
 * \file nms_with_mask_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_NMS_WITH_MASK_OPS_H_
#define OPS_OP_PROTO_INC_NMS_WITH_MASK_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Iteratively removes lower scoring boxes which have an IoU greater than
* iou_threshold with higher scoring box according to their
* intersection-over-union (IoU) . \n

* @par Inputs:
* box_scores: 2-D tensor with shape of [N, 5], including proposal boxes and
* corresponding confidence scores . Support dtype: [float16, float32, bfloat16], Support format: [ND]. \n

* @par Attributes:
* iou_threshold: An optional float. The threshold for deciding whether boxes
* overlap too much with respect to IOU . Default value is 0.5 \n

* @par Outputs:
* @li selected_boxes: 2-D tensor with shape of [N,5], representing filtered
* boxes including proposal boxes and corresponding confidence scores.
* Support dtype: [float16, float32, bfloat16], Support format: [ND]. \n
* @li selected_idx: 1-D tensor with shape of [N], representing the index of
* input proposal boxes. Support dtype: [int32], Support format: [ND]. \n
* @li selected_mask: 1-D tensor with shape of [N], the symbol judging whether
* the output proposal boxes is valid . Support dtype: [uint8], Support format: [ND]. \n

* @attention Constraints:
* The 2nd-dim of input box_scores must be equal to 5.\n

*/
REG_OP(NMSWithMask)
    .INPUT(box_scores, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(selected_boxes, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(selected_idx, TensorType({DT_INT32}))
    .OUTPUT(selected_mask, TensorType({DT_UINT8}))
    .ATTR(iou_threshold, Float, 0.5)
    .OP_END_FACTORY_REG(NMSWithMask)

} // namespace ge

#endif