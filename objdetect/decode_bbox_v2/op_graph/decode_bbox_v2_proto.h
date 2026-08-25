/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DECODE_BBOX_V2_PROTO_H
#define DECODE_BBOX_V2_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

/**
 *@brief Computes decode bboxv2 function.
 *
 *@par Inputs:
 *Inputs include:
 * @li boxes: A Tensor. Must be float16 or float32. Supported format list ["ND"].
 * @li anchors: A Tensor. Must be float16 or float32. Supported format list ["ND"].
 *
 *@par Attributes:
 * @li scales: optional, listfloat. Default value is [1.0,1.0,1.0,1.0].
 * @li decode_clip: optional, float, threahold of decode process. Default value is 0.0
 * @li reversed_boxes: optional, bool. Default value is false.
 *
 *@par Outputs:
 * y: A Tensor. Must have the same type as box_predictions. Supported format list ["ND"].
 */
REG_OP(DecodeBboxV2)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(anchors, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(scales, ListFloat, {1.0, 1.0, 1.0, 1.0})
    .ATTR(decode_clip, Float, 0.0)
    .ATTR(reversed_box, Bool, false)
    .OP_END_FACTORY_REG(DecodeBboxV2)

} // namespace ge

#endif
