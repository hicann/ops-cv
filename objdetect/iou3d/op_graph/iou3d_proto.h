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
 * \file iou3d_proto.h
 * \brief Iou3D 图模式（GE IR）算子原型定义
 */
#ifndef OPS_OP_PROTO_INC_IOU3D_H_
#define OPS_OP_PROTO_INC_IOU3D_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Calculate the intersection ratio of two rotated cuboids . \n

* @par Inputs:
* @li bboxes : data of grad increment, a 3D Tensor of type float32 with
* shape (B, 7, N). "N" indicates the number of boxes, and the value
* "7" refers to [x, y, z, w, h, d, theta].
* @li gtboxes: Bounding boxes, a 3D Tensor of type float32 with
* shape (B, 7, K). "K" indcates the number of boxes, and the value
* "7" refers to [x, y, z, w, h, d, theta].

* @par Outputs:
* iou: A 3D Tensor of float32 with shape [B, N, K].

* @attention Constraints:
* In each batch, the invalid box cannot appear before the valid box.
*/
REG_OP(Iou3D)
    .INPUT(bboxes, TensorType({DT_FLOAT}))
    .INPUT(gtboxes, TensorType({DT_FLOAT}))
    .OUTPUT(iou, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(Iou3D)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IOU3D_H_
