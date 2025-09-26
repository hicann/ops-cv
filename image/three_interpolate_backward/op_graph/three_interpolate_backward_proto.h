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
 * \file three_interpolate_backward_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_THREE_INTERPOLATE_BACKWARD_OPS_H_
#define OPS_OP_PROTO_INC_THREE_INTERPOLATE_BACKWARD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief three interpolate backward.
 * @par Inputs:
 * three input:
 * grad_x: The set of features points with dtype of float32 and float16 with shape [b,c,n]
 * idx: The set of index with dtype of int32 and int64 with shape [b,n,3]
 * weight : The set of weight points with dtype of float32 and float16 with shape[b,n,3]
 * m: The dims m of output with dtype int
 * @par y:
 * grad_y: A Tensor, the interpolate backward output with dtype of float32 and float16 with shape[b,c,m]
 */
REG_OP(ThreeInterpolateBackward)
    .INPUT(grad_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(grad_y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(m, Int)
    .OP_END_FACTORY_REG(ThreeInterpolateBackward)

} // namespace ge

#endif