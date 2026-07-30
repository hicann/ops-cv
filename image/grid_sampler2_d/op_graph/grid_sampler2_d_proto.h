/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRID_SAMPLER2_D_PROTO_H_
#define GRID_SAMPLER2_D_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief This operation samples input x by using interpolation based on flow
*field grid, which is usually gennerated by affine_grid. The grid of shape
*[N, H, W, 2] is the concatenation of (x, y) coordinates with shape [N, H, W]
*each, where x is indexing the 4th dimension (in width dimension) of input
*data x and y is indexng the 3rd dimention (in height dimension), finally
*results is the interpolation value of 4 nearest corner points. The output
*tensor shape will be [N, C, H, W].

*@par Inputs:
*@li x: 4-D Tensor with shape `[batch, channels, height, width]`. Must be one
*of the following types: float16, float.
*@li grid: flow field grid, 4-D Tensor with shape `[batch, height, width, 2]`
*and has the same dtype as `x`. \n

*@par Attributes:
*@li interpolation_mode: An optional string specifying the interpolation
*method, either 'bilinear', 'nearest' and 'bicubic'. Defaults to
*"bilinear".
*@li padding_mode: An optional string specifying the pad method, either
*"zeros", "border", or "reflection". Defaults to "zeros".
*@li align_corners: An optional bool. If "true", the centers of the corner
*pixels of the input and output tensors are aligned. Defaults to "false" . \n

*@par Outputs:
*y: Returns 4-D Tensor with the same dtype as `x`. \n

*@par Third-party framework compatibility
*Compatible with pytorch GridSampler2D operator.
*/
REG_OP(GridSampler2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(grid, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(interpolation_mode, String, "bilinear")
    .ATTR(padding_mode, String, "zeros")
    .ATTR(align_corners, Bool, false)
    .OP_END_FACTORY_REG(GridSampler2D)

} // namespace ge

#endif // GRID_SAMPLER2_D_PROTO_H_
