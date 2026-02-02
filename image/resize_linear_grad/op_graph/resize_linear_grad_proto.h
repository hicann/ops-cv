/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_linear_grad_proto.h
 * \brief
 */
#ifndef OPS_IMAGE_RESIZE_LINEAR_GRAPH_PLUGIN_RESIZE_LINEAR_GRAD_PROTO_H_
#define OPS_IMAGE_RESIZE_LINEAR_GRAPH_PLUGIN_RESIZE_LINEAR_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Backwards calculation of ResizeLinear.

*@par Inputs:
*@li grads: A 3D tensor, represents the gradient of output of ResizeLinear. Format must be NCL.
     Dtype must be float32, float16 or bfloat16.
*@li original_image: A 3D tensor, represents the resized image of ResizeLinear. Format and dtype must be the same as grads.
     The N, C dimension must be the same as grads.

*@par Outputs:
*y: A 3D tensor, represents the gradient of original_image. Format and dtype and shape must be the same as original_image.

*@par Attributes:
*@li align_corners: An optional bool. If true, the centers of the 2 corner pixels of the input and output tensors are
     aligned, preserving the values at the corner pixels.If false, calculate interpolation using half pixel centers.
     Defaults to false.
*@li scale: An optional float. The element represents the ratio of the L axis
     subscripts of pixels in grads to the L axis subscripts of pixels in y, only takes effect when align_corners is
     true and the value is greater than 0. Defaults to 0.0f. 

*@par Third-party framework compatibility
*Compatible with PyTorch upsample_linear1d_backward operator.
*/
REG_OP(ResizeLinearGrad)
    .INPUT(grads, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(original_image, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(align_corners, Bool, false)
    .ATTR(scale, Float, 0.0f)
    .OP_END_FACTORY_REG(ResizeLinearGrad)
}  // namespace ge

#endif  // OPS_IMAGE_RESIZE_LINEAR_GRAPH_PLUGIN_RESIZE_LINEAR_GRAD_PROTO_H_
