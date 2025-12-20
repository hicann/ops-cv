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
*@brief Resize images to size using linear interpolation.

*@par Inputs:
*Input images must be a 3-D tensor. Inputs include:
*@li x: 3-D with shape [batch, channels, L] (format is NCL), dtype must in (FLOAT32, BFLOAT16, FLOAT16).
*@li size: A 1-D int32 tensor of 1 elements: output L. The new size L for the images.

*@par Attributes:
*@li align_corners: An optional bool. If true, the centers of the 4 corner pixels of the input
and output tensors are aligned, preserving the values at the corner pixels.
Defaults to false.
* @li scale: An optional float. Multiplier for spatial size. Defaults to 0.0f.
*@par Outputs:
*y: 3-D with shape [batch, channels, L] (format is NCL), dtype and format is same as input x.
    The N, C dimension must be the same as x.

*@par Third-party framework compatibility
*Compatible with PyTorch upsample_linear1d operator.
*/
REG_OP(ResizeLinear)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(align_corners, Bool, false)
    .ATTR(scale, Float, 0.0f)
    .OP_END_FACTORY_REG(ResizeLinear)
}  // namespace ge

#endif  // OPS_IMAGE_RESIZE_LINEAR_GRAPH_PLUGIN_RESIZE_LINEAR_GRAD_PROTO_H_
