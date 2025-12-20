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
 * \file upsample_nearest3d_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_UPSAMPLE_NEAREST3D_GRAD_H_
#define OPS_OP_PROTO_INC_UPSAMPLE_NEAREST3D_GRAD_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Upsample the 3-D gradient data with the nearest neighbor ​interpolation algorithm. \n

*@par Inputs:
*One inputs, including: \n
*grad_output: A 5-D input tensor. The format must be NCDHW. Must be one of the following types:
* float16, float32, float64, uint8, bfloat16. \n

*@par Attributes:
*@li input_size: An required listInt.
* Contain 5 elements: [min_batch, channels, depth, height, width]. Must:
* input_size[0] == grad_output_tensor_size[0]
* input_size[1] == grad_output_tensor_size[1]. \n
*@li output_size: An optional listInt. Defaults to none.
* Contain 3 elements: depth, height, width. The number of elements of 'output_size' should
* be the same as the rank of input 'grad_output'. Only one of 'scales' and 'output_size' can be specified. Must:
* grad_output_tensor_size[2] == floor(input_size[2] * scales[0]) == output_size[0]
* grad_output_tensor_size[3] == floor(input_size[3] * scales[1]) == output_size[1]
* grad_output_tensor_size[4] == floor(input_size[4] * scales[2]) == output_size[2]. \n
*@li scales: An optional listFloat. Defaults to none.
* The scale array along each dimension, contain 3 elements: scale_depth, scale_height, scale_width.
* The number of elements of 'scales' should be the same as the rank of input 'grad_output'.
* One of 'scales' and 'output_size' must be specified and it is an error if both are specified. \n

*@par Outputs:
*y: A 5-D tensor. The format must be NCDHW. Must be one of the following types:
* float16, float32, float64, uint8, bfloat16.
* Has the same type as input grad_output, shape depends on Attributes:input_size. \n

*/

REG_OP(UpsampleNearest3dGrad)
    .INPUT(grad_output, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_BF16}))
    .REQUIRED_ATTR(input_size, ListInt)
    .ATTR(output_size, ListInt, {})
    .ATTR(scales, ListFloat, {})
    .OP_END_FACTORY_REG(UpsampleNearest3dGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_UPSAMPLE_NEAREST3D_GRAD_H_