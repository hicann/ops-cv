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
 * \file upsample_nearest_exact3d_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_UPSAMPLE_NEAREST_EXACT3D_H_
#define OPS_OP_PROTO_INC_UPSAMPLE_NEAREST_EXACT3D_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Upsample the 3-D data with the nearest neighbor ​interpolation algorithm. \n

*@par Inputs:
*One inputs, including: \n
*x: A 5-D input tensor. The format must be NCDHW. Must be one of the following types:
* float16, float32, bfloat16. \n

*@par Attributes:
*@li output_size: An required listInt. Defaults to none.
* Contain 3 elements: output_depth, output_height, output_width. The number of elements of 'output_size'
* should be the same as the rank of input 'x'. Only one of 'scales' and 'output_size' can be specified. \n
*@li scales: An optional listFloat. Defaults to `[0.0, 0.0, 0.0]`.
* The scale array along each dimension, contain 3 elements: scale_depth, scale_height, scale_width.
* The number of elements of 'scales' should be the same as the rank of input 'x'. One of 'scales' and
* 'output_size' must be specified and it is an error if both are specified. \n

*@par Outputs:
*y: A 5-D tensor. The format must be NCDHW. Must be one of the following types:
* float16, float32, bfloat16.
* Has the same type as input x, shape depends on x and output_size/scales. \n

*/

REG_OP(UpsampleNearestExact3d)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(scales, ListFloat, {0.0f, 0.0f, 0.0f})
    .OP_END_FACTORY_REG(UpsampleNearestExact3d)

}  // namespace ge

#endif  // OPS_OP_PROTO_INC_UPSAMPLE_NEAREST_EXACT3D_H_