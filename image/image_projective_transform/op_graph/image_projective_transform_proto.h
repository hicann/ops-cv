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
 * \file image_projective_transform_proto.h
 * \brief
 */
#ifndef OPS_CV_IMAGE_IMAGE_PROJECTIVE_TRANSFORM_PROTO_H
#define OPS_CV_IMAGE_IMAGE_PROJECTIVE_TRANSFORM_PROTO_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Applies a projective transformation to the input image. \n

*@par Inputs:
* Including:
* @li images: A Tensor. Must be one of the following types: uint8, int32, int64, float16, float, double.
*             A 4-D tensor of shape [batch, image_height, image_width, channels].
*             The format must be NHWC.
* @li transforms: A Tensor of type float. A 2-D tensor of shape [batch, 8], each row contains
*                 8 projective transformation parameters [a0, a1, a2, a3, a4, a5, a6, a7].
* @li output_shape: A Tensor of type int32. A 1-D tensor of 2 elements,
*                   output_shape = [output_height, output_width]. \n

*@par Attributes:
* @li interpolation: A string from {"BILINEAR", "NEAREST"}. The interpolation method.
* @li fill_mode: An optional string. Defaults to "CONSTANT". Currently only "CONSTANT"
*                is supported, the fill value is 0. \n

*@par Outputs:
* @li transformed_images: A Tensor. Has the same type as images.
*         A 4-D tensor of shape [batch, output_height, output_width, channels].
*         The format must be NHWC. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow ImageProjectiveTransformV2 operator.
*/
REG_OP(ImageProjectiveTransform)
    .INPUT(images, TensorType({DT_UINT8, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(transforms, TensorType({DT_FLOAT}))
    .INPUT(output_shape, TensorType({DT_INT32}))
    .REQUIRED_ATTR(interpolation, String)
    .ATTR(fill_mode, String, "CONSTANT")
    .OUTPUT(transformed_images, TensorType({DT_UINT8, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(ImageProjectiveTransform)

} // namespace ge

#endif // OPS_CV_IMAGE_IMAGE_PROJECTIVE_TRANSFORM_PROTO_H
