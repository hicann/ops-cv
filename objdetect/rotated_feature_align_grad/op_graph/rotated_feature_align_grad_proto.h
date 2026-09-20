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
 * \file rotated_feature_align_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ROTATED_FEATURE_ALIGN_GRAD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ROTATED_FEATURE_ALIGN_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief RotatedFeatureAlignGrad:Calculate the gradient of input features according to
* the gradient of output features. \n

* @par Inputs:
* @li dy: A tensor of type float32. The gradient of output features.
* @li bboxes: A tensor of type float32. The position information of bboxes. \n

* @par Outputs:
* @li dx: A tensor of type float32. The gradient of input features. \n

* @par Attributes:
* @li spatial_scale: A required float32. The scale of feature map to initial image.
* @li points: An optional int. Defaults to "1". The number of sample points. \n

* @par Third-party framework compatibility
* Compatible with MMCV RotatedFeatureAlign operator.
*/
#ifndef OPS_PROTO_DEF_ROTATEDFEATUREALIGNGRAD
#define OPS_PROTO_DEF_ROTATEDFEATUREALIGNGRAD
REG_OP(RotatedFeatureAlignGrad)
    .INPUT(dy, TensorType({DT_FLOAT}))
    .INPUT(bboxes, TensorType({DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(spatial_scale, Float)
    .ATTR(points, Int, 1)
    .OP_END_FACTORY_REG(RotatedFeatureAlignGrad)
#endif
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_ROTATED_FEATURE_ALIGN_GRAD_H_
