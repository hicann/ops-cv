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
 * \file blend_images_custom_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_

#include "graph/operator_reg.h"
namespace ge
{
/**
* @brief Generate rgb and frame images into a out image with alpha transparency. \n

* @par Inputs:
* @li rgb: A Int, dtype is uint8, rgb images data.
* @li alpha: A Int, dtype is uint8, alpha transparency images data.
* @li frame: A Int, dtype is uint8, frame images data.  \n

* @par Outputs:
* @li out: The out tensor. Dtype is same as rgb. \n
*/
REG_OP(BlendImagesCustom)
    .INPUT(rgb, TensorType({DT_UINT8}))
    .INPUT(alpha, TensorType({DT_UINT8}))
    .INPUT(frame, TensorType({DT_UINT8}))
    .OUTPUT(out, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(BlendImagesCustom)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_OTHER_H_