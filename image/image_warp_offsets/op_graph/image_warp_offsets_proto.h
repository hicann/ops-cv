/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMAGE_WARP_OFFSETS_PROTO_H
#define IMAGE_WARP_OFFSETS_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief This operation select images to warp_images according to offsets.

*@par Inputs:
*@li images: 4-D Tensor with shape [batch, height, width, 3].
*@li offsets: 4-D Tensor with shape [batch, 4, new_height, new_width].

*@par Outputs:
*warp_images: Returns 5-D Tensor with shape
[batch, 4, new_height, new_width, 3] and the same dtype as images.

*@attention Constraints:
*@li When input dtype of offsets is int32, images should be float16
*/
REG_OP(IMGWarpOffsets)
    .INPUT(images, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT}))
    .INPUT(offsets, TensorType({DT_FLOAT, DT_INT32}))
    .OUTPUT(warp_images, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(IMGWarpOffsets)
}

#endif