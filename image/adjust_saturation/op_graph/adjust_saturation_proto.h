/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADJUST_SATURATION_PROTO_H
#define ADJUST_SATURATION_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Adjusts the saturation of RGB images. \n
 *
 * @par Inputs:
 * @li images: A Tensor of type float16 or float. The last dimension must be 3.
 *     RGB images to adjust. At least 3-D.
 * @li scale: A Tensor of type float. A float scale to add to the saturation. \n
 *
 * @par Outputs:
 * y: A Tensor. Has the same type and shape as images. Adjusted image(s). \n
 *
 * @attention Constraints:
 * The last dimension of images must be size 3. \n
 *
 * @par Third-party framework compatibility
 * Compatible with tensorflow AdjustSaturation operator.
 */

REG_OP(AdjustSaturation)
    .INPUT(images, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(AdjustSaturation)
}

#endif
