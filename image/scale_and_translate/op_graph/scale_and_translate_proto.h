/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * The code snippet comes from Huawei's open-source Mindspore project.
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef SCALE_AND_TRANSLATE_PROTO_H
#define SCALE_AND_TRANSLATE_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Resizes "images" to "size" by scale and translate . \n
 *
 * @par Inputs:
 * @li images: A `Tensor`. Must be one of the following types: `int8`, `uint8`,
 * `int16`, `uint16`, `int32`, `int64`, `float16`, `float32`, `float64`.
 * @li size: A `Tensor` of type `int32`.
 * @li scale: A `Tensor` of type `float32`.
 * @li translation: A `Tensor` of type `float32` . \n
 *
 * @par Attributes:
 * @li kernel_type: type is string, default is lanczos3.
 * @li antialias: type is bool, default is true. \n
 *
 * @par Outputs:
 * y: A Tensor with type float32 . \n
 *
 * @par Third-party framework compatibility
 * Compatible with TensorFlow ScaleAndTranslate operator.
*/

REG_OP(ScaleAndTranslate)
    .INPUT(images, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                               DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(translation, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(kernel_type, String, "lanczos3")
    .ATTR(antialias, Bool, true)
    .OP_END_FACTORY_REG(ScaleAndTranslate)
}

#endif
