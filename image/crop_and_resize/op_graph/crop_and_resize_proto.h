/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 * The code snippet comes from Huawei's open-source Mindspore project.
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#ifndef crop_and_resize_PROTO_H
#define crop_and_resize_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Extracts crops from the input image tensor and resizes them. Extracts
crops from the input image tensor and resizes them using bilinear sampling or
nearest neighbor sampling to a common output size specified by crop_size . \n

*@par Inputs:
*Input x must be a 4-D tensor. Inputs include:
* @li x: A Tensor. Must be one of the following types:uint8, uint16, int8,
int16, int32, int64, float16, float, double. A 4-D tensor of shape
[batch, image_height, image_width, depth]. The format must be NHWC.
*@li boxes: A Tensor. Must be one of the following types: float16, float. A 2-D tensor of shape [num_boxes, 4].
*@li box_index: A Tensor of type int32. A 1-D tensor of shape [num_boxes] with
int32 values in [0, batch).
* @li crop_size: A Tensor of type int32. A 1-D tensor of 2 elements, crop_size
= [crop_height, crop_width]. All cropped image patches are resized to this size . \n

*@par Attributes:
*@li extrapolation_value: An optional float. Defaults to 0. Value used for
extrapolation, when applicable.
*@li method: An optional string from: '"bilinear", "nearest"'. Defaults to
"bilinear". Currently two sampling methods are supported: Bilinear and
NearestNeighbor . \n

*@par Outputs:
* y: A Tensor. Must be one of the following types: float16, float. The format must be NHWC. \n

*@attention Constraints:
*Input images must be a 4-D tensor . \n

*@par Third-party framework compatibility
*Compatible with tensorflow CropAndResize operator.
*/

REG_OP(CropAndResize)
    .INPUT(x, TensorType({DT_UINT8, DT_UINT16, DT_INT8, \
        DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(box_index, TensorType({DT_INT32}))
    .INPUT(crop_size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(extrapolation_value, Float, 0)
    .ATTR(method, String, "bilinear")
    .OP_END_FACTORY_REG(CropAndResize)
}

#endif