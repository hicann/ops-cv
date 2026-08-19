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
 * \file paste_sub_img_proto.h
 * \brief Operator proto registration for paste_sub_img
 */
#ifndef OPS_OP_PROTO_INC_PASTE_SUB_IMG_H_
#define OPS_OP_PROTO_INC_PASTE_SUB_IMG_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief paste sub img.
*
* @par Inputs:
* @li patch_img: A 3D Tensor, format is ND, dtype is uint8 or float16 or float32,
* shape is (H, W, C). The input image.
* @li patch_coord: A 1D Tensor, format is ND, dtype is int32, shape is (4,). The coordinates
* in the combined img.
* @li core_area_coord: A 1D Tensor, format is ND, dtype is int32, shape is (4,). The
* coordinates in the patch img
* @li combine_img: A 3D Tensor, format is ND, dtype is uint8 or float16 or float32, shape is
* (H, W, C). \n
*
* @par Outputs:
* @li combine_img: A 3D Tensor, format is ND. It has the same type and shape as input
 "combine_img". \n
*
* @par Attr
* @li scale: A required float, scale of coordinates. \n
*/
#ifndef OPS_PROTO_DEF_PASTESUBIMG
#define OPS_PROTO_DEF_PASTESUBIMG
REG_OP(PasteSubImg)
    .INPUT(patch_img, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT32}))
    .INPUT(patch_coord, TensorType({DT_INT32}))
    .INPUT(core_area_coord, TensorType({DT_INT32}))
    .INPUT(combine_img, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(combine_img, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT32}))
    .REQUIRED_ATTR(scale, Float)
    .OP_END_FACTORY_REG(PasteSubImg)
#endif
} // namespace ge

#endif // OPS_OP_PROTO_INC_PASTE_SUB_IMG_H_
