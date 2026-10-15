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
 * \file background_replace_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_BACKGROUND_REPLACE_H_
#define OPS_OP_PROTO_INC_BACKGROUND_REPLACE_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Replaces the background using out = bkg * (1 - mask) + src * mask.
 *
 * @par Inputs:
 * @li bkg: An ND background image tensor of type uint8 or float16, with HWC shape.
 * @li src: An ND source image tensor with the same dtype and shape as bkg.
 * @li mask: An ND float16 mask tensor with HWC shape and one channel.
 *
 * @par Outputs:
 * @li out: The blended image tensor, with the same dtype and shape as bkg.
 */
REG_OP(BackgroundReplace)
    .INPUT(bkg, TensorType({DT_UINT8, DT_FLOAT16}))
    .INPUT(src, TensorType({DT_UINT8, DT_FLOAT16}))
    .INPUT(mask, TensorType({DT_FLOAT16, DT_FLOAT16}))
    .OUTPUT(out, TensorType({DT_UINT8, DT_FLOAT16}))
    .OP_END_FACTORY_REG(BackgroundReplace)
} // namespace ge
#endif // OPS_OP_PROTO_INC_BACKGROUND_REPLACE_H_
