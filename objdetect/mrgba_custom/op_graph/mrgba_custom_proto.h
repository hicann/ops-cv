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
 * \file mgrba_custom_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_

#include "graph/operator_reg.h"
namespace ge
{
    /**
    * @brief Give transparency to the image.
    *
    * @par Inputs:
    * @li rgb: A tensor of the type DT_UINT8.
    * @li alpha:A tensor of the type DT_UINT8.
    *
    * @par Outputs:
    * @li dst: A tensor of the type DT_UINT8.
    */
    REG_OP(MrgbaCustom)
        .INPUT(rgb, TensorType({ DT_UINT8 }))
        .INPUT(alpha, TensorType({ DT_UINT8 }))
        .OUTPUT(dst, TensorType({ DT_UINT8 }))
        .OP_END_FACTORY_REG(MrgbaCustom)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_DETECT_OPS_H_