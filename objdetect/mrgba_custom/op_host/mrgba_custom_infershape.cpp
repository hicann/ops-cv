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
 * \file mrgba_custom.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
    static ge::graphStatus InferShape4MrgbaCustom(gert::InferShapeContext* context) {
        const gert::Shape *rgb_shape = context->GetInputShape(0);
        gert::Shape *dst_shape = context->GetOutputShape(0);
        *dst_shape = *rgb_shape;
        return GRAPH_SUCCESS;
    }

     static ge::graphStatus InferDataTypeForMrgbaCustom(gert::InferDataTypeContext *context)
    {
        const ge::DataType dst_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, dst_dtype);
        return GRAPH_SUCCESS;
    }

    IMPL_OP_INFERSHAPE(MrgbaCustom).InferShape(InferShape4MrgbaCustom).InferDataType(InferDataTypeForMrgbaCustom);
}  // namespace ops