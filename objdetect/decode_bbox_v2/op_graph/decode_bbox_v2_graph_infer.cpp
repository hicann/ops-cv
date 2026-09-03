/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

/**
 * InferDataTypeForDecodeBboxV2: GE data-type inference callback.
 *
 * y.dtype = boxes.dtype (same_as_first_input).  Validates that both
 * inputs have the same dtype (no type promotion).
 */
static ge::graphStatus InferDataTypeForDecodeBboxV2(gert::InferDataTypeContext* context)
{
    const ge::DataType boxesDtype = context->GetInputDataType(0);
    const ge::DataType anchorsDtype = context->GetInputDataType(1);
    if (boxesDtype != anchorsDtype) {
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(0, boxesDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(DecodeBboxV2).InferDataType(InferDataTypeForDecodeBboxV2);

} // namespace ops
