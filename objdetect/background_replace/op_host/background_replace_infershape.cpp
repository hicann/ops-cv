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
 * \file background_replace.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
using namespace ge;
namespace ge {
static ge::graphStatus InferShape4BackgroundReplace(gert::InferShapeContext* context) {
  const gert::Shape* bkgShape = context->GetInputShape(0);
  gert::Shape* outShape = context->GetOutputShape(0);
  *outShape = *bkgShape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BackgroundReplace).InferShape(InferShape4BackgroundReplace);
}  // namespace ops