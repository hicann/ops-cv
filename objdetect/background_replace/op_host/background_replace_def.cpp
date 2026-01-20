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
 * \file background_replace.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class BackgroundReplace : public OpDef {
 public:
  explicit BackgroundReplace(const char* name) : OpDef(name) {
    this->Input("bkg")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_UINT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("src")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_UINT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("mask")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_UINT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->AICore().AddConfig("ascend310p");
  }
};
OP_ADD(BackgroundReplace);
}  // namespace ops