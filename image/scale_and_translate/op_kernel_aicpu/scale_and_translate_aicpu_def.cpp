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
 * \file scale_and_translate_aicpu_def.cpp
 * \brief scale_and_translate def
 */

#include "register/op_def_registry.h"
#include "../../../common/inc/aicpu/aicpu_op_def.h"

namespace ops {
class ScaleAndTranslate : public OpDef {
public:
    explicit ScaleAndTranslate(const char* name) : OpDef(name)
    {
        ApplyCvAicpuDefaultCfg(*this);
        this->Input("images").DataType(
            {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16,
             ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE});
        this->Input("size").DataType({ge::DT_INT32});
        this->Input("scale").DataType({ge::DT_FLOAT});
        this->Input("translation").DataType({ge::DT_FLOAT});
        this->Output("y").DataType({ge::DT_FLOAT});
        this->Attr("kernel_type").AttrType(OPTIONAL).String("lanczos3");
        this->Attr("antialias").AttrType(OPTIONAL).Bool(true);
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
    }
};

OP_ADD(ScaleAndTranslate);
} // namespace ops
