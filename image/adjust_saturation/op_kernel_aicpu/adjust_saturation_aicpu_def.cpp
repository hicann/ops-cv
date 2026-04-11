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
 * \file adjust_saturation_aicpu_def.cpp
 * \brief AdjustSaturation def
 */

#include "register/op_def_registry.h"
#include "../../../common/inc/aicpu/aicpu_op_def.h"

namespace ops {
class AdjustSaturation : public OpDef {
public:
    explicit AdjustSaturation(const char* name) : OpDef(name)
    {
        ApplyCvAicpuDefaultCfg(*this);
        this->Input("images").DataType({ge::DT_FLOAT16, ge::DT_FLOAT});
        this->Input("scale").DataType({ge::DT_FLOAT});
        this->Output("y").DataType({ge::DT_FLOAT16, ge::DT_FLOAT});
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
    }
};

OP_ADD(AdjustSaturation);
} // namespace ops
