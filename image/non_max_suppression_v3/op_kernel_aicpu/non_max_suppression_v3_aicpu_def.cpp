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
 * \file non_max_suppression_v3_aicpu_def.cpp
 * \brief non_max_suppression_v3 def
 */

#include "register/op_def_registry.h"
#include "../../../common/inc/aicpu/aicpu_op_def.h"

namespace ops {
class NonMaxSuppressionV3 : public OpDef {
public:
    explicit NonMaxSuppressionV3(const char* name) : OpDef(name)
    {
        ApplyCvAicpuDefaultCfg(*this);
        this->Input("boxes").DataType({ge::DT_FLOAT, ge::DT_FLOAT16});
        this->Input("scores").DataType({ge::DT_FLOAT, ge::DT_FLOAT16});
        this->Input("max_output_size").DataType({ge::DT_INT32});
        this->Input("iou_threshold").DataType({ge::DT_FLOAT, ge::DT_FLOAT16});
        this->Input("score_threshold").DataType({ge::DT_FLOAT, ge::DT_FLOAT16});
        this->Output("selected_indices").DataType({ge::DT_INT32});
        this->AICPU().ExtendCfgInfo(OP_INFO_SUB_TYPE_OF_INFERSHAPE.c_str(), DEFAULT_SUB_TYPE_OF_INFERSHAPE_4.c_str());
    }
};

OP_ADD(NonMaxSuppressionV3);
} // namespace ops
