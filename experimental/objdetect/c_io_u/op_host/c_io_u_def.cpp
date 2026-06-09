/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file c_io_u_def.cpp
 * \brief OpDef registration for CIoU.
 *
 * Schema matches CANN built-in REG_OP(CIoU) in nn_detect_ops.h:
 *   inputs : bboxes, gtboxes (4, N)
 *   outputs: overlap, atan_sub (1, N)
 *   attrs  : trans (Bool), is_cross (Bool), mode (String), atan_sub_flag (Bool)
 */
#include "register/op_def_registry.h"

namespace ops {
class CIoU : public OpDef {
public:
    explicit CIoU(const char* name) : OpDef(name)
    {
        this->Input("bboxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gtboxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("overlap")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("atan_sub")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("trans").AttrType(OPTIONAL).Bool(false);
        this->Attr("is_cross").AttrType(OPTIONAL).Bool(false);
        this->Attr("mode").AttrType(OPTIONAL).String("iou");
        this->Attr("atan_sub_flag").AttrType(OPTIONAL).Bool(true);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(CIoU);
} // namespace ops
