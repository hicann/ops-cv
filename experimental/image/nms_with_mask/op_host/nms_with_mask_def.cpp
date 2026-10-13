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
 * \file nms_with_mask.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class NMSWithMask : public OpDef {
public:
    explicit NMSWithMask(const char* name) : OpDef(name)
    {
        this->Input("boxes")                     // 输入boxes定义
            .ParamType(REQUIRED)                 // 必选输入
            .DataType({ge::DT_FLOAT})            // 支持数据类型
            .Format({ge::FORMAT_ND})             // 支持format格式
            .UnknownShapeFormat({ge::FORMAT_ND}) // 未确定大小shape对应format格式
            .AutoContiguous();                   // 内存自动连续化
        this->Input("scores")                    // 输入scores定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("selected_indices") // 输出selected_indices定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("iou_threshold").AttrType(OPTIONAL).Float(1e-1);
        this->Attr("scores_threshold").AttrType(OPTIONAL).Float(1e-1);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(NMSWithMask); // 添加算子信息库
} // namespace ops
