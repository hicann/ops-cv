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
 * \file spatial_transformer_aicpu_def.cpp
 * \brief spatial_transformer def
 */

#include "register/op_def_registry.h"
#include "../../../common/inc/aicpu/aicpu_op_def.h"

namespace ops {
/*
此配置项针对动态shape的场景，配置AI CPU的infershape方式，有以下几种取值：
1：根据inputshape可直接推导outputshape。
2：部分计算的infershape，根据tensorData的值做简单的计算。
3：全部计算的infershape，但最大shape可预估，按照算子最大的shape申请内存。
4：无法由inputshape推导出算子的输出shape，也无法推导出输出range，此种情况需要在算子实现时进行内存的申请，计算完成后，返回output和输出shape。
默认值为1。
*/
class SpatialTransformer : public OpDef {
public:
    explicit SpatialTransformer(const char* name) : OpDef(name)
    {
        ApplyCvAicpuDefaultCfg(*this);
        this->Input("x")
            .DataType(
                {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT16, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_UINT64, ge::DT_INT64});
        this->Input("theta")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT16, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_UINT64, ge::DT_INT64});
        this->Output("y")
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT16, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_UINT64, ge::DT_INT64});
        this->Attr("output_size").AttrType(OPTIONAL).ListInt({-1, -1});
        this->Attr("default_theta").AttrType(OPTIONAL).ListFloat();
        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_default_theta").AttrType(OPTIONAL).ListInt();
        this->AICPU().ExtendCfgInfo(OP_INFO_FORMAT_AGNOSTIC.c_str(), TRUE_FORMAT_AGNOSTIC.c_str());
    }
};

OP_ADD(SpatialTransformer);
} // namespace ops
