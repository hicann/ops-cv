/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 # Authors (accounts):
 # - Liu Jun <@kbryantttt>
 # - Tu Yuanhang <@TuYHAAAAAA>
 # - Zhou Jianhua<@LePenseur>
 # - Liang Yanglin <@liang-yanglin>
 # - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file roi_align_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class RoiAlignV2 : public OpDef {
public:
    explicit RoiAlignV2(const char* name) : OpDef(name)
    {
        // 输入参数说明
        this->Input("features")                                      
            .ParamType(REQUIRED)                                
            .DataType({ge::DT_FLOAT})             
            .Format({ge::FORMAT_ND})             
            .UnknownShapeFormat({ge::FORMAT_ND}) 
            .AutoContiguous();                                 
        this->Input("rois")                                       
            .ParamType(REQUIRED)                              
            .DataType({ge::DT_FLOAT})             
            .Format({ge::FORMAT_ND})             
            .UnknownShapeFormat({ge::FORMAT_ND}) 
            .AutoContiguous();                                 
        /* ...此处补充其他输入输出参数说明 */
        this->Attr("pooled_height")
            .AttrType(OPTIONAL)
            .Int();
        this->Attr("pooled_width")
            .AttrType(OPTIONAL)
            .Int();
        this->Attr("spatial_scale")
            .AttrType(OPTIONAL)
            .Float();
        this->Attr("sampling_ratio")
            .AttrType(OPTIONAL)
            .Int();

        // 输出参数说明
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(RoiAlignV2); 
} // namespace ops