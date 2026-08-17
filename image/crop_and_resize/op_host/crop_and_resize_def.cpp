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
 * \file crop_and_resize_def.cpp
 * \brief Operator definition for crop_and_resize operator
 *
 * 4 dtype combinations (aligned with TBE op_select_format):
 *   #0: x=FP32, boxes=FP32, box_index=INT32, crop_size=INT32, y=FP32
 *   #1: x=FP16, boxes=FP32, box_index=INT32, crop_size=INT32, y=FP32
 *   #2: x=FP16, boxes=FP16, box_index=INT32, crop_size=INT32, y=FP16
 *   #3: x=FP32, boxes=FP16, box_index=INT32, crop_size=INT32, y=FP16
 */

#include "register/op_def_registry.h"

namespace ops {

class CropAndResize : public OpDef {
public:
    explicit CropAndResize(const char* name) : OpDef(name)
    {
        // 输入 x: 4D (N, H, W, C), float16/float32
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 boxes: 2D (num_boxes, 4), float16/float32
        // NaN 检查已移至 kernel 运行时，tiling 不再读取 boxes 值，无需 ValueDepend
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 box_index: 1D (num_boxes,), int32
        this->Input("box_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 crop_size: 1D (2,), int32
        // ValueDepend(OPTIONAL): infershape/tiling 需读取 crop_size 值（与 MDE §3.3 一致）
        this->Input("crop_size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();

        // 输出 y: 4D (num_boxes, crop_h, crop_w, C), dtype = boxes dtype
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 属性: extrapolation_value (Float, 默认 0)
        this->Attr("extrapolation_value").AttrType(OPTIONAL).Float(0);

        // 属性: method (String, 默认 "bilinear", 本次仅支持 bilinear)
        this->Attr("method").AttrType(OPTIONAL).String("bilinear");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "crop_and_resize_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(CropAndResize);
} // namespace ops
