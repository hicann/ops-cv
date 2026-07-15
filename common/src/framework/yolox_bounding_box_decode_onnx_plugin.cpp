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
 * \file yolox_bounding_box_decode_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {
static Status ParseParamsYoloxBoundingBoxDecode(const Message* op_src, ge::Operator& op_dest) { return SUCCESS; }

// register YoloxBoundingBoxDecode op info to GE
REGISTER_CUSTOM_OP("YoloxBoundingBoxDecode")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::YoloxBoundingBoxDecode"),
                   ge::AscendString("ai.onnx::9::YoloxBoundingBoxDecode"),
                   ge::AscendString("ai.onnx::10::YoloxBoundingBoxDecode"),
                   ge::AscendString("ai.onnx::11::YoloxBoundingBoxDecode"),
                   ge::AscendString("ai.onnx::12::YoloxBoundingBoxDecode"),
                   ge::AscendString("ai.onnx::13::YoloxBoundingBoxDecode"),
                   ge::AscendString("ai.onnx::14::YoloxBoundingBoxDecode"),
                   ge::AscendString("ai.onnx::15::YoloxBoundingBoxDecode"),
                   ge::AscendString("ai.onnx::16::YoloxBoundingBoxDecode")})
    .ParseParamsFn(ParseParamsYoloxBoundingBoxDecode)
    .ImplyType(ImplyType::TVM);
} // namespace domi
