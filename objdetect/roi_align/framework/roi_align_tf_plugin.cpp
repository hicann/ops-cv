/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "log/log.h"

#define NET_2D_H1_ROI_ALIGN_POOL_HEIGHT 7
#define NET_2D_H1_ROI_ALIGN_POOL_WIDTH 7

namespace domi {
static Status ROIAlignParams(const std::vector<const google::protobuf::Message*> insideNodes, ge::Operator& op)
{
    OP_LOGI(TbeGetName(op).c_str(), "Enter ROIAlign fusion parser.");
    auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);
    if (opDesc == nullptr) {
        OP_LOGE(TbeGetName(op).c_str(), "Get op desc failed.");
        return FAILED;
    }
    if (!ge::AttrUtils::SetFloat(opDesc, "spatial_scale", 1.0)) {
        OP_LOGE(TbeGetName(op).c_str(), "Set spatial_scale failed.");
        return FAILED;
    }
    if (!ge::AttrUtils::SetInt(opDesc, "pooled_height", NET_2D_H1_ROI_ALIGN_POOL_HEIGHT)) {
        OP_LOGE(TbeGetName(op).c_str(), "Set pooled_height failed.");
        return FAILED;
    }
    if (!ge::AttrUtils::SetInt(opDesc, "pooled_width", NET_2D_H1_ROI_ALIGN_POOL_WIDTH)) {
        OP_LOGE(TbeGetName(op).c_str(), "Set pooled_width failed.");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("ROIAlign")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ROIAlign")
    .FusionParseParamsFn(ROIAlignParams)
    .ImplyType(ImplyType::TVM);
} // namespace domi
