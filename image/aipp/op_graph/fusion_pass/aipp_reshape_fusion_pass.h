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
 * \file aipp_reshape_fusion_pass.h
 * \brief After InferShape: rewrite Aipp images, and copy format/origin format/size onto features.
 */

#ifndef OP_GRAPH_AIPP_RESHAPE_FUSION_PASS_H
#define OP_GRAPH_AIPP_RESHAPE_FUSION_PASS_H

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace Ops {
class __attribute__((visibility("default"))) AippReshapeFusionPass : public ge::fusion::FusionBasePass {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;
};
} // namespace Ops

#endif // OP_GRAPH_AIPP_RESHAPE_FUSION_PASS_H
