/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file onnx_common.h
 * \brief
 */

#ifndef CV_COMMON_ONNX_COMMON_H
#define CV_COMMON_ONNX_COMMON_H

#include <map>
#include <string>
#include <vector>

#include "base/err_msg.h"
#include "cv_plugin_util.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "log/log.h"
#include "onnx/proto/ge_onnx.pb.h"
#include "register/register.h"
#include "stub_ops.h"

// Vec2Tensor / CreateScalar live in cv_plugin_util.h (included above) for
// protobuf-free reuse.

#endif //  CV_COMMON_ONNX_COMMON_H
