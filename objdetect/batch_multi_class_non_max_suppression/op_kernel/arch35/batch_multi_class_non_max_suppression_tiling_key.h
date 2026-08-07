/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_TILING_KEY_H_
#define BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_TILING_KEY_H_

// TILING_KEY_IS consumes this value in the preprocessor, so it must remain a
// numeric macro rather than a C++ constexpr. The host tiler includes this
// header too, keeping the generated binary and runtime tiling key aligned.
#define BATCH_MULTI_CLASS_NMS_TILING_KEY 10000UL

#endif // BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_TILING_KEY_H_
