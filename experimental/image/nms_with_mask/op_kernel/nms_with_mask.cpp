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

#include "nms_with_mask.h"

enum class NMSWithMaskTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void nms_with_mask(GM_ADDR boxes, GM_ADDR scores, GM_ADDR selectedIndices, GM_ADDR workspace,
                                         GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NMSWithMaskTilingData);
    GET_TILING_DATA_WITH_STRUCT(NMSWithMaskTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(NMSWithMaskTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsNMSWithMask::NMSWithMask<float> op;                 // 算子kernel实例获取
        op.Init(boxes, scores, selectedIndices, &tilingData); // 算子kernel实例初始化
        op.Process();                                         // 算子kernel实例执行
    }
}
