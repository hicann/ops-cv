/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/gaussian_blur_tiling_key.h"

#if !GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS && defined(TILING_KEY_VAR)
#if TILING_KEY_VAR == 0UL
#define GAUSSIAN_BLUR_COMPILE_ROW_ONLY
#elif TILING_KEY_VAR == 1UL
#define GAUSSIAN_BLUR_COMPILE_COLUMN_ONLY
#endif
#endif

#include "arch35/gaussian_blur_kernel.h"

template <uint32_t passVariant>
__global__ __aicore__ void gaussian_blur(GM_ADDR src, GM_ADDR dst, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(GaussianBlurTilingData);
    GET_TILING_DATA_WITH_STRUCT(GaussianBlurTilingData, tilingData, tiling);
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
    if constexpr (passVariant == GAUSSIAN_BLUR_PASS_FUSED_K31_C4_RING) {
        NsGaussianBlurW128::ProcessFusedK31DualSimd(src, dst, &tilingData);
    } else if constexpr (passVariant == GAUSSIAN_BLUR_PASS_ROW_W128) {
        NsGaussianBlurW128::ProcessRow(src, dst, &tilingData);
    } else {
        NsGaussianBlurW128::ProcessColumn(src, dst, &tilingData);
    }
#else
    if constexpr (passVariant == GAUSSIAN_BLUR_PASS_FUSED_GENERIC_C8) {
        NsGaussianBlurW128::ProcessFusedGenericC8(src, dst, &tilingData);
    } else if constexpr (passVariant == GAUSSIAN_BLUR_PASS_ROW_W128) {
        NsGaussianBlurW128::ProcessRow(src, dst, &tilingData);
    } else {
        NsGaussianBlurW128::ProcessColumn(src, dst, &tilingData);
    }
#endif
}

#undef GAUSSIAN_BLUR_COMPILE_COLUMN_ONLY
#undef GAUSSIAN_BLUR_COMPILE_ROW_ONLY
