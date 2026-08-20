/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GAUSSIAN_BLUR_KERNEL_H_
#define GAUSSIAN_BLUR_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/vector_functions.h"
#include "gaussian_blur_tiling_data.h"

#define GAUSSIAN_BLUR_IMPL_NAMESPACE NsGaussianBlurW128
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
#define GAUSSIAN_BLUR_ROW_PATCHES 9U
#define GAUSSIAN_BLUR_ROW_TILE_W 288U
#define GAUSSIAN_BLUR_ROW_SHARED_W 352U
#define GAUSSIAN_BLUR_ROW_UB_PATCH_W GAUSSIAN_BLUR_ROW_SHARED_W
#define GAUSSIAN_BLUR_ROW_UB_BUFFER_BYTES \
    (GAUSSIAN_BLUR_ROW_TILE_H * GAUSSIAN_BLUR_ROW_UB_PATCH_W * (GAUSSIAN_BLUR_GENERIC_CHANNEL_GROUP) * sizeof(float))
#endif
#if defined(GAUSSIAN_BLUR_COMPILE_ROW_ONLY)
#define GAUSSIAN_BLUR_ROW_ONLY
#elif defined(GAUSSIAN_BLUR_COMPILE_COLUMN_ONLY)
#define GAUSSIAN_BLUR_COLUMN_ONLY
#endif
#include "gaussian_blur_kernel_impl.inl"
#if defined(GAUSSIAN_BLUR_COMPILE_ROW_ONLY)
#undef GAUSSIAN_BLUR_ROW_ONLY
#elif defined(GAUSSIAN_BLUR_COMPILE_COLUMN_ONLY)
#undef GAUSSIAN_BLUR_COLUMN_ONLY
#endif
#if GAUSSIAN_BLUR_ENABLE_COMPACT_TILING_KEYS
#undef GAUSSIAN_BLUR_ROW_UB_BUFFER_BYTES
#undef GAUSSIAN_BLUR_ROW_UB_PATCH_W
#undef GAUSSIAN_BLUR_ROW_SHARED_W
#undef GAUSSIAN_BLUR_ROW_TILE_W
#undef GAUSSIAN_BLUR_ROW_PATCHES
#endif
#undef GAUSSIAN_BLUR_IMPL_NAMESPACE

#endif // GAUSSIAN_BLUR_KERNEL_H_
