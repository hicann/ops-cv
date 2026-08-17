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
 * \file paste_sub_img_tiling_arch35.h
 * \brief Tiling header for paste_sub_img operator on arch35
 */
#ifndef PASTE_SUB_IMG_TILING_ARCH35_H_
#define PASTE_SUB_IMG_TILING_ARCH35_H_

#include <cstdint>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/paste_sub_img_tiling_data.h"
#include "../../op_kernel/arch35/paste_sub_img_tiling_key.h"

namespace optiling {

struct PasteSubImgCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling
#endif // PASTE_SUB_IMG_TILING_ARCH35_H_
