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
 * \file paste_sub_img_apt.cpp
 * \brief Kernel entry for paste_sub_img operator
 */
#include "kernel_operator.h"
#include "arch35/paste_sub_img_kernel.h"
#include "arch35/paste_sub_img_tiling_data.h"

template <uint64_t KEY>
__global__ __aicore__ void paste_sub_img(GM_ADDR patch_img, GM_ADDR patch_coord, GM_ADDR core_area_coord,
                                         GM_ADDR combine_img, GM_ADDR combine_img_out, GM_ADDR workspace,
                                         GM_ADDR tiling)
{
    AscendC::SetSysWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PasteSubImgTilingData);
    GET_TILING_DATA_WITH_STRUCT(PasteSubImgTilingData, td, tiling);
    (void)patch_coord;
    (void)core_area_coord;

    PasteSubImgKernel<DTYPE_PATCH_IMG, KEY> kernel;
    kernel.Init(patch_img, combine_img, combine_img_out, &td);
    kernel.Process();
}
