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
 * \file roi_align_grad_tiling_key.h
 * \brief RoiAlignGrad tiling key declarations.
 */

#ifndef __ROI_ALIGN_GRAD_TILING_KEY_H__
#define __ROI_ALIGN_GRAD_TILING_KEY_H__

#define ROI_ALIGN_GRAD_TPL_SCH_DEFAULT 10000
#define ROI_ALIGN_GRAD_TPL_SCH_MOVE_ONE_ROW 11000
#define ROI_ALIGN_GRAD_TPL_SCH_MOVE_ONE_ROW_SUM_IN_UB 11100
#define ROI_ALIGN_GRAD_TPL_SCH_NC1HWC0_ROI_SPLIT 12000

#endif // __ROI_ALIGN_GRAD_TILING_KEY_H__
