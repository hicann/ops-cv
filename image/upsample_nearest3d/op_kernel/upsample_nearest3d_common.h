/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_nearest3d_common.h
 * \brief
 */
#ifndef UPSAMPLE_NEAREST3D_COMMON_H
#define UPSAMPLE_NEAREST3D_COMMON_H
#include "kernel_operator.h"

using namespace AscendC;

template <typename T1, typename T2>
__aicore__ inline T1 CeilA2B(const T1 a, const T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
};

template <typename T1>
__aicore__ inline T1 Min(const T1 a, const T1 b)
{
    return a < b ? a : b;
};

template <typename T1>
__aicore__ inline T1 Max(const T1 a, const T1 b)
{
    return a > b ? a : b;
};
#endif  // UPSAMPLE_NEAREST3D_COMMON_H