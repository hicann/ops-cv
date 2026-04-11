/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sampling_kernels.h"
#include <map>

namespace aicpu {
SamplingKernelType SamplingKernelTypeFromString(const std::string &str)
{
    // Define map for different types of sampling kernels
    static const std::map<std::string, SamplingKernelType> SamplingTypesInfo {
        {"lanczos1",      LANCZOS1_KERNEL},
        {"lanczos3",      LANCZOS3_KERNEL},
        {"lanczos5",      LANCZOS5_KERNEL},
        {"gaussian",      GAUSSIAN_KERNEL},
        {"box",           BOX_KERNEL},
        {"triangle",      TRIANGLE_KERNEL},
        {"keyscubic",     KEYS_CUBIC_KERNEL},
        {"mitchellcubic", MITCHELL_CUBIC_KERNEL},
    };

    std::map<std::string, SamplingKernelType>::const_iterator iter = SamplingTypesInfo.find(str);
    if (iter != SamplingTypesInfo.end()) {
        return iter->second;
    }

    return SAMPLING_KERNEL_TYPE_END;
}
}  // namespace aicpu
