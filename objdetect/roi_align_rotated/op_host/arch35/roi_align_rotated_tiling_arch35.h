/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROI_ALIGN_ROTATED_ARCH35_TILING_H_
#define ROI_ALIGN_ROTATED_ARCH35_TILING_H_

#include <cstdint>

namespace optiling {

// ascend950（arch35）编译期缓存结构：TilingParse（opc 编译期）预计算平台常量并序列化进
// binary json，运行期 tiling 经 GetCompileInfo<RoiAlignRotatedArch35CompileInfo>() 读缓存，
// 免去每次 tiling 的平台信息查询。全 uint64_t 字段，无填充字节。
struct RoiAlignRotatedArch35CompileInfo {
    uint64_t coreNum = 0;             // AIV 核数（GetCoreNumAiv）
    uint64_t ubSize = 0;              // UB 总容量（GetCoreMemSize(UB)）
    uint64_t libApiWorkSpaceSize = 0; // 系统段 workspace 大小（GetLibApiWorkSpaceSize）
};

} // namespace optiling

#endif // ROI_ALIGN_ROTATED_ARCH35_TILING_H_
