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
 * \file col2_im_v2_tiling_data.h
 * \brief tiling data struct for col2_im_v2
 */

#ifndef __COL2_IM_V2_TILING_DATA_H__
#define __COL2_IM_V2_TILING_DATA_H__

#include <cstdint>

struct Col2ImV2TilingData {
    int64_t totalLength = 0; // 输出元素总数 n*c*outH*outW（<= INT32_MAX，tiling 校验）
    int64_t needCoreNum = 0; // 实际启动核数（SetBlockDim 同源，供 DFX）
    int64_t outputSizeH = 0; // outH（const tensor output_size 读值）
    int64_t outputSizeW = 0; // outW
    int64_t kernelSizeH = 0; // kH（const tensor kernel_size 读值）
    int64_t kernelSizeW = 0; // kW
    int64_t dilationH = 0;   // attr dilation[0]
    int64_t dilationW = 0;   // attr dilation[1]
    int64_t paddingH = 0;    // attr padding[0]
    int64_t paddingW = 0;    // attr padding[1]
    int64_t strideH = 0;     // attr stride[0]
    int64_t strideW = 0;     // attr stride[1]
    int64_t colH = 0;        // ho = (outH + 2*padH - dilH*(kH-1) - 1)/strideH + 1（tiling 校验用；不传 VF）
    int64_t colW = 0;        // wo 同理
    // 不包含 threadNum/blockDimX 等线程数参数（VF 线程数为编译期常量）
    // 不包含 UintDiv magic/shift（device 端 Process 中预计算，禁止入 TilingData）
};

#endif
