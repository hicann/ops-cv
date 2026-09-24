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
 * \file roi_align_rotated_grad_tiling.h
 * \brief roi_align_rotated_grad kernel UT 手写 tiling 头（经 cmake/ut.cmake 的 AddOpTestCase 以
 *        -include 强制包含，替代 gen_tiling_head_file.py 生成的 roi_align_rotated_grad_tiling_data.h）。
 *
 * 背景：本算子为双 arch 形态——
 *   - 老芯片（ascend910B1 等，arch22）：vector kernel（op_kernel/roi_align_rotated_grad.cpp），
 *     tiling 结构经 op_host/arch22 的 BEGIN_TILING_DATA_DEF/REGISTER_TILING_DATA_CLASS 注册，
 *     生成路径可正常产出头文件；
 *   - ascend950（arch35）：SIMT kernel（op_kernel/arch35/），tiling 结构为纯手写 struct
 *     （RoiAlignRotatedGradTilingData，op_kernel/arch35/roi_align_rotated_grad_tiling_data.h），
 *     不经 REGISTER_TILING_DATA_CLASS 注册，自动生成路径对 950 目标产出空头文件，导致
 *     GET_TILING_DATA_WITH_STRUCT / GET_TILING_DATA 未定义、编译失败。
 * 故参照 roi_align_rotated 先例改为手写本文件，同时覆盖两 soc 目标：
 *   - 全局 class RoiAlignRotatedGradTilingData：字段与 op_host/arch22 的注册结构逐字段一致
 *     （含 codegen 生成的 2 处对齐填充），供老 kernel 与老用例（ascend910B1 目标）使用；
 *   - arch35（950）的 tiling 结构由 op_kernel/arch35/roi_align_rotated_grad_tiling_data.h
 *     经 roi_align_rotated_grad_simt.h 提供，本文件在 3510 环境不再重复定义（避免同名冲突）。
 */

#ifndef _I_ROI_ALIGN_ROTATED_GRAD_UT_TILING_H_
#define _I_ROI_ALIGN_ROTATED_GRAD_UT_TILING_H_

#include <cstdint>
#include <cstring>

// 非 950（arch22）环境：补一份全局 tiling 结构（等价复刻生成头文件中的全局 class 定义，
// 字段顺序/类型/对齐填充与 op_host/arch22/roi_align_rotated_grad_tiling_arch22.h 注册结构一致）。
#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510
class RoiAlignRotatedGradTilingData {
public:
    uint32_t coreRoisNums;
    uint32_t coreRoisTail;
    uint32_t boxSize;
    int32_t pooledHeight;
    int32_t pooledWidth;
    uint32_t batchSize;
    uint32_t channelNum;
    uint32_t width;
    uint32_t height;
    bool aligned;
    bool clockwise;
    uint8_t samplingRatioPH[2];
    int32_t samplingRatio;
    float spatialScale;
    uint32_t coreNum;
    uint8_t RoiAlignRotatedGradTilingDataPH[4];
} __attribute__((__may_alias__));
#endif

// 与 gen_tiling_head_file.py 生成头文件中的宏语义一致（CPU 仿真路径：tiling 缓冲即普通 host 内存）
#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    std::memcpy(&tilingData, (tilingArg), sizeof(tilingStruct))

#define GET_TILING_DATA(tilingData, tilingArg) \
    RoiAlignRotatedGradTilingData tilingData;  \
    std::memcpy(&tilingData, (tilingArg), sizeof(RoiAlignRotatedGradTilingData))

#endif // _I_ROI_ALIGN_ROTATED_GRAD_UT_TILING_H_
