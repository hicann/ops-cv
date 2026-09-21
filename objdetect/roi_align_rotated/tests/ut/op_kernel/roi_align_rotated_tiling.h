/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_align_rotated_tiling.h
 * \brief roi_align_rotated kernel UT 手写 tiling 头（经 cmake/ut.cmake 的 AddOpTestCase 以
 *        -include 强制包含，替代 gen_tiling_head_file.py 生成的 roi_align_rotated_tiling_data.h）。
 *
 * 背景：本算子为双 arch 形态——
 *   - 老芯片（ascend910B1 等，arch22）：vector kernel（op_kernel/roi_align_rotated.cpp，由
 *     ut.cmake 的 KernelFile glob 编入各 soc UT 目标），tiling 结构经 op_host/arch22 的
 *     BEGIN_TILING_DATA_DEF/REGISTER_TILING_DATA_CLASS 注册，生成路径可正常产出头文件；
 *   - ascend950（arch35）：SIMT kernel（op_kernel/arch35/），tiling 结构为纯手写 struct
 *     （RoiAlignRotatedArch35TilingData），不经 REGISTER_TILING_DATA_CLASS 注册，自动生成
 *     路径对 950 目标产出空头文件（"RoiAlignRotated do not registe tiling struct"），
 *     导致 GET_TILING_DATA_WITH_STRUCT / GET_TILING_DATA 未定义、编译失败。
 * 故参照 gaussian_blur / rgb2yuv422 先例改为手写本文件，同时覆盖两 soc 目标：
 *   - 全局平铺 RoiAlignRotatedTilingData：字段与 op_host/arch22/roi_align_rotated_tiling_arch22.h
 *     中 BEGIN_TILING_DATA_DEF(RoiAlignRotatedTilingData) 逐字段一致（顺序/类型相同），
 *     等价复刻生成头文件中的全局结构定义，供老 kernel 与老用例（ascend910B1 目标）使用；
 *   - RoiAlignRotatedArch35TilingData：直接包含 op_kernel/arch35 自带定义，
 *     供 950 用例（test_roi_align_rotated_950.cpp）与 arch35 kernel 使用。
 */

#ifndef _I_ROI_ALIGN_ROTATED_UT_TILING_H_
#define _I_ROI_ALIGN_ROTATED_UT_TILING_H_

#include <cstdint>
#include <cstring>

// arch35（ascend950）SIMT kernel tiling 结构（与 op_kernel/arch35/roi_align_rotated_simt.h 共用同一份定义）
#include "../../../op_kernel/arch35/roi_align_rotated_tiling_data.h"

// 老芯片（arch22）vector kernel tiling 结构（全局平铺 class，字段布局与 arch22 注册结构一致）
class RoiAlignRotatedTilingData {
public:
    uint8_t aligned = 0;
    uint8_t clockwise = 0;
    uint32_t numBlocks = 0;
    uint32_t rois_num_per_Lcore = 0;
    uint32_t rois_num_per_Score = 0;
    uint32_t Lcore_num = 0;
    uint32_t Score_num = 0;
    uint32_t input_buffer_size = 0;
    uint32_t tileNum = 0;
    uint32_t batch_size = 0;
    uint32_t channels = 0;
    uint32_t channels_aligned = 0;
    uint32_t input_h = 0;
    uint32_t input_w = 0;
    uint32_t rois_num_aligned = 0;
    uint32_t tail_num = 0;
    float spatial_scale = 0;
    int32_t sampling_ratio = 0;
    int32_t pooled_height = 0;
    int32_t pooled_width = 0;
    uint64_t ub_total_size = 0;
};

// CPU 仿真（tikicpulib/ASCENDC_CPU_DEBUG）下 tiling 缓冲即普通 host 内存，按字节反序列化即可
inline void InitTilingData(const uint8_t* tiling, RoiAlignRotatedTilingData* tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(RoiAlignRotatedTilingData));
}

inline void InitTilingData(const uint8_t* tiling, RoiAlignRotatedArch35TilingData* tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(RoiAlignRotatedArch35TilingData));
}

// 与 gen_tiling_head_file.py 生成头文件中的宏语义一致（CPU 仿真路径）
#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    RoiAlignRotatedTilingData tilingData;      \
    InitTilingData(tilingArg, &tilingData)

#endif // _I_ROI_ALIGN_ROTATED_UT_TILING_H_
