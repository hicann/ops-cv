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
 * \file test_iou3d.cpp
 * \brief Iou3D kernel 执行 UT（arch35 / tikicpulib CPU 仿真）。
 *
 * 定位：验证 kernel 主路径在 CPU 仿真下可执行不崩溃，并对典型锚点做宽松数值 sanity check
 *   （非逐元素精度比对——逐元素精度由 assets/golden.py + 外部 ST/TTK 框架在真机上把关）。
 * 覆盖：
 *   - identical box（IoU≈1）+ disjoint box（IoU≈0）：走 Sin/Cos 向量批算 + 顶点收集 + 面积。
 *   - 旋转相交 box：触发 >3 交点的 Sort32 极角排序分支。
 *   - 空 Tensor（isEmpty=1）：Process 运行时短路，无输出写。
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/arch35/iou3d_tiling_data.h"

using namespace std;

// kernel 入口符号：OpType(Iou3D) 的 snake_case 形式 iou3_d（框架 CamelCase→snake_case 约定）。
// 仅前向声明，与 op_kernel/iou3d.cpp 的定义一致（C++ linkage，无 extern "C"）。
// kernel .cpp 由 AddOpTestCase 宏（cmake/ut.cmake）自动编入同一 case object，勿在此 #include 以免重复定义。
__global__ __aicore__ void iou3_d(GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR iou, GM_ADDR workspace, GM_ADDR tiling);

class iou3d_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "iou3d SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "iou3d_test TearDown\n" << endl; }
};

namespace {
// 写入一个框的 7-DoF 到 [B,7,D] 布局：base=b*7*D + c*D + idx。
void SetBox(float* gm, int64_t b, int64_t idx, int64_t dimSize, const std::array<float, 7>& box)
{
    for (int c = 0; c < 7; ++c) {
        gm[b * 7 * dimSize + static_cast<int64_t>(c) * dimSize + idx] = box[c];
    }
}

// 单核填 tiling：所有 (b,i,j) 对由 1 个核处理。
void FillTiling(Iou3DTilingData* t, uint32_t batch, uint32_t numN, uint32_t numK, uint32_t isEmpty)
{
    memset(t, 0, sizeof(Iou3DTilingData));
    t->batch = batch;
    t->numBboxes = numN;
    t->numGtboxes = numK;
    t->isEmpty = isEmpty;
    if (isEmpty != 0U) {
        t->coreNum = 1;
        return;
    }
    const uint32_t total = batch * numN * numK;
    t->coreNum = 1;
    t->pairsPerCore = total;
    t->tileLen = total < 256U ? total : 256U;
    t->tailLen = (t->tileLen == 0U) ? 0U : (total % t->tileLen);
    // Cos/Sin sharedTmpBuffer: host uses GetCosMaxMinTmpSize; UT(CPU sim) 用安全上界。
    //   8 elem: 768B, 256 elem: 3072B (POLYNOMIAL_APPROXIMATION max); 4096B 覆盖 tileLen≤256。
    t->cosTmpSize = 4096U;
}
} // namespace

// 正常框对：[1,7,2] × [1,7,2]。box0 identical(IoU≈1)，box1 与 box0 不相交(IoU≈0)。
TEST_F(iou3d_test, test_iou3d_basic_fp32)
{
    const uint32_t B = 1, N = 2, K = 2;
    size_t bboxesByteSize = B * 7 * N * sizeof(float);
    size_t gtboxesByteSize = B * 7 * K * sizeof(float);
    size_t iouByteSize = B * N * K * sizeof(float);
    size_t tilingDataSize = sizeof(Iou3DTilingData);

    uint8_t* bboxes = (uint8_t*)AscendC::GmAlloc(bboxesByteSize);
    uint8_t* gtboxes = (uint8_t*)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t* iou = (uint8_t*)AscendC::GmAlloc(iouByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    // 输入布局 [B,7,N]：box0 在原点、box1 远离。gtbox 同 bbox。
    float* pb = reinterpret_cast<float*>(bboxes);
    float* pg = reinterpret_cast<float*>(gtboxes);
    memset(pb, 0, bboxesByteSize);
    memset(pg, 0, gtboxesByteSize);
    // [x,y,z,w,h,d,theta]
    SetBox(pb, 0, 0, N, {0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f, 0.0f});
    SetBox(pb, 0, 1, N, {100.0f, 100.0f, 0.0f, 2.0f, 2.0f, 2.0f, 0.0f});
    SetBox(pg, 0, 0, K, {0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f, 0.0f});
    SetBox(pg, 0, 1, K, {100.0f, 100.0f, 0.0f, 2.0f, 2.0f, 2.0f, 0.0f});

    float* pi = reinterpret_cast<float*>(iou);
    memset(pi, 0xFF, iouByteSize); // 预置非零，验证 kernel 确实写回

    Iou3DTilingData* tilingData = reinterpret_cast<Iou3DTilingData*>(tiling);
    FillTiling(tilingData, B, N, K, 0U);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(iou3_d, numBlocks, bboxes, gtboxes, iou, workspace, tiling);

    // 宽松 sanity check：对角（identical）≈1，非对角（disjoint）≈0。
    EXPECT_NEAR(pi[0 * K + 0], 1.0f, 1e-3f); // (i=0,j=0)
    EXPECT_NEAR(pi[1 * K + 1], 1.0f, 1e-3f); // (i=1,j=1)
    EXPECT_NEAR(pi[0 * K + 1], 0.0f, 1e-3f); // (i=0,j=1) disjoint
    EXPECT_NEAR(pi[1 * K + 0], 0.0f, 1e-3f); // (i=1,j=0) disjoint

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(iou);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 旋转相交框对：[1,7,1] × [1,7,1]，theta 差 45° -> 交集为八边形（>3 顶点，触发 Sort32 分支）。
TEST_F(iou3d_test, test_iou3d_rotated_intersection_fp32)
{
    const uint32_t B = 1, N = 1, K = 1;
    size_t bboxesByteSize = B * 7 * N * sizeof(float);
    size_t gtboxesByteSize = B * 7 * K * sizeof(float);
    size_t iouByteSize = B * N * K * sizeof(float);
    size_t tilingDataSize = sizeof(Iou3DTilingData);

    uint8_t* bboxes = (uint8_t*)AscendC::GmAlloc(bboxesByteSize);
    uint8_t* gtboxes = (uint8_t*)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t* iou = (uint8_t*)AscendC::GmAlloc(iouByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    float* pb = reinterpret_cast<float*>(bboxes);
    float* pg = reinterpret_cast<float*>(gtboxes);
    memset(pb, 0, bboxesByteSize);
    memset(pg, 0, gtboxesByteSize);
    SetBox(pb, 0, 0, N, {0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f, 0.0f});
    SetBox(pg, 0, 0, K, {0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f, static_cast<float>(M_PI / 4.0)});

    float* pi = reinterpret_cast<float*>(iou);
    memset(pi, 0, iouByteSize);

    Iou3DTilingData* tilingData = reinterpret_cast<Iou3DTilingData*>(tiling);
    FillTiling(tilingData, B, N, K, 0U);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(iou3_d, numBlocks, bboxes, gtboxes, iou, workspace, tiling);

    // 45° 旋转的等尺寸同心框：BEV 交集为正八边形，3D IoU≈0.707（宽松容差，主要验证不崩溃）。
    EXPECT_GT(pi[0], 0.5f);
    EXPECT_LE(pi[0], 1.0f + 1e-3f);

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(iou);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 空 Tensor：N=0 -> isEmpty=1，Process 运行时短路，不写输出（不崩溃即通过）。
TEST_F(iou3d_test, test_iou3d_empty_tensor)
{
    const uint32_t B = 1, N = 0, K = 4;
    size_t bboxesByteSize = 1 * 7 * 1 * sizeof(float); // N=0 无法分配 0 字节，占位 1
    size_t gtboxesByteSize = B * 7 * K * sizeof(float);
    size_t iouByteSize = 1 * sizeof(float); // 空输出占位
    size_t tilingDataSize = sizeof(Iou3DTilingData);

    uint8_t* bboxes = (uint8_t*)AscendC::GmAlloc(bboxesByteSize);
    uint8_t* gtboxes = (uint8_t*)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t* iou = (uint8_t*)AscendC::GmAlloc(iouByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    memset(bboxes, 0, bboxesByteSize);
    memset(gtboxes, 0, gtboxesByteSize);

    Iou3DTilingData* tilingData = reinterpret_cast<Iou3DTilingData*>(tiling);
    FillTiling(tilingData, B, N, K, 1U);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(iou3_d, numBlocks, bboxes, gtboxes, iou, workspace, tiling);

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(iou);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
