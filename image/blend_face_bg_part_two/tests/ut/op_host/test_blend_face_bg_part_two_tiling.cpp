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
 * \file test_blend_face_bg_part_two_tiling.cpp
 * \brief BlendFaceBgPartTwo arch35 host Tiling UT（迭代 2，覆盖 FP32 + UINT8 分支）。
 *
 * 覆盖点（对齐迭代 2 scope wave2.a2）：
 *   - blockFormer 512 元素对齐 + 每核 >= 4KB 多核切分逻辑；
 *   - ubFormer 256B（64 fp32 元素）对齐 UB 分块；
 *   - 首/尾 block 循环次数与尾块大小（含单 block「首==尾」、多 block 首尾不同、非对齐尾块）；
 *   - TilingKey 选择（数据量足够 → 双缓冲 BUFFER_MODE=1；空 Tensor → 单缓冲 BUFFER_MODE=0）；
 *   - **bg_img dtype 分支**（DT_UINT8 → bufferDivisor=34 vs DT_FLOAT → bufferDivisor=40）；
 *   - epsilon 属性透传；workspace slot（0）声明。
 *
 * 说明：核数由 tiling 运行时 GetCoreNumAiv 决定，故不硬编码 coreNum/blockFormer 绝对值，
 * 而是以 tiling 代码承诺的数学关系做字段级断言（对齐 / CeilDiv / 首尾块重建 dim0），
 * 这样在不同平台 faker 核数下仍稳定，同时精确验证首尾块计算逻辑。
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "blend_face_bg_part_two_tiling_data.h"

namespace BlendFaceBgPartTwoUT {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "BlendFaceBgPartTwo";

// 切分常量（须与 op_host/arch35/blend_face_bg_part_two_tiling.cpp 保持一致）
static constexpr int64_t ELEM_ALIGN_FACTOR = 512;   // blockFormer 元素对齐因子
static constexpr int64_t FP32_ALIGN_ELEM = 256 / 4; // ubFormer 对齐（256B → 64 fp32 元素）
static constexpr uint64_t KEY_DOUBLE_BUFFER = 1UL;  // 数据量足够 → 双缓冲
static constexpr uint64_t KEY_SINGLE_BUFFER = 0UL;  // 空 Tensor → 单缓冲

static inline int64_t CeilDiv(int64_t a, int64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }

// 与 tiling 侧 TilingParse 的 CompileInfo 类型对应（空结构体即可，Build 需非空指针）
struct BlendFaceBgPartTwoCompileInfo {
} g_compileInfo;

// 构造四输入同 shape（前 3 输入 fp32 + bg_img 可变 dtype）+ 输出 fp32 + epsilon 属性的 tiling 上下文
static gert::TilingContextPara MakePara(const std::initializer_list<int64_t>& shape, float epsilon, uint64_t coreNum,
                                        uint64_t ubSize, ge::DataType bgImgDtype = ge::DT_FLOAT)
{
    gert::StorageShape s = {shape, shape};
    std::vector<gert::TilingContextPara::TensorDescription> inputs(
        {{s, ge::DT_FLOAT, ge::FORMAT_ND},
         {s, ge::DT_FLOAT, ge::FORMAT_ND},
         {s, ge::DT_FLOAT, ge::FORMAT_ND},
         {s, bgImgDtype, ge::FORMAT_ND}}); // bg_img dtype 可变（DT_FLOAT / DT_UINT8）
    std::vector<gert::TilingContextPara::TensorDescription> outputs({{s, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::OpAttr> attrs({{"epsilon", Ops::Math::AnyValue::CreateFrom<float>(epsilon)}});
    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, &g_compileInfo, coreNum, ubSize, 4096);
}

// 运行 tiling 并把 raw tiling data 还原为结构体
static const BlendFaceBgPartTwoTilingData* AsTilingData(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(BlendFaceBgPartTwoTilingData));
    return reinterpret_cast<const BlendFaceBgPartTwoTilingData*>(info.tilingData.get());
}

// 对成功路径统一校验首/尾块计算的数学关系
static void CheckTilingInvariants(const BlendFaceBgPartTwoTilingData* t, int64_t expectDim0)
{
    ASSERT_NE(t, nullptr);
    // dim0
    EXPECT_EQ(t->dim0, expectDim0);
    // 核数 >= 1
    EXPECT_GE(t->coreNum, 1);
    // blockFormer 512 元素对齐且 > 0
    EXPECT_GT(t->blockFormer, 0);
    EXPECT_EQ(t->blockFormer % ELEM_ALIGN_FACTOR, 0);
    // blockNum = CeilDiv(dim0, blockFormer)
    EXPECT_EQ(t->blockNum, CeilDiv(expectDim0, t->blockFormer));
    // ubFormer 256B（64 fp32 元素）对齐且 > 0
    EXPECT_GT(t->ubFormer, 0);
    EXPECT_EQ(t->ubFormer % FP32_ALIGN_ELEM, 0);
    // 首 block：循环次数与尾块大小
    EXPECT_EQ(t->ubLoopOfFormerBlock, CeilDiv(t->blockFormer, t->ubFormer));
    int64_t expectUbTailFormer = t->blockFormer - (t->ubLoopOfFormerBlock - 1) * t->ubFormer;
    EXPECT_EQ(t->ubTailOfFormerBlock, expectUbTailFormer);
    EXPECT_GT(t->ubTailOfFormerBlock, 0);
    EXPECT_LE(t->ubTailOfFormerBlock, t->ubFormer);
    // 尾 block：blockTail = dim0 - (blockNum-1)*blockFormer
    int64_t blockTail = expectDim0 - (t->blockNum - 1) * t->blockFormer;
    EXPECT_GT(blockTail, 0);
    EXPECT_LE(blockTail, t->blockFormer);
    EXPECT_EQ(t->ubLoopOfTailBlock, CeilDiv(blockTail, t->ubFormer));
    int64_t expectUbTailTail = blockTail - (t->ubLoopOfTailBlock - 1) * t->ubFormer;
    EXPECT_EQ(t->ubTailOfTailBlock, expectUbTailTail);
    EXPECT_GT(t->ubTailOfTailBlock, 0);
    EXPECT_LE(t->ubTailOfTailBlock, t->ubFormer);
    // 用首/尾块重建 dim0，验证切分自洽
    int64_t rebuilt = (t->blockNum - 1) * t->blockFormer + blockTail;
    EXPECT_EQ(rebuilt, expectDim0);
}

class BlendFaceBgPartTwoTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BlendFaceBgPartTwoTiling SetUp." << std::endl; }
    static void TearDownTestCase() { std::cout << "BlendFaceBgPartTwoTiling TearDown." << std::endl; }
};

// case 1：小 shape 单 block（首块==尾块），验证基础切分 + 双缓冲 TilingKey + epsilon 透传
TEST_F(BlendFaceBgPartTwoTiling, tiling_single_block_small_shape)
{
    auto para = MakePara({4, 4, 3}, 1.0e-12f, 64, 262144); // dim0 = 48
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_DOUBLE_BUFFER));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 48);
    // 数据量很小 → 单 block（首块与尾块为同一 block）
    EXPECT_EQ(t->blockNum, 1);
    EXPECT_FLOAT_EQ(t->epsilon, 1.0e-12f);
    // 单 block 场景：blockFormer 被 512 对齐上取整（512 > dim0=48），
    // 故首块统计按 blockFormer(512) 计算，尾块按真实 blockTail(48) 计算，二者不必相等。
    // 尾块（唯一实际 block）的元素数应等于 dim0。
    int64_t blockTail = t->dim0 - (t->blockNum - 1) * t->blockFormer;
    EXPECT_EQ(blockTail, 48);
    EXPECT_EQ(t->ubTailOfTailBlock, 48);
}

// case 2：大 shape 多 block，dim0 = 512*multiple 且整除 → 尾块为满 block，验证多核首尾块一致
TEST_F(BlendFaceBgPartTwoTiling, tiling_multi_block_aligned)
{
    // dim0 = 512 * 40 = 20480，跨多核；每核 >=4KB 切分生效
    auto para = MakePara({160, 128, 1}, 5.0e-5f, 64, 262144); // dim0 = 20480
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_DOUBLE_BUFFER));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 20480);
    EXPECT_FLOAT_EQ(t->epsilon, 5.0e-5f);
    // blockDim（框架 blockNum）应等于 tiling 结构体 blockNum
    EXPECT_EQ(static_cast<int64_t>(info.blockNum), t->blockNum);
}

// case 3：非 512 对齐 dim0 → 尾块非满，验证尾块尺寸计算（首块与尾块循环/尾数可能不同）
TEST_F(BlendFaceBgPartTwoTiling, tiling_multi_block_unaligned_tail)
{
    // dim0 = 33333（非 512 对齐，且足够大触发多核）
    auto para = MakePara({33333, 1, 1}, 1.0e-12f, 64, 262144);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_DOUBLE_BUFFER));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 33333);
    // 尾块严格小于满 block（非对齐场景）
    int64_t blockTail = t->dim0 - (t->blockNum - 1) * t->blockFormer;
    EXPECT_LT(blockTail, t->blockFormer);
}

// case 4：ubFormer 分块循环触发（小 UB → 每 block 需多次 UB 循环），验证 UB 切分与首块尾块
TEST_F(BlendFaceBgPartTwoTiling, tiling_ub_multi_loop_small_ub)
{
    // ubSize 调小，使 ubFormer < blockFormer，强制 ubLoopOfFormerBlock > 1
    // ubSize=8192 → maxElem = 8192/40 = 204 → ubFormer = FloorAlign(204,64)=192
    auto para = MakePara({8, 8, 64}, 1.0e-12f, 8, 8192); // dim0 = 4096
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_DOUBLE_BUFFER));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 4096);
    EXPECT_EQ(t->ubFormer, 192);          // FloorAlign(8192/40, 64)
    EXPECT_GT(t->ubLoopOfFormerBlock, 1); // 需要多次 UB 循环
}

// case 5：空 Tensor（dim0=0）→ 单缓冲 TilingKey，单核零操作路径
TEST_F(BlendFaceBgPartTwoTiling, tiling_empty_tensor)
{
    auto para = MakePara({0, 4, 3}, 1.0e-12f, 64, 262144); // dim0 = 0
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_SINGLE_BUFFER));
    const auto* t = AsTilingData(info);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->dim0, 0);
    EXPECT_EQ(t->coreNum, 1);
    EXPECT_EQ(t->blockFormer, 0);
    EXPECT_EQ(t->blockNum, 1);
    EXPECT_EQ(t->ubFormer, 0);
    EXPECT_EQ(static_cast<int64_t>(info.blockNum), 1);
}

// case 6：workspace slot 声明为 0（即使无 workspace 也须声明 1 个 slot，值为 0）
TEST_F(BlendFaceBgPartTwoTiling, tiling_workspace_slot)
{
    auto para = MakePara({4, 4, 3}, 1.0e-12f, 64, 262144);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], 0u);
}

// ========== 迭代 2 新增：uint8 分支覆盖（补齐 TilingKey 选择 + bufferDivisor 分支） ==========

// case 7：bg_img=uint8 单 block，验证 TilingKey 选择（仍为 BUFFER_MODE，与 dtype 无关）+ bufferDivisor=34
TEST_F(BlendFaceBgPartTwoTiling, tiling_uint8_single_block)
{
    auto para = MakePara({4, 4, 3}, 1.0e-12f, 64, 262144, ge::DT_UINT8); // dim0 = 48, bg_img=uint8
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_DOUBLE_BUFFER));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 48);
    EXPECT_EQ(t->blockNum, 1);
    EXPECT_FLOAT_EQ(t->epsilon, 1.0e-12f);
    // uint8 分支：bufferDivisor=34（vs fp32 的 40），相同 ubSize 下 ubFormer 应更大
    // ubFormer = FloorAlign(ubSize/34, 64) vs FloorAlign(ubSize/40, 64)
    // 262144/34=7710 → FloorAlign(7710,64)=7680 vs 262144/40=6553 → FloorAlign(6553,64)=6528
    EXPECT_EQ(t->ubFormer, 7680); // uint8 分支 ubFormer 更大（34 < 40）
}

// case 8：bg_img=uint8 多 block，验证 bufferDivisor=34 下的多核切分与 UB 分块
TEST_F(BlendFaceBgPartTwoTiling, tiling_uint8_multi_block)
{
    // dim0 = 20480（足够大触发多核），bg_img=uint8
    auto para = MakePara({160, 128, 1}, 1.0e-12f, 64, 262144, ge::DT_UINT8);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_DOUBLE_BUFFER));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 20480);
    // uint8 分支 minDtypeBits=8（vs fp32=32），相同 dim0 下核数计算受影响：
    // coreNum = min(CeilDiv(dim0 * minDtypeBits, 32768), availableCoreNum)
    // uint8: CeilDiv(20480*8, 32768) = CeilDiv(163840, 32768) = 5
    // fp32:  CeilDiv(20480*32, 32768) = CeilDiv(655360, 32768) = 20
    // 故 uint8 分支核数更少（数据量按 byte 计更小）
    EXPECT_GE(t->coreNum, 1);
    EXPECT_EQ(t->ubFormer, 7680); // uint8 分支 ubFormer（bufferDivisor=34）
}

// case 9：bg_img=uint8 小 UB 多次循环，验证 bufferDivisor=34 下 UB 切分
TEST_F(BlendFaceBgPartTwoTiling, tiling_uint8_ub_multi_loop_small_ub)
{
    // ubSize=8192 较小，强制多次 UB 循环
    // uint8: maxElem = 8192/34 = 241 → ubFormer = FloorAlign(241,64) = 192
    // fp32:  maxElem = 8192/40 = 204 → ubFormer = FloorAlign(204,64) = 192（相同结果，因对齐）
    auto para = MakePara({8, 8, 64}, 1.0e-12f, 8, 8192, ge::DT_UINT8); // dim0 = 4096
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_DOUBLE_BUFFER));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 4096);
    EXPECT_EQ(t->ubFormer, 192);          // 与 fp32 巧合相同（256B 对齐约束主导）
    EXPECT_GT(t->ubLoopOfFormerBlock, 1); // 需要多次 UB 循环
}

// case 10：bg_img=uint8 空 Tensor，验证单缓冲 TilingKey（与 fp32 行为一致）
TEST_F(BlendFaceBgPartTwoTiling, tiling_uint8_empty_tensor)
{
    auto para = MakePara({0, 4, 3}, 1.0e-12f, 64, 262144, ge::DT_UINT8); // dim0 = 0
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_SINGLE_BUFFER));
    const auto* t = AsTilingData(info);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->dim0, 0);
    EXPECT_EQ(t->coreNum, 1);
    EXPECT_EQ(t->blockFormer, 0);
    EXPECT_EQ(t->blockNum, 1);
    EXPECT_EQ(t->ubFormer, 0);
}

// case 11：bg_img=uint8 非对齐尾块，验证首/尾 block 计算与 bufferDivisor=34
TEST_F(BlendFaceBgPartTwoTiling, tiling_uint8_unaligned_tail)
{
    // dim0 = 33333（非 512 对齐），bg_img=uint8
    auto para = MakePara({33333, 1, 1}, 1.0e-12f, 64, 262144, ge::DT_UINT8);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(KEY_DOUBLE_BUFFER));
    const auto* t = AsTilingData(info);
    CheckTilingInvariants(t, 33333);
    // 尾块严格小于满 block
    int64_t blockTail = t->dim0 - (t->blockNum - 1) * t->blockFormer;
    EXPECT_LT(blockTail, t->blockFormer);
    EXPECT_EQ(t->ubFormer, 7680); // uint8 分支 ubFormer
}

} // namespace BlendFaceBgPartTwoUT
