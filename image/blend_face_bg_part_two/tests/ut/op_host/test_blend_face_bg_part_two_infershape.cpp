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
 * \file test_blend_face_bg_part_two_infershape.cpp
 * \brief BlendFaceBgPartTwo InferShape UT（迭代 1，KEY_FP32 核心路径）。
 *
 * 覆盖点：
 *   - rank=3 正常路径：输出 fused_img shape = acc_face shape（OneInOneOut）；
 *   - rank≠3 非法输入：infershape 返回 GRAPH_FAILED（rank=2 场景）；
 *   - unknown rank（-2）透传：输出继承 unknown rank，不做 rank 校验。
 *
 * 四输入（acc_face/acc_mask/max_mask/bg_img）均 fp32；infershape 只依赖 acc_face(idx=0)。
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace BlendFaceBgPartTwoUT {
using namespace std;

static const std::string OP_NAME = "BlendFaceBgPartTwo";

// 构造四输入同 shape + 一输出的 infershape 上下文
static gert::InfershapeContextPara MakePara(const gert::StorageShape& in, const gert::StorageShape& out)
{
    return gert::InfershapeContextPara(OP_NAME,
                                       {
                                           {in, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {in, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {in, ge::DT_FLOAT, ge::FORMAT_ND},
                                           {in, ge::DT_FLOAT, ge::FORMAT_ND},
                                       },
                                       {
                                           {out, ge::DT_FLOAT, ge::FORMAT_ND},
                                       });
}

class BlendFaceBgPartTwoInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BlendFaceBgPartTwoInfershape SetUp." << std::endl; }
    static void TearDownTestCase() { std::cout << "BlendFaceBgPartTwoInfershape TearDown." << std::endl; }
};

// case 1：rank=3 正常路径 → 输出 shape = acc_face shape
TEST_F(BlendFaceBgPartTwoInfershape, infershape_rank3_success)
{
    auto para = MakePara({{4, 4, 3}, {4, 4, 3}}, {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 2：rank=3 动态维 → 输出继承动态维
TEST_F(BlendFaceBgPartTwoInfershape, infershape_rank3_dynamic_success)
{
    auto para = MakePara({{-1, 128, 3}, {-1, 128, 3}}, {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 128, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 3：rank=2 非法输入 → GRAPH_FAILED
TEST_F(BlendFaceBgPartTwoInfershape, infershape_rank2_failed)
{
    auto para = MakePara({{2, 3}, {2, 3}}, {{}, {}});
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// case 4：rank=4 非法输入 → GRAPH_FAILED
TEST_F(BlendFaceBgPartTwoInfershape, infershape_rank4_failed)
{
    auto para = MakePara({{2, 4, 4, 3}, {2, 4, 4, 3}}, {{}, {}});
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// case 5：unknown rank（-2）透传 → 输出继承 unknown rank，成功
TEST_F(BlendFaceBgPartTwoInfershape, infershape_unknown_rank_passthrough)
{
    auto para = MakePara({{-2}, {-2}}, {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

} // namespace BlendFaceBgPartTwoUT
