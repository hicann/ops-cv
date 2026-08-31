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
 * \file test_crop_and_resize_infershape.cpp
 * \brief Infershape UT for crop_and_resize operator (new-style framework)
 *
 * 输入顺序: x(0), boxes(1), box_index(2), crop_size(3)
 * 值依赖: crop_size (input index 3)，数据类型 int32_t
 * 输出 shape: [num_boxes, crop_height, crop_width, depth]
 *   num_boxes = boxes.shape[0]
 *   crop_height = crop_size 数据值[0]
 *   crop_width = crop_size 数据值[1]
 *   depth = x.shape[3]
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class CropAndResizeInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CropAndResizeInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CropAndResizeInfershape TearDown" << std::endl; }
};

// ==================== 正常用例 (GRAPH_SUCCESS) ====================

// test01: fp16 x, fp32 boxes, crop_size=[14,14] → y={2,14,14,2}
TEST_F(CropAndResizeInfershape, test01_normal_fp16)
{
    std::vector<int32_t> cropSizeValues = {14, 14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            // x (index 0): 4D NHWC
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // boxes (index 1): 2D, shape[1]==4
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // box_index (index 2): 1D
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            // crop_size (index 3): 值依赖, shape==(2,), values=[14,14]
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            // y (output 0): dtype=boxes.dtype=fp32
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 14, 14, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// test03: crop_h=1 (最小 crop_h 边界) → y={2,1,14,2}
TEST_F(CropAndResizeInfershape, test03_crop_h_eq_1)
{
    std::vector<int32_t> cropSizeValues = {1, 14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 1, 14, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// test04: crop_w=1 (最小 crop_w 边界) → y={2,14,1,2}
TEST_F(CropAndResizeInfershape, test04_crop_w_eq_1)
{
    std::vector<int32_t> cropSizeValues = {14, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 14, 1, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// test05: crop_h=16 (最大 crop 边界) → y={2,16,16,2}
TEST_F(CropAndResizeInfershape, test05_crop_h_eq_16)
{
    std::vector<int32_t> cropSizeValues = {16, 16};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 16, 16, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 失败用例 (GRAPH_FAILED) ====================

// test06: x 为 3D (非 4D) → FAIL
TEST_F(CropAndResizeInfershape, test06_x_not_4d)
{
    std::vector<int32_t> cropSizeValues = {14, 14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// test07: x 为 5D (非 4D) → FAIL
TEST_F(CropAndResizeInfershape, test07_x_5d)
{
    std::vector<int32_t> cropSizeValues = {14, 14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2, 1}, {2, 3, 4, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// test08: boxes.shape[1]!=4 → FAIL
TEST_F(CropAndResizeInfershape, test08_boxes_dim1_ne_4)
{
    std::vector<int32_t> cropSizeValues = {14, 14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// test09: boxes 为 3D (非 2D) → FAIL
TEST_F(CropAndResizeInfershape, test09_boxes_not_2d)
{
    std::vector<int32_t> cropSizeValues = {14, 14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4, 1}, {2, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// test10: crop_size shape=(1,) (长度!=2) → FAIL
TEST_F(CropAndResizeInfershape, test10_crop_size_shape_ne_2)
{
    std::vector<int32_t> cropSizeValues = {14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// test11: crop_size 为 2D (非 1D) → FAIL
TEST_F(CropAndResizeInfershape, test11_crop_size_2d)
{
    std::vector<int32_t> cropSizeValues = {14, 14, 14, 14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// test12: crop_h=0 (<=0) → FAIL
TEST_F(CropAndResizeInfershape, test12_crop_h_le_0)
{
    std::vector<int32_t> cropSizeValues = {0, 14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// test13: crop_w=0 (<=0) → FAIL
TEST_F(CropAndResizeInfershape, test13_crop_w_le_0)
{
    std::vector<int32_t> cropSizeValues = {14, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// test14: crop_h=17 (>16) → SUCCESS：crop<=16 上限属 AiCore tiling 约束，
// 已从 infershape 移至 def.cpp CheckIfAICoreSupported 引擎路由层拦截（AiCpu 可承接）
TEST_F(CropAndResizeInfershape, test14_crop_h_gt_16)
{
    std::vector<int32_t> cropSizeValues = {17, 14};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 17, 14, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// test15: crop_w=17 (>16) → SUCCESS：同 test14，约束归引擎路由层
TEST_F(CropAndResizeInfershape, test15_crop_w_gt_16)
{
    std::vector<int32_t> cropSizeValues = {14, 17};
    gert::InfershapeContextPara infershapeContextPara(
        "CropAndResize",
        {
            {{{2, 3, 4, 2}, {2, 3, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 14, 17, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
