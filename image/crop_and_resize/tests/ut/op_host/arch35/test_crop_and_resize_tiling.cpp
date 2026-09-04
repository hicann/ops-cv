/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "image/crop_and_resize/op_kernel/arch35/crop_and_resize_tiling_data.h"
#include "image/crop_and_resize/op_kernel/arch35/crop_and_resize_tiling_key.h"

using namespace std;
using namespace ge;

struct CropAndResizeUtCompileInfo {};

class CropAndResizeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CropAndResizeTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CropAndResizeTiling TearDown" << std::endl; }
};

static gert::TilingContextPara::TensorDescription MkTd(std::initializer_list<int64_t> shape, ge::DataType dtype,
                                                       bool isConst = false, const void* constValue = nullptr,
                                                       ge::Format format = ge::FORMAT_ND)
{
    return gert::TilingContextPara::TensorDescription(gert::StorageShape(shape, shape), dtype, format, isConst,
                                                      const_cast<void*>(constValue));
}

// 构建标准 TilingContextPara。输入顺序：x(0), boxes(1), box_index(2), crop_size(3，值依赖)。
static gert::TilingContextPara BuildPara(std::initializer_list<int64_t> xShape,
                                         std::initializer_list<int64_t> boxesShape,
                                         std::initializer_list<int64_t> boxIndexShape,
                                         std::initializer_list<int64_t> cropSizeShape, const int32_t* cropSizeData,
                                         ge::DataType xDtype, ge::DataType boxesDtype, ge::DataType yDtype,
                                         float extrapolationValue = 0.0f, uint64_t coreNum = 64,
                                         uint64_t ubSize = 262144, ge::Format xFormat = ge::FORMAT_ND)
{
    static CropAndResizeUtCompileInfo compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        MkTd(xShape, xDtype, false, nullptr, xFormat),                                   // x
        MkTd(boxesShape, boxesDtype),                                                    // boxes
        MkTd(boxIndexShape, ge::DT_INT32),                                               // box_index
        MkTd(cropSizeShape, ge::DT_INT32, true, static_cast<const void*>(cropSizeData)), // crop_size (value dependency)
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MkTd({1, 1, 1, 1}, yDtype), // y (shape not used in tiling)
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        gert::TilingContextPara::OpAttr("extrapolation_value",
                                        Ops::Cv::AnyValue::CreateFrom<float>(extrapolationValue)),
        gert::TilingContextPara::OpAttr("method", Ops::Cv::AnyValue::CreateFrom<std::string>("bilinear")),
    };
    return gert::TilingContextPara("CropAndResize", inputs, outputs, attrs, &compileInfo, "Ascend950", coreNum, ubSize,
                                   4096);
}

// 正例公共校验：x=[2,4,4,256], boxes=[64,4], crop_size={8,8}
//   totalPositions = 64*8*8 = 4096; coreNum=64 -> perCore=1024 -> needCore=4
//   tilingKey = CROP_AND_RESIZE_MODE_BILINEAR_NHWC(0)
static void CheckPositiveTiling(const TilingInfo& info)
{
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_BILINEAR_NHWC);
    EXPECT_EQ(info.blockNum, 4u);
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_GE(info.workspaceSizes[0], 0);
    ASSERT_GE(info.tilingDataSize, sizeof(CropAndResizeTilingData));
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->totalPositions, 4096);
    EXPECT_EQ(td->batch, 2);
    EXPECT_EQ(td->imageHeight, 4);
    EXPECT_EQ(td->imageWidth, 4);
    EXPECT_EQ(td->depth, 256);
    EXPECT_EQ(td->cropHeight, 8);
    EXPECT_EQ(td->cropWidth, 8);
    EXPECT_EQ(td->numBoxes, 64);
    EXPECT_FLOAT_EQ(td->extrapolationValue, 0.0f);
}

// ===== 正例：4 种 dtype 组合 =====
// #0: x=FP32, boxes=FP32, box_index=INT32, crop_size=INT32, y=FP32
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_fp32_fp32)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckPositiveTiling(info);
}

// #1: x=FP16, boxes=FP32, box_index=INT32, crop_size=INT32, y=FP32
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_fp16_fp32)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckPositiveTiling(info);
}

// #2: x=FP16, boxes=FP16, box_index=INT32, crop_size=INT32, y=FP16
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_fp16_fp16)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckPositiveTiling(info);
}

// #3: x=FP32, boxes=FP16, box_index=INT32, crop_size=INT32, y=FP16
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_fp32_fp16)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    CheckPositiveTiling(info);
}

// ===== 负例：约束 1 x 必须 4D =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_x_not_4d)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 2 num_boxes 必须 > 50 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_num_boxes_le_50)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {50, 4}, {50}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 3 C 必须 >= 256 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_depth_lt_256)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 128}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 4 max(crop_h, crop_w) 必须 <= 16 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_h_gt_16)
{
    int32_t cropSize[2] = {17, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 5 H*W 必须 <= 65530 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_hw_gt_65530)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 256, 256, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 9 float32 要求 H*W <= 32765 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_float32_hw_gt_32765)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 200, 200, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 7 x dtype 必须 float16/float32 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_unsupported_dtype)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 10 boxes.shape[0] == box_index.shape[0] =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_boxes_ne_box_index)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {63}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 11 boxes.shape[1] == 4 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_boxes_dim1_ne_4)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 3}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 12 crop_size.shape == (2,) =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_size_shape_ne_2)
{
    int32_t cropSize[3] = {8, 8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {3}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 正例：crop_h=1，验证中心点坐标路径（SE §1.4 crop_height=1 → in_y=0.5*(y1+y2)*(H-1)）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_h_eq_1)
{
    int32_t cropSize[2] = {1, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_BILINEAR_NHWC);
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->cropHeight, 1);
    EXPECT_EQ(td->cropWidth, 8);
    EXPECT_EQ(td->totalPositions, 512); // 64*1*8
}

// ===== 正例：crop_w=1，验证中心点坐标路径 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_w_eq_1)
{
    int32_t cropSize[2] = {8, 1};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_BILINEAR_NHWC);
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->cropHeight, 8);
    EXPECT_EQ(td->cropWidth, 1);
    EXPECT_EQ(td->totalPositions, 512); // 64*8*1
}

// ===== 正例：extrapolation_value 非默认值（SE §1.4 超出边界 → 输出 extrapolation_value）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_extrapolation_nonzero)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 1.5f);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_FLOAT_EQ(td->extrapolationValue, 1.5f);
}

// ===== 负例：约束 2 num_boxes > 4000 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_num_boxes_gt_4000)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {4001, 4}, {4001}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 3 C > 2048 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_depth_gt_2048)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 2049}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 4 crop_w > 16 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_w_gt_16)
{
    int32_t cropSize[2] = {8, 17};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 4 前置 crop_h <= 0（SE §1.4/§5.4 边界条件）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_h_le_0)
{
    int32_t cropSize[2] = {0, 8};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：约束 4 前置 crop_w <= 0（SE §1.4/§5.4 边界条件）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_crop_w_le_0)
{
    int32_t cropSize[2] = {8, 0};
    auto para = BuildPara({2, 4, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 正例：NCHW x=(N,C,H,W)=(2,256,4,4)，dims 按正确 dim 解析 + NCHW TilingKey；若误按 NHWC 解析则 C=4 违反
// [256,2048] 直接失败 =====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_nchw_fp16_fp32)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 256, 4, 4}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                          0.0f, 64, 262144, ge::FORMAT_NCHW);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, CROP_AND_RESIZE_MODE_BILINEAR_NCHW);
    EXPECT_EQ(info.blockNum, 4u);
    const CropAndResizeTilingData* td = reinterpret_cast<const CropAndResizeTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->depth, 256);     // C 来自 x.shape[1]（NCHW dims[1]）
    EXPECT_EQ(td->imageHeight, 4); // H 来自 x.shape[2]
    EXPECT_EQ(td->imageWidth, 4);  // W 来自 x.shape[3]
    EXPECT_EQ(td->batch, 2);
    EXPECT_EQ(td->totalPositions, 4096); // 64*8*8，与 layout 无关
}

// ===== 负例：NCHW depth 约束在 dim1 生效（C=x.shape[1]=255 < 256）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_nchw_depth_lt_256)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 255, 4, 4}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                          0.0f, 64, 262144, ge::FORMAT_NCHW);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：非法 format（HWCN 非 ND/NHWC/NCHW，ExtractInputInfo 新增拦截分支）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_invalid_format)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 256, 4, 4}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                          0.0f, 64, 262144, ge::FORMAT_HWCN);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：x 含 -1（动态维）。check_supported 已前置 fallback AiCpu，此为 AiCore 认领后
// tiling 阶段的兜底校验（all-dims-positive 拒绝；N/H/W/C 各维 -1 均走此分支，取 H=-1 代表）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_x_dim_unknown)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, -1, 4, 256}, {64, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ===== 负例：num_boxes=-1（boxes.shape[0] 动态），tiling 兜底拒绝（区间检查分支）=====
TEST_F(CropAndResizeTiling, crop_and_resize_tiling_num_boxes_unknown)
{
    int32_t cropSize[2] = {8, 8};
    auto para = BuildPara({2, 4, 4, 256}, {-1, 4}, {64}, {2}, cropSize, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
