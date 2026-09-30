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
 * \file test_aipp_infer_pass.cpp
 * \brief Aipp InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForAipp(gert::InferDataTypeContext* context);
}

namespace {
const char* kStaticCfg = R"({"aipp_mode":"static"})";
const char* kDynamicCfg = R"({"aipp_mode":"dynamic","max_src_image_size":752640})";

gert::ContextHolder<gert::InferDataTypeContext> BuildAippDtypeContext(ge::DataType outDtype, const char* cfg,
                                                                      ge::DataType imagesDtype = ge::DT_UINT8)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Aipp").OpName("Aipp").IONum(2, 1);
    builder.InputTensorDesc(0, imagesDtype, ge::FORMAT_NHWC, ge::FORMAT_NHWC);
    builder.OutputTensorDesc(0, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.AppendAttr(ge::AscendString(cfg));
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    EXPECT_NE(context, nullptr);
    if (context != nullptr) {
        EXPECT_EQ(context->SetOutputDataType(0, outDtype), ge::GRAPH_SUCCESS);
    }
    return holder;
}
} // namespace

TEST(AippGraphInfer, InferDataTypeStaticKeepFp16)
{
    auto holder = BuildAippDtypeContext(ge::DT_FLOAT16, kStaticCfg);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForAipp(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(AippGraphInfer, InferDataTypeStaticKeepUint8)
{
    auto holder = BuildAippDtypeContext(ge::DT_UINT8, kStaticCfg);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForAipp(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_UINT8);
}

TEST(AippGraphInfer, InferDataTypeStaticKeepUndefined)
{
    auto holder = BuildAippDtypeContext(ge::DT_UNDEFINED, kStaticCfg);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForAipp(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_UNDEFINED);
}

TEST(AippGraphInfer, InferDataTypeStaticKeepFloat)
{
    auto holder = BuildAippDtypeContext(ge::DT_FLOAT, kStaticCfg);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForAipp(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

// RT1 DynamicModeInfershape copies the images desc onto features, so features dtype follows images.
TEST(AippGraphInfer, InferDataTypeDynamicFp16FollowsImagesUint8)
{
    auto holder = BuildAippDtypeContext(ge::DT_FLOAT16, kDynamicCfg, ge::DT_UINT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForAipp(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_UINT8);
}

TEST(AippGraphInfer, InferDataTypeDynamicFollowsImagesFp16)
{
    auto holder = BuildAippDtypeContext(ge::DT_UINT8, kDynamicCfg, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForAipp(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(AippGraphInfer, InferDataTypeRejectBadMode)
{
    auto holder = BuildAippDtypeContext(ge::DT_FLOAT16, R"({"aipp_mode":"xx"})");
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(ops::InferDataTypeForAipp(context), ge::GRAPH_FAILED);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(AippGraphInfer, RejectsNullContext) { EXPECT_EQ(ops::InferDataTypeForAipp(nullptr), ge::GRAPH_FAILED); }
