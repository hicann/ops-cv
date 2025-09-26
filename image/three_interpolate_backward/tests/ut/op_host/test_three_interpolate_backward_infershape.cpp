/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "experiment_ops.h"
#include "common/utils/ut_op_common.h"

class ThreeInterpolateGradProtoTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ThreeInterpolateGradProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ThreeInterpolateGradProtoTest TearDown" << std::endl;
    }
};

// fp16 infer dtype
TEST_F(ThreeInterpolateGradProtoTest, ThreeInterpolateGrad_inferdatatype_test01)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ThreeInterpolateBackward"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ThreeInterpolateBackward")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType featureType = ge::DT_FLOAT16;
        ge::DataType indexType = ge::DT_INT32;
        ge::DataType weightType = ge::DT_FLOAT16;
        ge::DataType outType = ge::DT_FLOAT16; // 初始化的时候设置为未定义的类型

        auto holder = gert::InferDataTypeContextFaker()
                          .IrInputNum(3)
                          .NodeIoNum(3, 1)
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .InputDataTypes({&featureType, &indexType, &weightType})
                          .OutputDataTypes({&outType})
                          .Build();

        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), outType);
    }
}

// fp32 infer dtype
TEST_F(ThreeInterpolateGradProtoTest, ThreeInterpolateGrad_inferdatatype_test02)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ThreeInterpolateBackward"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ThreeInterpolateBackward")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType featureType = ge::DT_FLOAT;
        ge::DataType indexType = ge::DT_INT32;
        ge::DataType weightType = ge::DT_FLOAT;
        ge::DataType outType = ge::DT_FLOAT; // 初始化的时候设置为未定义的类型

        auto holder = gert::InferDataTypeContextFaker()
                          .IrInputNum(3)
                          .NodeIoNum(3, 1)
                          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .InputDataTypes({&featureType, &indexType, &weightType})
                          .OutputDataTypes({&outType})
                          .Build();

        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), outType);
    }
}