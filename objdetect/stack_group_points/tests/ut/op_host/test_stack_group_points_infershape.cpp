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
#include "common/utils/ut_op_common.h"
#include "../../../op_graph/stack_group_points_proto.h"

class StackGroupPoints : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "StackGroupPoints Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StackGroupPoints Proto Test TearDown" << std::endl;
    }
};

TEST_F(StackGroupPoints, StackGroupPoints_infershape_case_0)
{
    ge::op::StackGroupPoints op;
    op.UpdateInputDesc("features", create_desc({32, 64}, ge::DT_FLOAT));
    op.UpdateInputDesc("indices", create_desc({20, 3}, ge::DT_INT32));

    op.UpdateInputDesc("indfeatures_batch_cntices", create_desc({4}, ge::DT_INT32));
    op.UpdateInputDesc("indices_batch_cnt", create_desc({4}, ge::DT_INT32));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    std::vector<int64_t> expected_output_shape = {20, 64, 3};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(StackGroupPoints, StackGroupPoints_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StackGroupPoints"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StackGroupPoints")->infer_datatype;
    if (data_type_func != nullptr)
    {
        ge::DataType features_ref = ge::DT_FLOAT;
        ge::DataType indices_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&features_ref, &indices_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), features_ref);
        EXPECT_EQ(context->GetInputDataType(1), indices_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}