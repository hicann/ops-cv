/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <utility>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

using TensorDesc = gert::TilingContextPara::TensorDescription;

struct CIoUCompileInfo {};

gert::TilingContextPara MakeContext(ge::DataType bboxesDtype, ge::DataType gtboxesDtype)
{
    static CIoUCompileInfo compileInfo;
    gert::StorageShape boxesShape = {{4, 8}, {4, 8}};
    gert::StorageShape outShape = {{1, 8}, {1, 8}};
    return gert::TilingContextPara(
        "CIoU",
        {TensorDesc(boxesShape, bboxesDtype, ge::FORMAT_ND), TensorDesc(boxesShape, gtboxesDtype, ge::FORMAT_ND)},
        {TensorDesc(outShape, bboxesDtype, ge::FORMAT_ND), TensorDesc(outShape, bboxesDtype, ge::FORMAT_ND)},
        &compileInfo);
}

} // namespace

TEST(CIoUTilingDtypeValidation, accepts_matching_input_dtypes)
{
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(MakeContext(dtype, dtype), info));
    }
}

TEST(CIoUTilingDtypeValidation, rejects_mixed_input_dtypes)
{
    for (const auto dtypes : {std::pair{ge::DT_FLOAT, ge::DT_FLOAT16}, std::pair{ge::DT_FLOAT16, ge::DT_FLOAT}}) {
        ExecuteTestCase(MakeContext(dtypes.first, dtypes.second), ge::GRAPH_FAILED);
    }
}
