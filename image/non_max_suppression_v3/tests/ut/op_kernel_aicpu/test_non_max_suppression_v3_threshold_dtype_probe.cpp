/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Host/AICPU regression test for NonMaxSuppressionV3 threshold dtype validation.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "Eigen/Core"
#include "allocator_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/aicpu_test_utils.h"

using namespace aicpu;

namespace {
struct HalfWithGuard {
    Eigen::half value;
    uint16_t guard;
};

uint32_t RunCase(const char* name, DataType iou_dtype, DataType score_threshold_dtype, void* iou_data,
                 void* score_threshold_data, std::vector<int32_t>& selected)
{
    selected.clear();
    std::vector<float> boxes = {0.0F, 0.0F, 1.0F, 1.0F, 10.0F, 10.0F, 11.0F, 11.0F};
    std::vector<float> scores = {0.8F, 0.6F};
    int32_t max_output_size = 2;
    uint64_t result_summary[4] = {0, 0, 0, 0};

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "NonMaxSuppressionV3", "NonMaxSuppressionV3")
        .Input({"boxes", DT_FLOAT, {2, 4}, boxes.data()})
        .Input({"scores", DT_FLOAT, {2}, scores.data()})
        .Input({"max_output_size", DT_INT32, {}, &max_output_size})
        .Input({"iou_threshold", iou_dtype, {}, iou_data})
        .Input({"score_threshold", score_threshold_dtype, {}, score_threshold_data})
        .Output({"selected_indices", DT_UINT64, {-1}, result_summary});

    std::string node_def_str;
    node_def->SerializeToString(node_def_str);
    CpuKernelContext ctx(HOST);
    uint32_t init_ret = ctx.Init(node_def.get());
    if (init_ret != KERNEL_STATUS_OK) {
        std::cout << "case=" << name << " init_ret=" << init_ret << std::endl;
        return init_ret;
    }
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    std::cout << "case=" << name << " ret=" << ret;
    if (ret == KERNEL_STATUS_OK && result_summary[2] != 0) {
        const auto count = result_summary[3] / sizeof(int32_t);
        const auto* data = reinterpret_cast<const int32_t*>(result_summary[2]);
        selected.assign(data, data + count);
        std::cout << " output_count=" << count << " output=";
        for (auto value : selected) {
            std::cout << value << ",";
        }
    }
    std::cout << std::endl;
    return ret;
}
} // namespace

TEST(NonMaxSuppressionV3ThresholdDtypeProbe, MixedThresholdDtypesAreRejected)
{
    float iou_float = 0.5F;
    float score_float = 0.75F;
    HalfWithGuard iou_half = {Eigen::half(0.5F), 0};
    HalfWithGuard score_half = {Eigen::half(0.75F), 0};
    std::vector<int32_t> output;

    const auto baseline_ret = RunCase("float_float_baseline", DT_FLOAT, DT_FLOAT, &iou_float, &score_float, output);
    ASSERT_EQ(baseline_ret, KERNEL_STATUS_OK);
    ASSERT_EQ(output, (std::vector<int32_t>{0}));

    const auto half_half_ret = RunCase("half_half_baseline", DT_FLOAT16, DT_FLOAT16, &iou_half.value, &score_half.value,
                                       output);
    ASSERT_EQ(half_half_ret, KERNEL_STATUS_OK);
    ASSERT_EQ(output, (std::vector<int32_t>{0}));

    const auto float_half_ret = RunCase("float_half_mixed", DT_FLOAT, DT_FLOAT16, &iou_float, &score_half.value,
                                        output);
    ASSERT_EQ(float_half_ret, KERNEL_STATUS_PARAM_INVALID);

    const auto half_float_ret = RunCase("half_float_mixed", DT_FLOAT16, DT_FLOAT, &iou_half.value, &score_float,
                                        output);
    ASSERT_EQ(half_float_ret, KERNEL_STATUS_PARAM_INVALID);
}
