/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_SCALE_AND_TRANSLATE_H_
#define AICPU_KERNELS_NORMALIZED_SCALE_AND_TRANSLATE_H_

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "sampling_kernels.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {

class ScaleAndTranslateCpuKernel : public CpuKernel {
public:
    ScaleAndTranslateCpuKernel() = default;
    ~ScaleAndTranslateCpuKernel() override = default;

protected:
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    static uint32_t ScaleAndTranslateCheck(CpuKernelContext &ctx);

    template <typename T>
    static uint32_t ScaleAndTranslateCompute(CpuKernelContext &ctx);
};

struct Spans {
    // The maximum span size of any output pixel.
    int span_size;
    // int32 tensor of size [output_dim].
    Eigen::Tensor<int32_t, 1> *starts;
    // float tensor of size [output_dim, span_size].
    Eigen::Tensor<float, 1> *weights;
};

template <typename T>
struct GatherSpans {
    uint32_t operator()(aicpu::CpuKernelContext &context, int row_span_size,
                        Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> row_starts,
                        Eigen::TensorMap<Eigen::Tensor<float, 1>> row_weights,
                        int col_span_size,
                        Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> col_starts,
                        Eigen::TensorMap<Eigen::Tensor<float, 1>> col_weights,
                        typename TTypes<T, 4>::Tensor input_images,
                        Eigen::TensorMap<Eigen::Tensor<float, 4>> intermediate_buffer,
                        typename TTypes<float, 4>::Tensor output_images);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_SCALE_AND_TRANSLATE_H_
