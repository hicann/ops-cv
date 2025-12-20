/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_NON_MAX_SUPPRESSION_V3_KERNELS_H
#define AICPU_KERNELS_NORMALIZED_NON_MAX_SUPPRESSION_V3_KERNELS_H

#include "cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {
struct CorrectBox {
  int ymin_index;
  int xmin_index;
  int ymax_index;
  int xmax_index;
};
class NonMaxSuppressionV3CpuKernel : public CpuKernel {
 public:
  ~NonMaxSuppressionV3CpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetInputAndCheck(const CpuKernelContext &ctx);
  template <typename T>
  inline float IOUSimilarity(const T *box_1, const T *box_2, const float offset,
                             const CorrectBox &correct_box) const;
  template <typename T, typename T_threshold>
  uint32_t DoCompute();

  const Tensor *boxes_ = nullptr;
  Tensor *scores_ = nullptr;
  Tensor *iou_threshold_tensor_ = nullptr;
  Tensor *score_threshold_tensor_ = nullptr;
  Tensor *output_indices_ = nullptr;
  int64_t num_boxes_ = 0;
  int32_t max_output_size_ = 0;
  int32_t offset_ = 0;
  DataType threshold_dtype_ = DT_UINT32;
  DataType boxes_scores_dtype_ = DT_UINT32;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_NON_MAX_SUPPRESSION_V3_KERNELS_H_
