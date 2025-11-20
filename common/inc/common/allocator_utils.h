/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_UTILS_ALLOCATOR_UTILS_H_
#define AICPU_UTILS_ALLOCATOR_UTILS_H_
#include <memory>

#include "cpu_attr_value.h"
#include "cpu_context.h"
#include "cpu_node_def.h"
#include "cpu_tensor.h"

namespace aicpu {
class CpuKernelAllocatorUtils {
 public:
  static uint32_t ParamCheck(const std::vector<int64_t> &dims, const void *data_ptr,
                             Tensor *&output_result_tensor);
  static uint32_t UpdateOutputDataTensor(const std::vector<int64_t> &dims,
                                         DataType type, const void *data_ptr,
                                         int64_t input_data_size,
                                         Tensor *&output_result_tensor);
  static uint32_t CheckOutputDataPtr(const uint64_t data_ptr);
  static uint32_t DeleteOutputDataPtr(const uint64_t data_ptr);
  static int64_t GetInputDataSize(const std::vector<int64_t> &dims,
                                  DataType type);
};
}  // namespace aicpu
#endif  // AICPU_UTILS_ALLOCATOR_UTILS_H_
