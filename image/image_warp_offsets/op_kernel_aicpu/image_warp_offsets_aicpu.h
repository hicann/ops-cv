/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMAGE_WARP_OFFSETS_OP_KERNELS_IMAGE_WARP_OFFSET_H_
#define IMAGE_WARP_OFFSETS_OP_KERNELS_IMAGE_WARP_OFFSET_H_

#include <map>
#include <string>
#include <vector>

#include "cpu_kernel.h"

namespace aicpu {
class ImageWarpOffsetsCpuKernel : public CpuKernel {
 public:
  ImageWarpOffsetsCpuKernel() = default;

  ~ImageWarpOffsetsCpuKernel() override = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename TImage, typename TIndex>
  static uint32_t DoCompute(const CpuKernelContext &ctx);

  uint32_t CheckParams(const CpuKernelContext &ctx) const;

  uint32_t CheckParam(const CpuKernelContext &ctx, const std::string &in_or_out,
                      uint32_t index, size_t rank) const;

  uint32_t CheckShapes(const CpuKernelContext &ctx) const;

  using KernelFunction = uint32_t (*)(const CpuKernelContext &ctx);
  static const std::map<std::string, KernelFunction> kernels_;
  static const std::vector<std::string> kernels_name_;
};
}  // namespace aicpu
#endif