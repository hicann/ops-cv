/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_nearest3d_torch.cpp
 * \brief
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"

// 直接调用image目录中已经实现的算子公共逻辑
#include "image/upsample_nearest3d/op_kernel/upsample_nearest3d.h"
#include "image/upsample_nearest3d/op_host/upsample_nearest3d_tiling_common.h"

namespace ascend_ops {

namespace UpsampleNearest3dFastKernel {

using namespace UpsampleNearest3d;

// 算子kernel实现，需要根据具体API的接口定义修改
template <typename T>
__global__ __aicore__ void upsample_nearest3d_kernel(
    __gm__ uint8_t* x, __gm__ uint8_t* y, const UpsampleNearest3dTilingData tilingData)
{
    if constexpr (std::is_same_v<T, c10::Half>) {
        UpsampleNearest3dKernelImpl<UPSAMPLE_NEAREST3D_TPL_FP16, UPSAMPLE_NEAREST3D_TPL_FP16>(x, y, false, &tilingData);
        return;
    }
    if constexpr (std::is_same_v<T, c10::BFloat16>) {
        UpsampleNearest3dKernelImpl<UPSAMPLE_NEAREST3D_TPL_BF16, UPSAMPLE_NEAREST3D_TPL_BF16>(x, y, false, &tilingData);
        return;
    }
    if constexpr (std::is_same_v<T, float>) {
        UpsampleNearest3dKernelImpl<UPSAMPLE_NEAREST3D_TPL_FP32, UPSAMPLE_NEAREST3D_TPL_FP32>(x, y, false, &tilingData);
        return;
    }
}

// 算子入口实现，在该方法中使用<<<>>>的方式调用算子kernel，需要根据具体API的接口定义修改
template <typename T>
void upsample_nearest3d_api(aclrtStream stream, const at::Tensor& x, const int64_t* output_size, const at::Tensor& y)
{
    UpsampleNearest3dTilingData tilingData;
    float scales[3] = {0.0f, 0.0f, 0.0f};
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = ascendc_platform->GetCoreNumAiv();
    UpsampleNearest3dTiling::UpsampleNearest3dCommonTiling<at::Tensor>(x, scales, output_size, tilingData, coreNum);
    uint32_t blockDim = tilingData.needCoreNum;
    auto x_ptr = x.data_ptr<T>();
    auto y_ptr = y.data_ptr<T>();
    upsample_nearest3d_kernel<T>
        <<<blockDim, nullptr, stream>>>((__gm__ uint8_t*)x_ptr, (__gm__ uint8_t*)y_ptr, tilingData);
}

template <>
void upsample_nearest3d_api<double>(
    aclrtStream stream, const at::Tensor& x, const int64_t* output_size, const at::Tensor& y)
{
    throw std::runtime_error("double is not supported on aicore!");
}

// 算子wrapper接口，用于向pytorch注册自定义接口，需要根据具体API的接口定义修改
torch::Tensor upsample_nearest3d_npu(const torch::Tensor& x, at::IntArrayRef output_size)
{
    TORCH_CHECK(torch_npu::utils::is_npu(x), "Input tensor must be on NPU device");
    TORCH_CHECK(x.scalar_type() != at::kDouble, "Double type is not supported by upsample_nearest3d_npu");
    std::vector<int64_t> sizes = {x.size(0), x.size(1), output_size[0], output_size[1], output_size[2]};
    at::IntArrayRef shape(sizes);
    at::Tensor y = at::empty(shape, x.options());
    int64_t output_sizes[] = {output_size[0], output_size[1], output_size[2]};
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() -> int {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "upsample_nearest3d_npu", [&] {
            upsample_nearest3d_api<scalar_t>(stream, x, output_sizes, y);
        });
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("UpsampleNearest3d", acl_call);
    return y;
}

torch::Tensor upsample_nearest3d_meta(const torch::Tensor& x, at::IntArrayRef output_size)
{
    TORCH_CHECK(x.defined(), "Input tensor must be defined");
    std::vector<int64_t> sizes = {x.size(0), x.size(1), output_size[0], output_size[1], output_size[2]};
    at::IntArrayRef shape(sizes);
    return torch::empty(
        shape, torch::TensorOptions().dtype(x.dtype()).device(torch::kMeta).memory_format(x.suggest_memory_format()));
}

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("upsample_nearest3d(Tensor x, int[] size) -> Tensor");
}

// PyTorch提供的宏，用于在后端注册算子，需要根据具体API的接口定义修改
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("upsample_nearest3d", upsample_nearest3d_npu);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("upsample_nearest3d", TORCH_FN(upsample_nearest3d_meta));
}

} // namespace UpsampleNearest3dFastKernel
} // namespace ascend_ops
