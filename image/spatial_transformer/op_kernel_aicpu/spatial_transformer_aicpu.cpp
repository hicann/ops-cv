/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "spatial_transformer_aicpu.h"

#include <algorithm>
#include <chrono>
#include <securec.h>

#include "cpu_kernel.h"

namespace {
const char *const kSpatialTransformer = "SpatialTransformer";
constexpr uint32_t kTotalThetaNumber = 6;
constexpr int32_t kDimSizeIndex0 = 0;
constexpr int32_t kDimSizeIndex1 = 1;
constexpr int32_t kDimSizeIndex2 = 2;
constexpr int32_t kDimSizeIndex3 = 3;
constexpr int32_t kDimSizeIndex4 = 4;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr size_t kIndex5 = 5;
constexpr size_t kIndex6 = 6;
constexpr float kNumber2 = 2;
constexpr uint32_t kGridInedxOffset = 2;
constexpr uint32_t kOutputNum = 1;

#define STN_COMPUTE_CASE(DTYPE, TYPE, CTX)                             \
  case (DTYPE): {                                                      \
    if (DoCompute<TYPE>(CTX) != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("SpatialTransformer kernel compute failed.");   \
      return static_cast<uint32_t>(KERNEL_STATUS_INNER_ERROR);         \
    }                                                                  \
    break;                                                             \
  }

#define STN_INNER_COMPUTE_CASE(DTYPE, TYPE, CTX)                       \
  case (DTYPE): {                                                      \
    uint32_t ret = static_cast<uint32_t>(KERNEL_STATUS_OK);            \
    if (date_format_ == FORMAT_NCHW) {                                 \
      ret = static_cast<uint32_t>(DoCompute4D<T, TYPE>());             \
    } else if (date_format_ == FORMAT_NC1HWC0) {                       \
      if (stn_ori_channel_ == 1) {                                     \
        ret = static_cast<uint32_t>(DoCompute5D_C1<T, TYPE>());        \
      } else {                                                         \
        ret = static_cast<uint32_t>(DoCompute5D<T, TYPE>());           \
      }                                                                \
    }                                                                  \
    if (ret != static_cast<uint32_t>(KERNEL_STATUS_OK)) {              \
      KERNEL_LOG_ERROR("SpatialTransformer kernel compute failed.");   \
      return ret;                                                      \
    }                                                                  \
    break;                                                             \
  }
}

namespace aicpu {
KernelStatus SpatialTransformerCpuKernel::GetInputAndCheckValid(const CpuKernelContext &ctx) {
  input_tensor_ = ctx.Input(0);
  input_theta_ = ctx.Input(1);
  output_tensor_ = ctx.Output(0);
  if (input_tensor_ == nullptr || input_theta_ == nullptr || output_tensor_ == nullptr) {
    KERNEL_LOG_ERROR("Input or output invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // only support NCHW and NHWC
  date_format_ = input_tensor_->GetTensorShape()->GetFormat();
  if (date_format_ == FORMAT_NCHW) {
    input_n_ = static_cast<int32_t>(input_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex0));
    input_c_ = static_cast<int32_t>(input_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex1));
    input_h_ = static_cast<int32_t>(input_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex2));
    input_w_ = static_cast<int32_t>(input_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex3));
    output_h_ = static_cast<int32_t>(output_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex2));
    output_w_ = static_cast<int32_t>(output_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex3));
  } else if (date_format_ == FORMAT_NC1HWC0) {
    input_n_ = static_cast<int32_t>(input_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex0));
    input_c1_ = static_cast<int32_t>(input_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex1));
    input_h_ = static_cast<int32_t>(input_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex2));
    input_w_ = static_cast<int32_t>(input_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex3));
    input_c0_ = static_cast<int32_t>(input_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex4));
    input_c_ = input_c1_ * input_c0_;
    output_h_ = static_cast<int32_t>(output_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex2));
    output_w_ = static_cast<int32_t>(output_tensor_->GetTensorShape()->GetDimSize(kDimSizeIndex3));
  }
  else {
    KERNEL_LOG_ERROR("Can't support data format[%d].", static_cast<int>(date_format_));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  bool dims_error_flag = (input_n_ == 0 || input_c_ == 0 || input_h_ == 0 ||
                          input_w_ == 0 || output_h_ == 0 || output_w_ == 0);
  if (dims_error_flag) {
    KERNEL_LOG_ERROR("Dims error.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get and check data type
  input_data_type_ = static_cast<DataType>(input_tensor_->GetDataType());
  input_theta_type_ = static_cast<DataType>(input_theta_->GetDataType());
  output_data_type_ = static_cast<DataType>(output_tensor_->GetDataType());
  if (input_data_type_ != output_data_type_) {
    KERNEL_LOG_ERROR("Input data type[%s] and output data type[%s] are not same.",
        DTypeStr(input_data_type_).c_str(), DTypeStr(output_data_type_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return GetAttrs(ctx);
}

KernelStatus SpatialTransformerCpuKernel::GetAttrs(const CpuKernelContext &ctx) {
  // get theta list
  AttrValue *theta_ptr = ctx.GetAttr("default_theta");
  KERNEL_CHECK_NULLPTR(theta_ptr, KERNEL_STATUS_PARAM_INVALID,
      "[%s] get attr default_theta fail.", kSpatialTransformer);
  theta_ = theta_ptr->GetListFloat();

  // get theta valid list
  AttrValue *theta_valid_ptr = ctx.GetAttr("use_default_theta");
  KERNEL_CHECK_NULLPTR(theta_valid_ptr, KERNEL_STATUS_PARAM_INVALID,
      "[%s] get attr use_default_theta fail.", kSpatialTransformer);
  theta_valid_ = theta_valid_ptr->GetListInt();

  // get stn original channel
  AttrValue *ori_channel_ptr = ctx.GetAttr("stn_ori_channel");
  KERNEL_CHECK_NULLPTR(ori_channel_ptr, KERNEL_STATUS_PARAM_INVALID,
      "[%s] get attr stn_ori_channel fail.", kSpatialTransformer);
  stn_ori_channel_ = static_cast<int32_t>(ori_channel_ptr->GetInt());

  return KERNEL_STATUS_OK;
}

template <typename T1>
void SpatialTransformerCpuKernel::InitTheta(const T1* input_theta, uint32_t& input_theta_idx,
    std::vector<float>& theta) {
  uint32_t predf_theta_idx = 0;
  for (uint32_t j = 0; j < kTotalThetaNumber; j++) {
    if (theta_valid_[j] == 1) {
      theta[j] = theta_[predf_theta_idx];
      predf_theta_idx++;
    }
    else {
      theta[j] = static_cast<float>(input_theta[input_theta_idx]);
      input_theta_idx++;
    }
  }
}

void SpatialTransformerCpuKernel::ComputeGrid(const std::vector<float>& theta, float* input_grid) {
  for (int32_t s = 0; s < output_h_; ++s) {
    for (int32_t t = 0; t < output_w_; ++t) {
      uint32_t input_grid_idx = static_cast<uint32_t>((s * output_w_ + t) * 2);
      float ref_output_grid_1 = (static_cast<float>(s) / output_h_) * kNumber2 - 1;
      float ref_output_grid_2 = (static_cast<float>(t) / output_w_) * kNumber2 - 1;
      input_grid[input_grid_idx] = ((ref_output_grid_1 * theta[0] +
          ref_output_grid_2 * theta[1] + theta[kIndex2] + 1) / kNumber2) * input_h_;
      input_grid[input_grid_idx + 1] = ((ref_output_grid_1 * theta[kIndex3] +
          ref_output_grid_2 * theta[kIndex4] + theta[kIndex5] + 1) / kNumber2) * input_w_;
    }
  }
}

template <typename T>
float SpatialTransformerCpuKernel::BilinearInterpolateScalar(const T* data, int32_t base_idx,
    float x, float y, int32_t row_stride, int32_t col_stride) {
  float x_floor = floor(x);
  float y_floor = floor(y);
  float x_ref_1 = x - x_floor;
  float y_ref_1 = y - y_floor;
  float x_ref_0 = 1.0f - x_ref_1;
  float y_ref_0 = 1.0f - y_ref_1;
  float res = 0.0f;

  int32_t m = static_cast<int32_t>(x_floor);
  int32_t n = static_cast<int32_t>(y_floor);
  if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
    res += x_ref_0 * y_ref_0 * static_cast<float>(data[base_idx + m * row_stride + n * col_stride]);
  }

  n = static_cast<int32_t>(y_floor + 1);
  if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
    res += x_ref_0 * y_ref_1 * static_cast<float>(data[base_idx + m * row_stride + n * col_stride]);
  }

  m = static_cast<int32_t>(x_floor + 1);
  n = static_cast<int32_t>(y_floor);
  if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
    res += x_ref_1 * y_ref_0 * static_cast<float>(data[base_idx + m * row_stride + n * col_stride]);
  }

  n = static_cast<int32_t>(y_floor + 1);
  if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
    res += x_ref_1 * y_ref_1 * static_cast<float>(data[base_idx + m * row_stride + n * col_stride]);
  }

  return res;
}

template <typename T>
void SpatialTransformerCpuKernel::BilinearInterpolFillData(const T* data, int32_t base_idx,
    float x, float y, float* res) {
  float x_floor = floor(x);
  float y_floor = floor(y);
  float x_ref_1 = x - x_floor;
  float y_ref_1 = y - y_floor;
  float x_ref_0 = 1.0f - x_ref_1;
  float y_ref_0 = 1.0f - y_ref_1;

  int32_t m = static_cast<int32_t>(x_floor);
  int32_t n = static_cast<int32_t>(y_floor);
  if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
    int32_t spos = base_idx + m * input_w_ * input_c0_ + n * input_c0_;
    for (int32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
      res[c0_i] += x_ref_0 * y_ref_0 * static_cast<float>(data[spos + c0_i]);
    }
  }
}

template <typename T>
void SpatialTransformerCpuKernel::BilinearInterpolateVector(const T* data, int32_t base_idx,
    float x, float y, float* res) {
  float x_floor = floor(x);
  float y_floor = floor(y);
  float x_ref_1 = x - x_floor;
  float y_ref_1 = y - y_floor;
  float x_ref_0 = 1.0f - x_ref_1;
  float y_ref_0 = 1.0f - y_ref_1;

  BilinearInterpolFillData(data, base_idx, x, y, res);

  int32_t m = static_cast<int32_t>(x_floor);
  int32_t n = static_cast<int32_t>(y_floor + 1);
  if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
    int32_t spos = base_idx + m * input_w_ * input_c0_ + n * input_c0_;
    for (int32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
      res[c0_i] += x_ref_0 * y_ref_1 * static_cast<float>(data[spos + c0_i]);
    }
  }

  m = static_cast<int32_t>(x_floor + 1);
  n = static_cast<int32_t>(y_floor);
  if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
    int32_t spos = base_idx + m * input_w_ * input_c0_ + n * input_c0_;
    for (int32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
      res[c0_i] += x_ref_1 * y_ref_0 * static_cast<float>(data[spos + c0_i]);
    }
  }

  n = static_cast<int32_t>(y_floor + 1);
  if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
    int32_t spos = base_idx + m * input_w_ * input_c0_ + n * input_c0_;
    for (int32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
      res[c0_i] += x_ref_1 * y_ref_1 * static_cast<float>(data[spos + c0_i]);
    }
  }
}

template <typename T, typename T1>
KernelStatus SpatialTransformerCpuKernel::DoCompute4D() {
  KERNEL_LOG_INFO("Enter SpatialTransformerCpuKernel::DoCompute4D.");
  const T* input_data_ptr = reinterpret_cast<T *>(input_tensor_->GetData());
  const T1* input_theta = reinterpret_cast<T1 *>(input_theta_->GetData());
  T* output_data_ptr = reinterpret_cast<T *>(output_tensor_->GetData());

  // init ouput_grid and input_grid, [M, 3] * [3, 2] = [M, 2]
  float* input_grid = (float *)malloc(sizeof(float) * output_h_ * output_w_ * 2);
  KERNEL_CHECK_NULLPTR(input_grid, KERNEL_STATUS_INNER_ERROR, "Can't malloc input_grid.");

  // init var
  std::vector<float> theta(kIndex6);
  uint32_t input_theta_idx = 0;
  int32_t output_idx = 0;
  int32_t input_idx = 0;

  for (int32_t i = 0; i < input_n_; i++) {
    InitTheta(input_theta, input_theta_idx, theta);
    ComputeGrid(theta, input_grid);

    // calc output data
    for (int32_t j = 0; j < input_c_; j++) {
      uint32_t input_grid_idx = 0;
      for (int32_t k = 0; k < output_h_ * output_w_; k++) {
        float x = input_grid[input_grid_idx];
        float y = input_grid[input_grid_idx + 1];
        output_data_ptr[output_idx] = static_cast<T>(
            BilinearInterpolateScalar(input_data_ptr, input_idx, x, y, input_w_, 1));
        input_grid_idx += kGridInedxOffset;
        output_idx++;
      }
      input_idx += input_h_ * input_w_;
    }
  }

  free(input_grid);
  input_grid = nullptr;

  return KERNEL_STATUS_OK;
}

template <typename T, typename T1>
KernelStatus SpatialTransformerCpuKernel::DoCompute5D() {
  KERNEL_LOG_INFO("Enter SpatialTransformerCpuKernel::DoCompute5D");
  const T* input_data = reinterpret_cast<T *>(input_tensor_->GetData());
  const T1* input_theta = reinterpret_cast<T1*>(input_theta_->GetData());
  T* output_data_ptr = reinterpret_cast<T*>(output_tensor_->GetData());

  // init ouput_grid and input_grid, [M, 3] * [3, 2] = [M, 2]
  float* input_grid = (float *)malloc(sizeof(float) * output_w_ * output_h_ * 2);
  KERNEL_CHECK_NULLPTR(input_grid, KERNEL_STATUS_INNER_ERROR, "Can't malloc input_grid");

  float *res = (float *)malloc(sizeof(float) * input_c0_);
  if (res == nullptr) {
    KERNEL_LOG_ERROR("Can't malloc res.");
    free(input_grid);
    return KERNEL_STATUS_INNER_ERROR;
  }
  // init var
  std::vector<float> theta(kIndex6);
  uint32_t input_theta_idx = 0;
  uint32_t input_idx = 0;
  uint32_t output_index = 0;

  for (int32_t i = 0; i < input_n_; i++) {
    InitTheta(input_theta, input_theta_idx, theta);
    ComputeGrid(theta, input_grid);

    // calc output data
    for (int32_t j = 0; j < input_c1_; j++) {
      uint32_t input_grid_idx = 0;
      for (int32_t k = 0; k < output_h_ * output_w_; k++) {
        float x = input_grid[input_grid_idx];
        float y = input_grid[input_grid_idx + 1];

        (void)memset_s(res, sizeof(float) * input_c0_, 0.0f, sizeof(float) * input_c0_);
        BilinearInterpolateVector(input_data, static_cast<int32_t>(input_idx), x, y, res);

        for (int32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
          output_data_ptr[output_index + c0_i] = (T)res[c0_i];
        }
        input_grid_idx += kGridInedxOffset;
        output_index += input_c0_;
      }
      input_idx += input_h_ * input_w_ * input_c0_;
    }
  }

  free(input_grid);
  input_grid = nullptr;
  free(res);
  res = nullptr;

  return KERNEL_STATUS_OK;
}

template <typename T, typename T1>
KernelStatus SpatialTransformerCpuKernel::DoCompute5D_C1() {
  KERNEL_LOG_INFO("Enter SpatialTransformerCpuKernel::DoCompute5D_C1");
  const T* input_data_ptr = reinterpret_cast<T *>(input_tensor_->GetData());
  const T1* input_theta = reinterpret_cast<T1 *>(input_theta_->GetData());
  T* output_data_ptr = reinterpret_cast<T *>(output_tensor_->GetData());

  // init ouput_grid and input_grid, [M, 3] * [3, 2] = [M, 2]
  float* input_grid = (float *)malloc(sizeof(float) * output_h_ * output_w_ * 2);
  KERNEL_CHECK_NULLPTR(input_grid, KERNEL_STATUS_INNER_ERROR, "Can't malloc input_grid");

  // init var
  std::vector<float> theta(kIndex6);
  uint32_t input_theta_idx = 0;
  int32_t output_idx = 0;
  int32_t input_idx = 0;

  for (int32_t i = 0; i < input_n_; i++) {
    InitTheta(input_theta, input_theta_idx, theta);
    ComputeGrid(theta, input_grid);

    // calc output data
    for (int32_t j = 0; j < input_c1_; j++) {
      uint32_t input_grid_idx = 0;
      for (int32_t k = 0; k < output_h_ * output_w_; k++) {
        float x = input_grid[input_grid_idx];
        float y = input_grid[input_grid_idx + 1];
        output_data_ptr[output_idx] = static_cast<T>(
            BilinearInterpolateScalar(input_data_ptr, input_idx, x, y,
                input_w_ * input_c0_, input_c0_));
        input_grid_idx += kGridInedxOffset;
        output_idx += input_c0_;
      }
      input_idx += input_h_ * input_w_ * input_c0_;
    }
  }

  free(input_grid);
  input_grid = nullptr;

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SpatialTransformerCpuKernel::DoCompute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Enter SpatialTransformerCpuKernel::DoCompute");

  switch (input_theta_type_) {
    STN_INNER_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    STN_INNER_COMPUTE_CASE(DT_FLOAT, float, ctx)
    STN_INNER_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_DOUBLE, double, ctx)
  default:
    KERNEL_LOG_ERROR("SpatialTransformer kernel data type [%s] not support.", DTypeStr(input_data_type_).c_str());
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }

  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

uint32_t SpatialTransformerCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t realInputSize = ctx.GetInputsSize();
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, realInputSize, kOutputNum),
                      "[%s] normal check params failed.", kSpatialTransformer);
  KernelStatus res = GetInputAndCheckValid(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
      "GetInputAndCheckValid process failed");

  switch (input_data_type_) {
    STN_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    STN_COMPUTE_CASE(DT_FLOAT, float, ctx)
    STN_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    STN_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    STN_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    STN_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    STN_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    STN_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    STN_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    STN_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    STN_COMPUTE_CASE(DT_DOUBLE, double, ctx)
  default:
    KERNEL_LOG_ERROR("SpatialTransformer kernel data type [%s] not support.", DTypeStr(input_data_type_).c_str());
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }

  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

REGISTER_CPU_KERNEL(kSpatialTransformer, SpatialTransformerCpuKernel);
}  // namespace aicpu