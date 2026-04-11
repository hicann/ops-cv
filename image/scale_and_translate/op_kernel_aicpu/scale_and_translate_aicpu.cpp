/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scale_and_translate_aicpu.h"

#include <iostream>
#include <type_traits>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "sampling_kernels.h"
#include "log.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 4;
constexpr int64_t kParallelDataNums = 1024;
const char *kScaleAndTranslate = "ScaleAndTranslate";

#define SCALEANDTRANSLATE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
    case (DTYPE): {                                                 \
        uint32_t result = ScaleAndTranslateCompute<TYPE>(CTX);      \
        if (result != KERNEL_STATUS_OK) {                           \
            KERNEL_LOG_ERROR("ScaleAndTranslate kernel compute failed."); \
            return result;                                          \
        }                                                           \
        break;                                                      \
    }

#define SWITCH_PARALLEL(SHARD, end_num, ctx)                                 \
    if ((end_num) <= kParallelDataNums) {                                    \
        for (size_t i = 0; i < size_t(end_num); i++) {                       \
            SHARD(i, i + 1);                                                 \
        }                                                                    \
    } else {                                                                 \
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, end_num, 1, SHARD), \
                            "ScaleAndTranslate #SHARD Compute failed.")      \
    }

}  // namespace

namespace aicpu {
uint32_t ScaleAndTranslateCpuKernel::Compute(CpuKernelContext &ctx)
{
    // check params
    KERNEL_HANDLE_ERROR(
        NormalCheck(ctx, kInputNum, kOutputNum),
        "ScaleAndTranslate check input and output number failed.");
    KERNEL_HANDLE_ERROR(ScaleAndTranslateCheck(ctx),
                        "ScaleAndTranslate check params failed.");
    auto data_type = ctx.Input(0)->GetDataType();
    switch (data_type) {
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_FLOAT, float, ctx)
        SCALEANDTRANSLATE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
        default:
            KERNEL_LOG_ERROR("ScaleAndTranslate kernel data type [%s] not support.",
                             DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ScaleAndTranslateCpuKernel::ScaleAndTranslateCheck(
    CpuKernelContext &ctx)
{
    auto input0_shape = ctx.Input(0)->GetTensorShape();
    auto input1_shape = ctx.Input(1)->GetTensorShape();
    auto input2_shape = ctx.Input(2)->GetTensorShape();
    auto input3_shape = ctx.Input(3)->GetTensorShape();
    // dims check
    KERNEL_CHECK_FALSE(
        (input0_shape->GetDims() == 4), KERNEL_STATUS_PARAM_INVALID,
        "The input0's dims=[%d] must be 4-dimensional", input0_shape->GetDims())
    KERNEL_CHECK_FALSE(
        (input1_shape->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID,
        "The input1's dims=[%d] must be 1-dimensional", input1_shape->GetDims())
    KERNEL_CHECK_FALSE((input1_shape->NumElements() == 2),
                       KERNEL_STATUS_PARAM_INVALID,
                       "The input1's numelements=[%d] must have two elements",
                       input1_shape->NumElements())

    DataType input1_type = ctx.Input(1)->GetDataType();
    DataType input2_type = ctx.Input(2)->GetDataType();
    DataType input3_type = ctx.Input(3)->GetDataType();

    // dtypes check
    KERNEL_CHECK_FALSE((input1_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                       "The input1's dtype=[%d] must be DT_INT32",
                       DTypeStr(input1_type).c_str())
    KERNEL_CHECK_FALSE((input2_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                       "The input2's dtype=[%d] must be DT_FLOAT",
                       DTypeStr(input2_type).c_str())
    KERNEL_CHECK_FALSE((input3_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                       "The input3's dtype=[%d] must be DT_FLOAT",
                       DTypeStr(input3_type).c_str())

    KERNEL_LOG_INFO(
        "ScaleAndTranslateCpuKernel[%s], input0: size[%llu], input1: size[%llu];"
        "input2: size[%llu], input3: size[%llu], output: size[%llu].",
        ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(),
        ctx.Input(1)->GetDataSize(), ctx.Input(2)->GetDataSize(),
        ctx.Input(3)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_OK;
}

template <typename T>
inline const T &Clamp(const T &lower, const T &higher, const T &value)
{
    if (higher < value) {
        return higher;
    }
    if (value < lower) {
        return lower;
    }
    return value;
}

static void NormalizeSpanWeights(const std::vector<float> &temp_weights,
                                 float total_weight, int span_size, int x,
                                 Eigen::TensorMap<Eigen::Tensor<float, 1>> &weights_vec)
{
    if (std::abs(total_weight) >=
        1000.0f * std::numeric_limits<float>::min()) {
        float one_over_total_weight = 1.0f / total_weight;
        int out_index = span_size * x;
        for (float weight : temp_weights) {
            weights_vec(out_index) = weight * one_over_total_weight;
            ++out_index;
        }
    }
}

template <typename Kernel>
uint32_t InitSpans(const Kernel &kernel, int64_t output_size,
                   int64_t input_size, bool antialias, float inv_scale,
                   Spans *spans, float &kernel_scale)
{
    kernel_scale = antialias ? std::max(inv_scale, 1.0f) : 1.0f;
    spans->span_size = std::min(
        2 * static_cast<int>(std::ceil(kernel.Radius() * kernel_scale)) + 1,
        static_cast<int>(input_size));

    spans->starts = new (std::nothrow) Eigen::Tensor<int32_t, 1>(output_size);
    KERNEL_CHECK_NULLPTR(spans->starts, KERNEL_STATUS_PARAM_INVALID,
                         "New spans starts failed.")
    spans->weights = new (std::nothrow) Eigen::Tensor<float, 1>(spans->span_size * output_size);
    KERNEL_CHECK_NULLPTR(spans->weights, KERNEL_STATUS_PARAM_INVALID,
                         "New spans weights failed.")
    return KERNEL_STATUS_OK;
}

template <typename Kernel>
uint32_t ComputeSpansCore(CpuKernelContext &context, const Kernel &kernel, const int64_t output_size,
                          const int64_t input_size, const float scale, const float translate,
                          const bool antialias, Spans *spans)
{
    const float inv_scale = 1.0 / scale;
    const float inv_translate = -inv_scale * translate;
    float kernel_scale = 0.0f;
    KERNEL_HANDLE_ERROR(InitSpans(kernel, output_size, input_size, antialias,
                                  inv_scale, spans, kernel_scale), "InitSpans failed.");
    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> starts_vec( spans->starts->data(), spans->starts->dimensions());
    Eigen::TensorMap<Eigen::Tensor<float, 1>> weights_vec(spans->weights->data(), spans->weights->dimensions());
    weights_vec.setZero();
    const float one_over_kernel_scale = 1.0f / kernel_scale;
    int max_span_size = 0;
    std::vector<float> temp_weights;
    uint32_t shard_ret = KERNEL_STATUS_OK;
    auto shard_x = [&](int start, int end) {
        for (auto x = start; x < end; ++x) {
            const float col_f = x + 0.5f;
            const float sample_f = col_f * inv_scale + inv_translate;
            // Don't sample when the sampling location is outside the source image.
            if (sample_f < 0 || sample_f > input_size) {
                // Add an empty span.
                starts_vec(x) = 0;
                continue;
            }
            int64_t span_start = std::ceil(sample_f - kernel.Radius() * kernel_scale - 0.5f);
            int64_t span_end = std::floor(sample_f + kernel.Radius() * kernel_scale - 0.5f);
            span_start = Clamp(static_cast<int64_t>(0), input_size - 1, span_start);
            span_end = Clamp(static_cast<int64_t>(0), input_size - 1, span_end) + 1;
            const int this_span_size = span_end - span_start;
            if (this_span_size > spans->span_size) {
                KERNEL_LOG_ERROR("Span is too large: [%d] vs [%d].", this_span_size, spans->span_size);
                shard_ret = KERNEL_STATUS_PARAM_INVALID;
                return;
            }
            float total_weight = 0.0f;
            temp_weights.clear();
            for (int source = span_start; source < span_end; ++source) {
                float kernel_pos = static_cast<float>(source) + 0.5f - sample_f;
                float weight = kernel(std::abs(kernel_pos * one_over_kernel_scale));
                total_weight += weight;
                temp_weights.push_back(weight);
            }
            max_span_size = std::max(max_span_size, this_span_size);
            NormalizeSpanWeights(temp_weights, total_weight, spans->span_size, x, weights_vec);
            starts_vec(x) = span_start;
        }
    };
    SWITCH_PARALLEL(shard_x, output_size, context);
    if (shard_ret != KERNEL_STATUS_OK) {
        return shard_ret;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ComputeSpans(CpuKernelContext &context,
                      const SamplingKernelType kernel_type,
                      const int64_t output_size, const int64_t input_size,
                      const float scale, const float translate,
                      const bool antialias, Spans *spans)
{
    switch (kernel_type) {
        case LANCZOS1_KERNEL: {
            return ComputeSpansCore(context, CreateLanczos1Kernel(), output_size,
                                    input_size, scale, translate, antialias, spans);
        }
        case LANCZOS3_KERNEL: {
            return ComputeSpansCore(context, CreateLanczos3Kernel(), output_size,
                                    input_size, scale, translate, antialias, spans);
        }
        case LANCZOS5_KERNEL: {
            return ComputeSpansCore(context, CreateLanczos5Kernel(), output_size,
                                    input_size, scale, translate, antialias, spans);
        }
        case GAUSSIAN_KERNEL: {
            return ComputeSpansCore(context, CreateGaussianKernel(), output_size,
                                    input_size, scale, translate, antialias, spans);
        }
        case BOX_KERNEL: {
            return ComputeSpansCore(context, CreateBoxKernel(), output_size,
                                    input_size, scale, translate, antialias, spans);
        }
        case TRIANGLE_KERNEL: {
            return ComputeSpansCore(context, CreateTriangleKernel(), output_size,
                                    input_size, scale, translate, antialias, spans);
        }
        case KEYS_CUBIC_KERNEL: {
            return ComputeSpansCore(context, CreateKeysCubicKernel(), output_size,
                                    input_size, scale, translate, antialias, spans);
        }
        case MITCHELL_CUBIC_KERNEL: {
            return ComputeSpansCore(context, CreateMitchellCubicKernel(), output_size,
                                    input_size, scale, translate, antialias, spans);
        }
        default:
            KERNEL_LOG_ERROR("kernel_type kernel data type [%u] not support.",
                             kernel_type);
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

struct ScaleAndTranslateParams {
    SamplingKernelType kernel_type;
    bool antialias;
    int64_t batch_size;
    int64_t input_height;
    int64_t input_width;
    int64_t channels;
    int64_t output_height;
    int64_t output_width;
    float row_scale;
    float col_scale;
    float row_translation;
    float col_translation;
};

static uint32_t ParseScaleAndTranslateParams(CpuKernelContext &ctx,
                                             ScaleAndTranslateParams &p)
{
    auto input_size = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
    auto input_scale = reinterpret_cast<float *>(ctx.Input(2)->GetData());
    auto input_translation = reinterpret_cast<float *>(ctx.Input(3)->GetData());
    KERNEL_CHECK_NULLPTR(ctx.GetAttr("kernel_type"), KERNEL_STATUS_PARAM_INVALID, "Get attr [kernel_type] failed.");
    std::string kernel_type_str = ctx.GetAttr("kernel_type")->GetString();
    KERNEL_CHECK_NULLPTR(ctx.GetAttr("antialias"), KERNEL_STATUS_PARAM_INVALID, "Get attr [antialias] failed.");
    p.antialias = ctx.GetAttr("antialias")->GetBool();
    p.kernel_type = SamplingKernelTypeFromString(kernel_type_str);

    auto input0_shape = ctx.Input(0)->GetTensorShape();

    p.output_height = input_size[0];
    p.output_width = input_size[1];

    p.batch_size = input0_shape->GetDimSize(0);
    p.input_height = input0_shape->GetDimSize(1);
    p.input_width = input0_shape->GetDimSize(2);
    p.channels = input0_shape->GetDimSize(3);

    KERNEL_CHECK_FALSE(
        (p.output_height > 0 && p.output_width > 0), KERNEL_STATUS_PARAM_INVALID,
        "output_height = [%d] and output_width = [%d] must be positive",
        p.output_height, p.output_width)
    KERNEL_CHECK_FALSE((p.channels > 0), KERNEL_STATUS_PARAM_INVALID,
                       "image_channel = [%d] must have at least one", p.channels)
    KERNEL_CHECK_FALSE(
        (p.input_height > 0 && p.input_width > 0), KERNEL_STATUS_PARAM_INVALID,
        "input_height = [%d] and input_width = [%d] must be of non-zero size",
        p.input_height, p.input_width)

    p.row_scale = input_scale[0];
    p.col_scale = input_scale[1];

    KERNEL_CHECK_FALSE(
        (p.row_scale > 0 && p.col_scale > 0), KERNEL_STATUS_PARAM_INVALID,
        "row_scale = [%d] and col_scale = [%d] must be greater than zero.",
        p.row_scale, p.col_scale)

    p.row_translation = input_translation[0];
    p.col_translation = input_translation[1];
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ScaleAndTranslateCpuKernel::ScaleAndTranslateCompute(
    CpuKernelContext &ctx)
{
    ScaleAndTranslateParams p;
    KERNEL_HANDLE_ERROR(ParseScaleAndTranslateParams(ctx, p),
                        "ScaleAndTranslate parse params failed.");

    Tensor *input = ctx.Input(0);
    Tensor *output = ctx.Output(0);

    EigenTensor inputTensor(input, input->GetData());
    EigenTensor outputTensor(output, output->GetData());

    typename TTypes<T, 4>::Tensor image_data(inputTensor.tensor<T, 4>());

    typename TTypes<float, 4>::Tensor output_data(
        outputTensor.tensor<float, 4>());

    Spans col_spans;
    ComputeSpans(ctx, p.kernel_type, p.output_width, p.input_width, p.col_scale,
                 p.col_translation, p.antialias, &col_spans);

    Spans row_spans;
    ComputeSpans(ctx, p.kernel_type, p.output_height, p.input_height, p.row_scale,
                 p.row_translation, p.antialias, &row_spans);

    Eigen::Tensor<float, 4> intermediate_tensor_middle(p.batch_size, p.output_height,
                                                       p.input_width, p.channels);
    Eigen::TensorMap<Eigen::Tensor<float, 4>> intermediate_data(
        intermediate_tensor_middle.data(),
        intermediate_tensor_middle.dimensions());
    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> row_starts(
        row_spans.starts->data(), row_spans.starts->dimensions());
    Eigen::TensorMap<Eigen::Tensor<float, 1>> row_weights(
        row_spans.weights->data(), row_spans.weights->dimensions());
    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> col_starts(
        col_spans.starts->data(), col_spans.starts->dimensions());
    Eigen::TensorMap<Eigen::Tensor<float, 1>> col_weights(
        col_spans.weights->data(), col_spans.weights->dimensions());

    GatherSpans<T>()(ctx, row_spans.span_size, row_starts, row_weights,
                     col_spans.span_size, col_starts, col_weights, image_data,
                     intermediate_data, output_data);

    delete col_spans.starts;
    delete col_spans.weights;
    delete row_spans.starts;
    delete row_spans.weights;

    return KERNEL_STATUS_OK;
}

template <typename T>
inline void GatherColumnPixel(const T *input_row_start, const int32_t *starts,
                              const float *weights, int x, int span_size,
                              int64_t input_width, int channels, float *out_pixel)
{
    const T *in_pixel = input_row_start + starts[x] * channels;
    const float *weights_start = weights + x * span_size;
    const int real_span_size =
        std::min(starts[x] + span_size, static_cast<int>(input_width)) - starts[x];
    const float *weights_end = weights_start + real_span_size;
    for (int c = 0; c < channels; ++c) {
        out_pixel[c] = 0.0f;
    }
    for (const float *weight_ptr = weights_start; weight_ptr != weights_end; ++weight_ptr) {
        float weight = *weight_ptr;
        for (int c = 0; c < channels; ++c) {
            out_pixel[c] += weight * static_cast<float>(in_pixel[c]);
        }
        in_pixel += channels;
    }
}

template <typename T>
uint32_t GatherColumns(CpuKernelContext &context, int span_size,
                       const int32_t *starts, const float *weights,
                       const T *image, const int64_t input_height,
                       const int64_t input_width, const int64_t output_height,
                       const int64_t output_width, const int channels,
                       float *output)
{
    const int64_t in_row_size = input_width * channels;
    const int64_t out_row_size = output_width * channels;
    auto shard_column = [&](int start, int end) {
        for (int y = start; y < end; ++y) {
            const T *input_row_start = image + in_row_size * y;
            float *out_pixel = output + out_row_size * y;
            for (int x = 0; x < output_width; ++x, out_pixel += channels) {
                GatherColumnPixel(input_row_start, starts, weights, x, span_size,
                                  input_width, channels, out_pixel);
            }
        }
    };
    SWITCH_PARALLEL(shard_column, output_height, context);
    return KERNEL_STATUS_OK;
}

template <typename T>
inline void AddScaledVector(const T *in_vec, int vec_length, float weight,
                            float *out_vec)
{
    float *out_vec_end = out_vec + vec_length;
    for (; out_vec != out_vec_end; ++out_vec, ++in_vec) {
        *out_vec += weight * static_cast<float>(*in_vec);
    }
}

template <typename T>
uint32_t GatherRows(CpuKernelContext &context, int span_size,
                    const int32_t *starts, const float *weights, const T *image,
                    const int64_t input_height, const int64_t input_width,
                    const int64_t output_height, const int64_t output_width,
                    const int channels, float *output)
{
    const int64_t in_row_size = input_width * channels;
    const int64_t out_row_size = output_width * channels;
    auto shard_rows = [&](int start, int end) {
        for (int y = start; y < end; ++y) {
            float *output_row_data = output + out_row_size * y;
            std::fill(output_row_data, output_row_data + out_row_size, 0.0f);
            int in_row = starts[y];
            const T *input_row_data = image + in_row_size * in_row;
            const float *weights_start = weights + y * span_size;
            const int real_span_size =
                std::min(starts[y] + span_size, static_cast<int>(input_height)) - starts[y];
            const float *const weights_end = weights_start + real_span_size;

            for (const float *weight_it = weights_start; weight_it != weights_end; ++weight_it) {
                AddScaledVector(input_row_data, in_row_size, *weight_it, output_row_data);
                input_row_data += in_row_size;
            }
        }
    };
    SWITCH_PARALLEL(shard_rows, output_height, context);
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GatherSpans<T>::operator()(
    aicpu::CpuKernelContext &context, int row_span_size,
    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> row_starts,
    Eigen::TensorMap<Eigen::Tensor<float, 1>> row_weights, int col_span_size,
    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> col_starts,
    Eigen::TensorMap<Eigen::Tensor<float, 1>> col_weights,
    typename TTypes<T, 4>::Tensor images,
    Eigen::TensorMap<Eigen::Tensor<float, 4>> intermediate_buffer,
    typename TTypes<float, 4>::Tensor resized_images)
{
    const int batch_size = images.dimension(0);
    const int64_t input_height = images.dimension(1);
    const int64_t input_width = images.dimension(2);
    const int channels = images.dimension(3);

    const int64_t output_height = resized_images.dimension(1);
    const int64_t output_width = resized_images.dimension(2);

    const int64_t input_pix_per_batch = input_width * input_height * channels;
    const int64_t intermediate_pix_per_batch =
        input_width * output_height * channels;
    const int64_t output_pix_per_batch = output_width * output_height * channels;
    float *intermediate_ptr = intermediate_buffer.data();

    const T *image_ptr = images.data();
    float *out_ptr = resized_images.data();

    auto row_start_data = row_starts.data();
    auto row_weights_data = row_weights.data();
    for (int b = 0; b < batch_size; ++b, image_ptr += input_pix_per_batch,
             intermediate_ptr += intermediate_pix_per_batch,
             out_ptr += output_pix_per_batch) {
        GatherRows(context, row_span_size, row_start_data, row_weights_data,
                   image_ptr, input_height, input_width, output_height, input_width,
                   channels, intermediate_ptr);
        GatherColumns(context, col_span_size, col_starts.data(), col_weights.data(),
                      intermediate_ptr, output_height, input_width, output_height,
                      output_width, channels, out_ptr);
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kScaleAndTranslate, ScaleAndTranslateCpuKernel);
}  // namespace aicpu
