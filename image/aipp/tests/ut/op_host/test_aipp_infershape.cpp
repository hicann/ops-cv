/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aipp_infershape.cpp
 * \brief RT2 InferShape / InferShapeRange UT for Aipp.
 */

#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"
#include "op_infer_shape_range_context_builder.h"
#include "register/op_impl_kernel_registry.h"

namespace {
constexpr size_t kParamsBlobSize = 128;
constexpr int32_t kParamsHeadSize = 64;
constexpr uint8_t kFormatYuv420sp = 1;
constexpr uint8_t kFormatXrgb8888 = 2;
constexpr uint8_t kFormatRgb888 = 5;
constexpr uint8_t kFormatYuyv = 7;
constexpr uint8_t kFormatYuv400 = 10;

const char*
    kStaticRgb = R"({"aipp_mode":"static","input_format":"RGB888_U8","src_image_size_h":224,"src_image_size_w":224})";
const char* kStaticRgbCrop =
    R"({"aipp_mode":"static","input_format":"RGB888_U8","src_image_size_h":256,"src_image_size_w":256,)"
    R"("crop":true,"crop_size_h":224,"crop_size_w":224})";
const char* kStaticRgbResize =
    R"({"aipp_mode":"static","input_format":"RGB888_U8","src_image_size_h":256,"src_image_size_w":256,)"
    R"("resize":true,"resize_output_h":224,"resize_output_w":224})";
const char* kStaticRgbPad =
    R"({"aipp_mode":"static","input_format":"RGB888_U8","src_image_size_h":224,"src_image_size_w":224,)"
    R"("padding":true,"left_padding_size":10,"right_padding_size":10,"top_padding_size":10,"bottom_padding_size":10})";
const char* kStaticXrgb =
    R"({"aipp_mode":"static","input_format":"XRGB8888_U8","src_image_size_h":224,"src_image_size_w":224})";
const char* kDynamicCfg = R"({"aipp_mode":"dynamic","max_src_image_size":1000000})";
const char* kDynamicNoMax = R"({"aipp_mode":"dynamic"})";

void FillAippParams(uint8_t* buf, uint8_t format, int8_t batch, int32_t srcW, int32_t srcH, int8_t crop = 0,
                    int32_t cropW = 0, int32_t cropH = 0, int8_t scf = 0, int32_t scfW = 0, int32_t scfH = 0,
                    int8_t pad = 0, int32_t padT = 0, int32_t padB = 0, int32_t padL = 0, int32_t padR = 0)
{
    (void)memset(buf, 0, kParamsBlobSize);
    buf[0] = format;
    buf[4] = static_cast<uint8_t>(batch);
    (void)memcpy(buf + 8, &srcW, sizeof(srcW));
    (void)memcpy(buf + 12, &srcH, sizeof(srcH));
    buf[kParamsHeadSize] = static_cast<uint8_t>(crop);
    buf[kParamsHeadSize + 1] = static_cast<uint8_t>(scf);
    buf[kParamsHeadSize + 2] = static_cast<uint8_t>(pad);
    (void)memcpy(buf + kParamsHeadSize + 16, &cropW, sizeof(cropW));
    (void)memcpy(buf + kParamsHeadSize + 20, &cropH, sizeof(cropH));
    (void)memcpy(buf + kParamsHeadSize + 32, &scfW, sizeof(scfW));
    (void)memcpy(buf + kParamsHeadSize + 36, &scfH, sizeof(scfH));
    (void)memcpy(buf + kParamsHeadSize + 40, &padT, sizeof(padT));
    (void)memcpy(buf + kParamsHeadSize + 44, &padB, sizeof(padB));
    (void)memcpy(buf + kParamsHeadSize + 48, &padL, sizeof(padL));
    (void)memcpy(buf + kParamsHeadSize + 52, &padR, sizeof(padR));
}

gert::InfershapeContextPara MakeInferPara(const gert::StorageShape& imagesShape, ge::Format imagesFormat,
                                          ge::Format featuresFormat, const std::string& cfg, void* paramsAddr = nullptr,
                                          size_t paramsBytes = kParamsBlobSize)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {{imagesShape, ge::DT_UINT8, imagesFormat}};
    if (paramsAddr != nullptr) {
        gert::StorageShape paramsShape = {{static_cast<int64_t>(paramsBytes)}, {static_cast<int64_t>(paramsBytes)}};
        inputs.push_back({paramsShape, ge::DT_UINT8, ge::FORMAT_ND, true, paramsAddr});
    }
    gert::StorageShape featuresShape = {{}, {}};
    return gert::InfershapeContextPara(
        "Aipp", inputs, {{featuresShape, ge::DT_FLOAT16, featuresFormat}},
        {gert::InfershapeContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<std::string>(cfg))});
}

auto GetAippOpImpl()
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    EXPECT_NE(spaceRegistry, nullptr);
    if (spaceRegistry == nullptr) {
        return static_cast<const gert::OpImplKernelRegistry::OpImplFunctionsV2*>(nullptr);
    }
    return spaceRegistry->GetOpImpl("Aipp");
}

std::vector<int64_t> ShapeToVec(const gert::Shape& shape)
{
    std::vector<int64_t> out(shape.GetDimNum(), 0);
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        out[i] = shape.GetDim(i);
    }
    return out;
}
} // namespace

class AippInfershape : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(AippInfershape, registers_infer_and_params_data_dependency)
{
    const auto* opImpl = GetAippOpImpl();
    ASSERT_NE(opImpl, nullptr);
    EXPECT_NE(opImpl->infer_shape, nullptr);
    EXPECT_NE(opImpl->infer_shape_range, nullptr);
    EXPECT_TRUE(opImpl->IsInputDataDependency(1));
}

TEST_F(AippInfershape, static_nhwc_images_nchw_features)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kStaticRgb);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 224, 224, 3}});
}

TEST_F(AippInfershape, static_nchw_images_nchw_features)
{
    gert::InfershapeContextPara para = MakeInferPara({{2, 3, 224, 224}, {2, 3, 224, 224}}, ge::FORMAT_NCHW,
                                                     ge::FORMAT_NCHW, kStaticRgb);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3, 224, 224}});
}

TEST_F(AippInfershape, static_nhwc_images_nhwc_features)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NHWC, kStaticRgb);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 224, 224, 3}});
}

TEST_F(AippInfershape, static_c04_copies_images_shape)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 1, 224, 224, 4}, {1, 1, 224, 224, 4}}, ge::FORMAT_NC1HWC0_C04,
                                                     ge::FORMAT_NCHW, kStaticRgb);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 1, 224, 224, 4}});
}

TEST_F(AippInfershape, static_xrgb_features_c_follows_images_not_cfg)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kStaticXrgb);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 224, 224, 3}});
}

TEST_F(AippInfershape, static_crop_match)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kStaticRgbCrop);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 224, 224, 3}});
}

TEST_F(AippInfershape, static_crop_mismatch)
{
    const char*
        cfg = R"({"aipp_mode":"static","input_format":"RGB888_U8","src_image_size_h":256,"src_image_size_w":256,)"
              R"("crop":true,"crop_size_h":200,"crop_size_w":200})";
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, cfg);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_resize_match)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kStaticRgbResize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 224, 224, 3}});
}

TEST_F(AippInfershape, static_padding_match)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 244, 244, 3}, {1, 244, 244, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kStaticRgbPad);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 244, 244, 3}});
}

TEST_F(AippInfershape, static_padding_mismatch)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kStaticRgbPad);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_unknown_hw_not_equal_known_crop)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, -1, -1, 3}, {1, -1, -1, 3}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     kStaticRgbCrop);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_unknown_hw_equal_when_cfg_also_unknown)
{
    const char* cfg = R"({"aipp_mode":"static","input_format":"RGB888_U8"})";
    gert::InfershapeContextPara para = MakeInferPara({{1, -1, -1, 3}, {1, -1, -1, 3}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     cfg);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, -1, -1, 3}});
}

TEST_F(AippInfershape, static_invalid_images_format)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_ND,
                                                     ge::FORMAT_NCHW, kStaticRgb);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_invalid_images_rank)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224}, {1, 224, 224}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     kStaticRgb);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_features_nd_copies_images_shape)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_ND, kStaticRgb);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 224, 224, 3}});
}

TEST_F(AippInfershape, static_invalid_mode)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, R"({"aipp_mode":"foo"})");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_missing_mode)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, R"({"input_format":"RGB888_U8"})");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_empty_cfg)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, "");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_cfg_path_not_json)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, "aipp_ut_test_01.cfg");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_file_path_with_dotdot_rejected)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, "../aipp.cfg");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, dynamic_compile_unknown_4d)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kDynamicCfg);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 224, 224, 3}});
}

TEST_F(AippInfershape, dynamic_compile_nhwc_features)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NHWC, kDynamicCfg);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 224, 224, 3}});
}

TEST_F(AippInfershape, dynamic_compile_nd_features_copies_images_shape)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 3, -1, -1}, {1, 3, -1, -1}}, ge::FORMAT_NCHW, ge::FORMAT_ND,
                                                     kDynamicCfg);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 3, -1, -1}});
}

TEST_F(AippInfershape, dynamic_compile_max_src_zero)
{
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kDynamicNoMax);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, dynamic_compile_unknown_rank)
{
    gert::InfershapeContextPara para = MakeInferPara({{-2}, {-2}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW, kDynamicCfg);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, dynamic_runtime_rgb_nchw)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatRgb888, 2, 64, 64);
    gert::InfershapeContextPara para = MakeInferPara({{2, 64, 64, 3}, {2, 64, 64, 3}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     kDynamicCfg, params);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3, 64, 64}});
}

TEST_F(AippInfershape, dynamic_runtime_xrgb_channel4)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatXrgb8888, 1, 32, 32);
    gert::InfershapeContextPara para = MakeInferPara({{1, 32, 32, 3}, {1, 32, 32, 3}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     kDynamicCfg, params);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 4, 32, 32}});
}

TEST_F(AippInfershape, dynamic_runtime_yuv400_channel1)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatYuv400, 1, 32, 32);
    gert::InfershapeContextPara para = MakeInferPara({{1, 32, 32, 1}, {1, 32, 32, 1}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     kDynamicCfg, params);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 1, 32, 32}});
}

TEST_F(AippInfershape, dynamic_runtime_yuv420_nhwc_features)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatYuv420sp, 2, 64, 48);
    gert::InfershapeContextPara para = MakeInferPara({{2, 48, 64, 3}, {2, 48, 64, 3}}, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                                     kDynamicCfg, params);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 48, 64, 3}});
}

TEST_F(AippInfershape, dynamic_runtime_crop_scf_pad)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatRgb888, 1, 64, 64, 1, 32, 32, 1, 16, 16, 1, 2, 2, 4, 4);
    gert::InfershapeContextPara para = MakeInferPara({{1, 20, 24, 3}, {1, 20, 24, 3}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     kDynamicCfg, params);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 3, 20, 24}});
}

TEST_F(AippInfershape, dynamic_runtime_src_lt_8)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatRgb888, 1, 4, 4);
    gert::InfershapeContextPara para = MakeInferPara({{1, 4, 4, 3}, {1, 4, 4, 3}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     kDynamicCfg, params);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, dynamic_runtime_crop_lt_8)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatRgb888, 1, 64, 64, 1, 4, 4);
    gert::InfershapeContextPara para = MakeInferPara({{1, 4, 4, 3}, {1, 4, 4, 3}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     kDynamicCfg, params);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, dynamic_runtime_invalid_format)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatYuyv, 1, 32, 32);
    gert::InfershapeContextPara para = MakeInferPara({{1, 32, 32, 3}, {1, 32, 32, 3}}, ge::FORMAT_NHWC, ge::FORMAT_NCHW,
                                                     kDynamicCfg, params);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, dynamic_runtime_invalid_features_format)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatRgb888, 1, 32, 32);
    gert::InfershapeContextPara para = MakeInferPara({{1, 32, 32, 3}, {1, 32, 32, 3}}, ge::FORMAT_NHWC, ge::FORMAT_ND,
                                                     kDynamicCfg, params);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, dynamic_runtime_params_shorter_than_120_failed)
{
    constexpr size_t kShortParamsBytes = 8;
    uint8_t params[kShortParamsBytes] = {};
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kDynamicCfg, params, kShortParamsBytes);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, static_cfg_with_params_blob_stays_static)
{
    uint8_t params[kParamsBlobSize];
    FillAippParams(params, kFormatRgb888, 8, 32, 32);
    gert::InfershapeContextPara para = MakeInferPara({{1, 224, 224, 3}, {1, 224, 224, 3}}, ge::FORMAT_NHWC,
                                                     ge::FORMAT_NCHW, kStaticRgb, params);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 224, 224, 3}});
}

TEST_F(AippInfershape, infer_shape_range_invalid_json_failed)
{
    const auto* opImpl = GetAippOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::Tensor imagesMin({{1, 224, 224, 3}, {1, 224, 224, 3}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost,
                           ge::DT_UINT8, nullptr);
    gert::Tensor imagesMax({{2, 224, 224, 3}, {2, 224, 224, 3}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost,
                           ge::DT_UINT8, nullptr);
    gert::Range<gert::Tensor> imagesRange(&imagesMin, &imagesMax);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Aipp").OpName("Aipp").IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorsRange({&imagesRange});
    builder.AppendAttr(ge::AscendString("not-json"));
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, infer_shape_range_attr_type_mismatch_failed)
{
    const auto* opImpl = GetAippOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::Tensor imagesMin({{1, 224, 224, 3}, {1, 224, 224, 3}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost,
                           ge::DT_UINT8, nullptr);
    gert::Tensor imagesMax({{2, 224, 224, 3}, {2, 224, 224, 3}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost,
                           ge::DT_UINT8, nullptr);
    gert::Range<gert::Tensor> imagesRange(&imagesMin, &imagesMax);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Aipp").OpName("Aipp").IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorsRange({&imagesRange});
    builder.AppendAttr(ge::AscendString(R"({"aipp_mode":1})"));
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_FAILED);
}

TEST_F(AippInfershape, infer_shape_range_static)
{
    const auto* opImpl = GetAippOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::Tensor imagesMin({{1, 224, 224, 3}, {1, 224, 224, 3}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost,
                           ge::DT_UINT8, nullptr);
    gert::Tensor imagesMax({{2, 224, 224, 3}, {2, 224, 224, 3}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost,
                           ge::DT_UINT8, nullptr);
    gert::Range<gert::Tensor> imagesRange(&imagesMin, &imagesMax);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Aipp").OpName("Aipp").IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorsRange({&imagesRange});
    builder.AppendAttr(ge::AscendString(kStaticRgb));
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    const auto* outRange = context->GetOutputShapeRange(0);
    ASSERT_NE(outRange, nullptr);
    ASSERT_NE(outRange->GetMin(), nullptr);
    ASSERT_NE(outRange->GetMax(), nullptr);
    EXPECT_EQ(ShapeToVec(*outRange->GetMin()), (std::vector<int64_t>{1, 224, 224, 3}));
    EXPECT_EQ(ShapeToVec(*outRange->GetMax()), (std::vector<int64_t>{2, 224, 224, 3}));
}

TEST_F(AippInfershape, infer_shape_range_dynamic_4d_n)
{
    const auto* opImpl = GetAippOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::Tensor imagesMin({{1, 224, 224, 3}, {1, 224, 224, 3}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost,
                           ge::DT_UINT8, nullptr);
    gert::Tensor imagesMax({{4, 224, 224, 3}, {4, 224, 224, 3}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost,
                           ge::DT_UINT8, nullptr);
    gert::Range<gert::Tensor> imagesRange(&imagesMin, &imagesMax);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Aipp").OpName("Aipp").IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorsRange({&imagesRange});
    builder.AppendAttr(ge::AscendString(kDynamicCfg));
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    const auto* outRange = context->GetOutputShapeRange(0);
    ASSERT_NE(outRange, nullptr);
    EXPECT_EQ(ShapeToVec(*outRange->GetMin()), (std::vector<int64_t>{1, 224, 224, 3}));
    EXPECT_EQ(ShapeToVec(*outRange->GetMax()), (std::vector<int64_t>{4, 224, 224, 3}));
}

TEST_F(AippInfershape, infer_shape_range_dynamic_not_4d)
{
    const auto* opImpl = GetAippOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::Tensor imagesMin({{1, 1000}, {1, 1000}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost, ge::DT_UINT8,
                           nullptr);
    gert::Tensor imagesMax({{1, 1000}, {1, 1000}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost, ge::DT_UINT8,
                           nullptr);
    gert::Range<gert::Tensor> imagesRange(&imagesMin, &imagesMax);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Aipp").OpName("Aipp").IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC);
    builder.InputTensorsRange({&imagesRange});
    builder.AppendAttr(ge::AscendString(kDynamicCfg));
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    const auto* outRange = context->GetOutputShapeRange(0);
    ASSERT_NE(outRange, nullptr);
    EXPECT_EQ(ShapeToVec(*outRange->GetMin()), (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(ShapeToVec(*outRange->GetMax()), (std::vector<int64_t>{1, 1000}));
}

TEST_F(AippInfershape, infer_shape_range_dynamic_max_src_zero)
{
    const auto* opImpl = GetAippOpImpl();
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::Tensor imagesMin({{1, 224, 224, 3}, {1, 224, 224, 3}}, {ge::FORMAT_NHWC, ge::FORMAT_NHWC, {}}, gert::kOnHost,
                           ge::DT_UINT8, nullptr);
    gert::Range<gert::Tensor> imagesRange(&imagesMin, &imagesMin);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("Aipp").OpName("Aipp").IONum(1, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorsRange({&imagesRange});
    builder.AppendAttr(ge::AscendString(kDynamicNoMax));
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_FAILED);
}
