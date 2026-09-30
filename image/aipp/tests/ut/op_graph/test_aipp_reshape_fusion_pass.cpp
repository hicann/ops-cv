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
 * \file test_aipp_reshape_fusion_pass.cpp
 * \brief UT for AippReshapeFusionPass.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/attr_value.h"
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/aipp_reshape_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace std;

#define CONV_DEBUG false

namespace {
constexpr int32_t kImagesIdx = 0;

nlohmann::json MakeStaticCfg(const string& inputFormat, int64_t srcH, int64_t srcW, bool crop = false,
                             int64_t cropH = 0, int64_t cropW = 0)
{
    nlohmann::json cfg;
    cfg["aipp_mode"] = "static";
    cfg["input_format"] = inputFormat;
    cfg["src_image_size_h"] = srcH;
    cfg["src_image_size_w"] = srcW;
    cfg["crop"] = crop;
    if (crop) {
        cfg["crop_size_h"] = cropH;
        cfg["crop_size_w"] = cropW;
    }
    return cfg;
}

void SetStrAttr(GNode& node, const char* name, const string& value)
{
    AscendString attrName(name);
    AscendString attrValue(value.c_str());
    EXPECT_EQ(node.SetAttr(attrName, attrValue), GRAPH_SUCCESS);
}

GNode FindAipp(const GraphPtr& graph)
{
    for (auto node : graph->GetDirectNode()) {
        AscendString type;
        node.GetType(type);
        if (type == "Aipp") {
            return node;
        }
    }
    return GNode();
}

GraphPtr BuildAippGraph(const string& graphName, const vector<int64_t>& imagesShape, Format imagesFormat,
                        DataType imagesDtype, const string& cfgJson, bool setCfg = true,
                        Format featuresFormat = FORMAT_RESERVED)
{
    EsGraphBuilder graphBuilder(graphName.c_str());
    auto images = graphBuilder.CreateInput(0, "images", imagesDtype, imagesFormat, imagesShape);
    Graph* cGraph = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode aipp = CompliantNodeBuilder(cGraph)
                     .OpType("Aipp")
                     .Name("aipp")
                     .IrDefInputs({{"images", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"params", CompliantNodeBuilder::kEsIrInputOptional, ""}})
                     .IrDefOutputs({{"features", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs({{"aipp_config_path", CompliantNodeBuilder::kEsAttrOptional, "String", AttrValue()},
                                  {"has_infered_verified", CompliantNodeBuilder::kEsAttrOptional, "Int", AttrValue()},
                                  {"input_dims", CompliantNodeBuilder::kEsAttrOptional, "ListInt", AttrValue()}})
                     .Build();
    if (setCfg) {
        SetStrAttr(aipp, "aipp_config_path", cfgJson);
    }
    GNode* imagesProducer = images.GetProducer();
    EXPECT_NE(imagesProducer, nullptr);
    TensorDesc imagesDesc(Shape(imagesShape), imagesFormat, imagesDtype);
    imagesDesc.SetOriginFormat(imagesFormat);
    imagesDesc.SetOriginShape(Shape(imagesShape));
    imagesProducer->UpdateOutputDesc(0, imagesDesc);
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*cGraph, *imagesProducer, images.GetProducerOutIndex(), aipp, kImagesIdx),
              GRAPH_SUCCESS);
    aipp.UpdateInputDesc(kImagesIdx, imagesDesc);
    const Format outFormat = (featuresFormat == FORMAT_RESERVED) ? imagesFormat : featuresFormat;
    TensorDesc featuresDesc(Shape(imagesShape), outFormat, DT_FLOAT16);
    featuresDesc.SetOriginFormat(outFormat);
    featuresDesc.SetOriginShape(Shape(imagesShape));
    aipp.UpdateOutputDesc(0, featuresDesc);
    EsTensorHolder yHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(aipp, 0));
    return graphBuilder.BuildAndReset({yHolder});
}

void TestTotalPass(const string& passName, GraphPtr& graph, Status expectRes)
{
    CustomPassContext passContext;
    passContext.SetPassName(passName.c_str());
    Ops::AippReshapeFusionPass pass;
    if (CONV_DEBUG) {
        graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
    }
    Status res = pass.Run(graph, passContext);
    if (CONV_DEBUG) {
        graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
    }
    EXPECT_EQ(res, expectRes);
}
} // namespace

class AippReshapeFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(AippReshapeFusionPassTest, nv12Crop256To224MatchesPackedAndInputDims)
{
    auto cfg = MakeStaticCfg("YUV420SP_U8", 256, 256, true, 224, 224);
    GraphPtr graph = BuildAippGraph("nv12_crop", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, cfg.dump());
    TestTotalPass("nv12_crop", graph, SUCCESS);

    GNode aipp = FindAipp(graph);
    TensorDesc imagesDesc;
    ASSERT_EQ(aipp.GetInputDesc(kImagesIdx, imagesDesc), GRAPH_SUCCESS);
    EXPECT_EQ(imagesDesc.GetSize(), 98304);
    EXPECT_EQ(imagesDesc.GetShape().GetDims(), (vector<int64_t>{1, 256, 256, 3}));
    EXPECT_EQ(imagesDesc.GetOriginShape().GetDims(), (vector<int64_t>{1, 256, 256, 3}));
    EXPECT_EQ(imagesDesc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(imagesDesc.GetOriginFormat(), FORMAT_NHWC);
    EXPECT_EQ(imagesDesc.GetDataType(), DT_UINT8);
    EXPECT_TRUE(aipp.HasAttr(AscendString("has_infered_verified")));
    vector<int64_t> inputDims;
    AscendString dimsName("input_dims");
    ASSERT_EQ(aipp.GetAttr(dimsName, inputDims), GRAPH_SUCCESS);
    EXPECT_EQ(inputDims, (vector<int64_t>{1, 384, 256, 1}));
    EXPECT_EQ(inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3], 98304);
    TensorDesc featuresDesc;
    ASSERT_EQ(aipp.GetOutputDesc(0, featuresDesc), GRAPH_SUCCESS);
    EXPECT_EQ(featuresDesc.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(featuresDesc.GetOriginFormat(), FORMAT_NCHW);
    EXPECT_EQ(featuresDesc.GetShape().GetDims(), (vector<int64_t>{1, 3, 224, 224}));
    EXPECT_EQ(featuresDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(featuresDesc.GetSize(), 98304);
}

TEST_F(AippReshapeFusionPassTest, staticFormatWritebackTable)
{
    struct Case {
        const char* name;
        const char* format;
        vector<int64_t> netShape;
        vector<int64_t> expectShape;
        int64_t expectSize;
        DataType expectDtype;
        vector<int64_t> expectDims;
    };
    const vector<Case> cases = {
        {"rgb888", "RGB888_U8", {1, 3, 224, 224}, {1, 256, 256, 3}, 1 * 3 * 256 * 256, DT_UINT8, {1, 256, 256, 3}},
        {"xrgb", "XRGB8888_U8", {1, 3, 224, 224}, {1, 256, 256, 4}, 1 * 4 * 256 * 256, DT_UINT8, {1, 256, 256, 4}},
        {"argb", "ARGB8888_U8", {1, 3, 224, 224}, {1, 256, 256, 4}, 1 * 4 * 256 * 256, DT_UINT8, {1, 256, 256, 4}},
        {"ayuv", "AYUV444_U8", {1, 3, 224, 224}, {1, 256, 256, 4}, 1 * 4 * 256 * 256, DT_UINT8, {1, 256, 256, 4}},
        {"yuv400", "YUV400_U8", {1, 1, 224, 224}, {1, 256, 256, 1}, 1 * 256 * 256, DT_UINT8, {1, 256, 256, 1}},
        {"yuv422", "YUV422SP_U8", {1, 3, 224, 224}, {1, 256, 256, 3}, 1 * 2 * 256 * 256, DT_UINT8, {1, 512, 256, 1}},
        {"yuyv", "YUYV_U8", {1, 3, 224, 224}, {1, 256, 256, 3}, 1 * 2 * 256 * 256, DT_UINT8, {1, 256, 256, 2}},
        {"raw16", "RAW16", {1, 1, 224, 224}, {1, 256, 256, 1}, 1 * 256 * 256 * 2, DT_UINT16, {1, 256, 256, 1}},
        {"raw10", "RAW10", {1, 1, 224, 224}, {1, 256, 256, 1}, 1 * 256 * 256 * 2, DT_UINT16, {1, 256, 256, 1}},
        {"raw12", "RAW12", {1, 1, 224, 224}, {1, 256, 256, 1}, 1 * 256 * 256 * 2, DT_UINT16, {1, 256, 256, 1}},
        {"raw24", "RAW24", {1, 1, 224, 224}, {1, 256, 256, 1}, 1 * 256 * 256 * 4, DT_UINT32, {1, 256, 256, 1}},
        {"rgb16", "RGB16", {1, 3, 224, 224}, {1, 256, 256, 3}, 1 * 256 * 256 * 3, DT_UINT16, {1, 256, 256, 3}},
        {"rgb20", "RGB20", {1, 3, 224, 224}, {1, 256, 256, 3}, 1 * 256 * 256 * 3, DT_UINT32, {1, 256, 256, 3}},
        {"rgb24", "RGB24", {1, 3, 224, 224}, {1, 256, 256, 3}, 1 * 256 * 256 * 3, DT_UINT32, {1, 256, 256, 3}},
        {"rgb8_ir", "RGB8_IR", {1, 4, 224, 224}, {1, 256, 256, 4}, 1 * 256 * 256 * 4, DT_UINT8, {1, 256, 256, 4}},
        {"rgb16_ir", "RGB16_IR", {1, 4, 224, 224}, {1, 256, 256, 4}, 1 * 256 * 256 * 4, DT_UINT16, {1, 256, 256, 4}},
        {"rgb24_ir", "RGB24_IR", {1, 4, 224, 224}, {1, 256, 256, 4}, 1 * 256 * 256 * 4, DT_UINT32, {1, 256, 256, 4}},
    };
    for (const auto& c : cases) {
        auto cfg = MakeStaticCfg(c.format, 256, 256, true, 224, 224);
        GraphPtr graph = BuildAippGraph(c.name, c.netShape, FORMAT_NCHW, DT_FLOAT, cfg.dump());
        TestTotalPass(c.name, graph, SUCCESS);
        GNode aipp = FindAipp(graph);
        TensorDesc imagesDesc;
        ASSERT_EQ(aipp.GetInputDesc(kImagesIdx, imagesDesc), GRAPH_SUCCESS) << c.name;
        EXPECT_EQ(imagesDesc.GetSize(), c.expectSize) << c.name;
        EXPECT_EQ(imagesDesc.GetShape().GetDims(), c.expectShape) << c.name;
        EXPECT_EQ(imagesDesc.GetDataType(), c.expectDtype) << c.name;
        EXPECT_EQ(imagesDesc.GetFormat(), FORMAT_NHWC) << c.name;
        vector<int64_t> inputDims;
        AscendString dimsName("input_dims");
        ASSERT_EQ(aipp.GetAttr(dimsName, inputDims), GRAPH_SUCCESS) << c.name;
        EXPECT_EQ(inputDims, c.expectDims) << c.name;
    }
}

TEST_F(AippReshapeFusionPassTest, nhwcNetworkDescStillPackedNhwc)
{
    auto cfg = MakeStaticCfg("RGB888_U8", 256, 256, true, 224, 224);
    GraphPtr graph = BuildAippGraph("nhwc_in", {1, 224, 224, 3}, FORMAT_NHWC, DT_FLOAT, cfg.dump());
    TestTotalPass("nhwc_in", graph, SUCCESS);
    GNode aipp = FindAipp(graph);
    TensorDesc imagesDesc;
    ASSERT_EQ(aipp.GetInputDesc(kImagesIdx, imagesDesc), GRAPH_SUCCESS);
    EXPECT_EQ(imagesDesc.GetShape().GetDims(), (vector<int64_t>{1, 256, 256, 3}));
    EXPECT_EQ(imagesDesc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(imagesDesc.GetSize(), 1 * 3 * 256 * 256);
}

TEST_F(AippReshapeFusionPassTest, c04KeepsFiveDimShapeAndFormat)
{
    auto cfg = MakeStaticCfg("YUV420SP_U8", 256, 256, true, 224, 224);
    GraphPtr graph = BuildAippGraph("c04", {1, 1, 224, 224, 4}, FORMAT_NC1HWC0_C04, DT_FLOAT, cfg.dump());
    TestTotalPass("c04", graph, SUCCESS);
    GNode aipp = FindAipp(graph);
    TensorDesc imagesDesc;
    ASSERT_EQ(aipp.GetInputDesc(kImagesIdx, imagesDesc), GRAPH_SUCCESS);
    EXPECT_EQ(imagesDesc.GetShape().GetDims(), (vector<int64_t>{1, 1, 256, 256, 4}));
    EXPECT_EQ(imagesDesc.GetFormat(), FORMAT_NC1HWC0_C04);
}

TEST_F(AippReshapeFusionPassTest, dynamicCompileWritesMaxSrcAndNoInputDims)
{
    nlohmann::json cfg;
    cfg["aipp_mode"] = "dynamic";
    cfg["max_src_image_size"] = 1966400;
    GraphPtr graph = BuildAippGraph("dyn", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, cfg.dump());
    TensorDesc beforeDesc;
    FindAipp(graph).GetInputDesc(kImagesIdx, beforeDesc);
    vector<pair<int64_t, int64_t>> range = {{1, 1}, {3, 3}, {224, 224}, {224, 224}};
    beforeDesc.SetShapeRange(range);
    FindAipp(graph).UpdateInputDesc(kImagesIdx, beforeDesc);

    TestTotalPass("dyn", graph, SUCCESS);
    GNode aipp = FindAipp(graph);
    TensorDesc imagesDesc;
    ASSERT_EQ(aipp.GetInputDesc(kImagesIdx, imagesDesc), GRAPH_SUCCESS);
    EXPECT_EQ(imagesDesc.GetSize(), 1966400);
    EXPECT_EQ(imagesDesc.GetShape().GetDims(), (vector<int64_t>{1, 1966400}));
    EXPECT_EQ(imagesDesc.GetOriginShape().GetDims(), (vector<int64_t>{1, 1966400}));
    EXPECT_EQ(imagesDesc.GetDataType(), DT_UINT8);
    EXPECT_EQ(imagesDesc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(imagesDesc.GetOriginFormat(), FORMAT_NHWC);
    EXPECT_TRUE(aipp.HasAttr(AscendString("has_infered_verified")));
    EXPECT_FALSE(aipp.HasAttr(AscendString("input_dims")));
    vector<pair<int64_t, int64_t>> outRange;
    ASSERT_EQ(imagesDesc.GetShapeRange(outRange), GRAPH_SUCCESS);
    EXPECT_EQ(outRange, (vector<pair<int64_t, int64_t>>{{1, 1}, {1, 1966400}}));
    TensorDesc featuresDesc;
    ASSERT_EQ(aipp.GetOutputDesc(0, featuresDesc), GRAPH_SUCCESS);
    EXPECT_EQ(featuresDesc.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(featuresDesc.GetOriginFormat(), FORMAT_NCHW);
    EXPECT_EQ(featuresDesc.GetShape().GetDims(), (vector<int64_t>{1, 3, 224, 224}));
    EXPECT_EQ(featuresDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(featuresDesc.GetSize(), beforeDesc.GetSize());
}

TEST_F(AippReshapeFusionPassTest, staticNdFeaturesTakesImagesFormatAndPackedSize)
{
    auto cfg = MakeStaticCfg("YUV420SP_U8", 256, 256, true, 224, 224);
    GraphPtr graph = BuildAippGraph("feat_nd", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, cfg.dump(), true, FORMAT_ND);
    TestTotalPass("feat_nd", graph, SUCCESS);
    GNode aipp = FindAipp(graph);
    TensorDesc featuresDesc;
    ASSERT_EQ(aipp.GetOutputDesc(0, featuresDesc), GRAPH_SUCCESS);
    EXPECT_EQ(featuresDesc.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(featuresDesc.GetOriginFormat(), FORMAT_NCHW);
    EXPECT_EQ(featuresDesc.GetShape().GetDims(), (vector<int64_t>{1, 3, 224, 224}));
    EXPECT_EQ(featuresDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(featuresDesc.GetSize(), 98304);
    TensorDesc imagesDesc;
    ASSERT_EQ(aipp.GetInputDesc(kImagesIdx, imagesDesc), GRAPH_SUCCESS);
    EXPECT_EQ(imagesDesc.GetFormat(), FORMAT_NHWC);
}

TEST_F(AippReshapeFusionPassTest, hasInferedVerifiedSkipsRewrite)
{
    auto cfg = MakeStaticCfg("YUV420SP_U8", 256, 256, true, 224, 224);
    GraphPtr graph = BuildAippGraph("skip", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, cfg.dump());
    GNode aipp = FindAipp(graph);
    AscendString verifiedName("has_infered_verified");
    int64_t verified = 1;
    ASSERT_EQ(aipp.SetAttr(verifiedName, verified), GRAPH_SUCCESS);
    TensorDesc beforeDesc;
    ASSERT_EQ(aipp.GetInputDesc(kImagesIdx, beforeDesc), GRAPH_SUCCESS);
    TestTotalPass("skip", graph, GRAPH_NOT_CHANGED);
    TensorDesc afterDesc;
    ASSERT_EQ(FindAipp(graph).GetInputDesc(kImagesIdx, afterDesc), GRAPH_SUCCESS);
    EXPECT_EQ(afterDesc.GetShape().GetDims(), beforeDesc.GetShape().GetDims());
    EXPECT_EQ(afterDesc.GetDataType(), beforeDesc.GetDataType());
    EXPECT_EQ(afterDesc.GetFormat(), beforeDesc.GetFormat());
}

TEST_F(AippReshapeFusionPassTest, reentrantSecondRunNotChanged)
{
    auto cfg = MakeStaticCfg("YUV420SP_U8", 256, 256, true, 224, 224);
    GraphPtr graph = BuildAippGraph("reenter", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, cfg.dump());
    TestTotalPass("reenter_first", graph, SUCCESS);
    TensorDesc firstDesc;
    ASSERT_EQ(FindAipp(graph).GetInputDesc(kImagesIdx, firstDesc), GRAPH_SUCCESS);
    TestTotalPass("reenter_second", graph, GRAPH_NOT_CHANGED);
    TensorDesc secondDesc;
    ASSERT_EQ(FindAipp(graph).GetInputDesc(kImagesIdx, secondDesc), GRAPH_SUCCESS);
    EXPECT_EQ(secondDesc.GetSize(), firstDesc.GetSize());
    EXPECT_EQ(secondDesc.GetShape().GetDims(), firstDesc.GetShape().GetDims());
}

TEST_F(AippReshapeFusionPassTest, rejectNoCfgInvalidJsonMissingModeMaxSrcDimAndFormat)
{
    GraphPtr noCfg = BuildAippGraph("no_cfg", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, "{}", false);
    TestTotalPass("no_cfg", noCfg, FAILED);

    GraphPtr badJson = BuildAippGraph("bad_json", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, "not-json");
    TestTotalPass("bad_json", badJson, FAILED);

    GraphPtr badType = BuildAippGraph("bad_type", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, R"({"aipp_mode":1})");
    TestTotalPass("bad_type", badType, FAILED);

    GraphPtr noMode = BuildAippGraph("no_mode", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, R"({"foo":1})");
    TestTotalPass("no_mode", noMode, FAILED);

    GraphPtr badMode = BuildAippGraph("bad_mode", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT, R"({"aipp_mode":"xx"})");
    TestTotalPass("bad_mode", badMode, FAILED);

    GraphPtr maxSrc0 = BuildAippGraph("max0", {1, 3, 224, 224}, FORMAT_NCHW, DT_FLOAT,
                                      R"({"aipp_mode":"dynamic","max_src_image_size":0})");
    TestTotalPass("max0", maxSrc0, FAILED);

    auto cfg = MakeStaticCfg("RGB888_U8", 224, 224);
    GraphPtr shortDim = BuildAippGraph("short_dim", {1, 3, 224}, FORMAT_NCHW, DT_FLOAT, cfg.dump());
    TestTotalPass("short_dim", shortDim, FAILED);

    GraphPtr badFmt = BuildAippGraph("bad_fmt", {1, 3, 224, 224}, FORMAT_ND, DT_FLOAT, cfg.dump());
    TestTotalPass("bad_fmt", badFmt, FAILED);
}

TEST_F(AippReshapeFusionPassTest, noAippNodeNotChanged)
{
    EsGraphBuilder graphBuilder("empty");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, {1, 8});
    GraphPtr graph = graphBuilder.BuildAndReset({x});
    TestTotalPass("empty", graph, GRAPH_NOT_CHANGED);
}
