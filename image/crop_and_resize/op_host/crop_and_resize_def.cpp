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
 * \file crop_and_resize_def.cpp
 * \brief Operator definition for crop_and_resize operator
 *
 * 4 dtype combinations (aligned with TBE op_select_format):
 *   #0: x=FP32, boxes=FP32, box_index=INT32, crop_size=INT32, y=FP32
 *   #1: x=FP16, boxes=FP32, box_index=INT32, crop_size=INT32, y=FP32
 *   #2: x=FP16, boxes=FP16, box_index=INT32, crop_size=INT32, y=FP16
 *   #3: x=FP32, boxes=FP16, box_index=INT32, crop_size=INT32, y=FP16
 *
 * 构建注意：本文件注释会被构建脚本按行抓取拼接进 shell 命令，
 * 注释中勿出现注册宏名与未闭合括号（否则 CMake configure 报语法错误）。
 */

#include "register/op_def_registry.h"
#include "crop_and_resize_constraints.h"

#include <string>
#include <vector>

namespace ops {

// check_supported 回调（V1，ge::Operator）：引擎分配阶段拒绝违反 AiCore tiling 约束的
// 输入，GE 轮询 fallback 到 AiCpu 实现（性能路由，非错误）。
// 经 NeedCheckSupportFlag true + SetCheckSupport 注册；注册登记由 op_def_registry.h 中
// 的注册宏在 TILING 编译单元内自动完成（需 CMakeLists 将本文件编入 tiling_obj）。
// 值依赖：crop_size 为 const 时经 GetInputConstData 读前驱 Constant 节点 weights
// （图编译期可用，实测 check 期可读），据此完成 crop 值校验（正值、<=16、面积上限）
// 并提前路由 AiCpu；非 const 时读不到值，该项由 tiling 阶段 CheckTbeConstraints 兜底。
// 动态 shape -1 放行，由 DynamicCompileStatic 编译期兜底。
// 结构：约束判定拆为纯数据函数 CheckCropAndResizeConstraints（与 ge::Operator 适配
// 解耦，便于独立审查与后续扩展测试），FE 回调 CheckIfAICoreSupported 仅做取值适配。
struct CropAndResizeCheckInput {
    ge::DataType xDtype = ge::DT_UNDEFINED;
    ge::DataType boxesDtype = ge::DT_UNDEFINED;
    ge::DataType boxIndexDtype = ge::DT_UNDEFINED;
    // unknown rank([-2]) 统一表示：dims 清空 + 标志位置 true（由读取处归一化）
    std::vector<int64_t> xDims;     // 空 = x 为 unknown rank([-2])
    std::vector<int64_t> boxesDims; // 空 = boxes 为 unknown rank([-2])
    bool boxesUnknownRank = false;  // boxes unknown rank 标志（与 boxesDims 空等价，供放行判定直读）
    // box_index 不携带 dims（无 shape 约束），仅保留 unknown rank 标志
    bool boxIndexUnknownRank = false;
    std::string method; // 空 = 属性缺省或读取失败，均按默认 bilinear 处理
    bool cropSizeHasValue = false;
    int32_t cropH = 0;
    int32_t cropW = 0;
};

static bool CheckCropAndResizeConstraints(const CropAndResizeCheckInput& in, std::string& rejectDetail)
{
    auto rejected = [&rejectDetail](const std::string& detail) {
        rejectDetail = detail;
        return false;
    };

    // dtype：x/boxes 限 f32/f16，box_index 限 int32（AiCpu 侧 x 共收 9 种 dtype，
    // 其中 f64/uint8/int8/uint16/int16/int32/int64 这 7 种仅 AiCpu 可承接，故此处拒绝即 fallback）
    if (in.xDtype != ge::DT_FLOAT && in.xDtype != ge::DT_FLOAT16) {
        return rejected("x dtype not in {float32, float16}");
    }
    if (in.boxesDtype != ge::DT_FLOAT && in.boxesDtype != ge::DT_FLOAT16) {
        return rejected("boxes dtype not in {float32, float16}");
    }
    if (in.boxIndexDtype != ge::DT_INT32) {
        return rejected("box_index dtype must be int32");
    }

    // method 插值方式：SIMT kernel 仅实现 bilinear
    if (!in.method.empty() && in.method != "bilinear") {
        return rejected("method=" + in.method + ", only bilinear");
    }

    // unknown rank([-2]) 放行 AiCore 交动态编译（与 infershape 的 unknown rank
    // 传播一致）；若在此按 rank 判定拒绝，AiCore 认领后无 fallback 且整图编译失败。
    // x/boxes/box_index 任一为 -2 即触发
    if (in.xDims.empty() || in.boxesUnknownRank || in.boxIndexUnknownRank) {
        return true;
    }

    // x shape: 4D (N,H,W,C)
    if (in.xDims.size() != static_cast<size_t>(X_DIM)) {
        return rejected("x must be 4D (N,H,W,C)");
    }
    int64_t h = in.xDims[1];
    int64_t w = in.xDims[2];
    int64_t c = in.xDims[3];
    if (c != ge::UNKNOWN_DIM && (c < DEPTH_MIN || c > DEPTH_MAX)) {
        return rejected("depth (x.shape[3]) must be in [" + std::to_string(DEPTH_MIN) + ", " +
                        std::to_string(DEPTH_MAX) + "], got " + std::to_string(c));
    }
    // h/w 非 UNKNOWN 且为正才检查 H*W 上限（0/负维对 AiCpu 同样非法，
    // 由 tiling 阶段 all-dims-positive 拒绝，此处不重复拦）
    if (h != ge::UNKNOWN_DIM && w != ge::UNKNOWN_DIM && h > 0 && w > 0) {
        int64_t hw = h * w;
        int64_t hwMax = (in.xDtype == ge::DT_FLOAT) ? HW_FP32_MAX : HW_MAX;
        if (hw > hwMax) {
            return rejected("H*W must be <= " + std::to_string(hwMax));
        }
    }

    // boxes shape: 2D (num_boxes, 4)
    if (in.boxesDims.size() != static_cast<size_t>(BOXES_DIM)) {
        return rejected("boxes must be 2D (num_boxes, 4)");
    }
    int64_t numBoxes = in.boxesDims[0];
    if (numBoxes != ge::UNKNOWN_DIM && (numBoxes <= NUM_BOXES_MIN || numBoxes > NUM_BOXES_MAX)) {
        return rejected("num_boxes (boxes.shape[0]) must be in (" + std::to_string(NUM_BOXES_MIN) + ", " +
                        std::to_string(NUM_BOXES_MAX) + "], got " + std::to_string(numBoxes) +
                        ", performance is better on aicpu outside this range");
    }
    int64_t boxCoords = in.boxesDims[1];
    if (boxCoords != ge::UNKNOWN_DIM && boxCoords != BOX_COORDS) {
        return rejected("boxes.shape[1] must be " + std::to_string(BOX_COORDS));
    }

    // crop_size 值：正值 + max(crop_h, crop_w) <= 16 + crop 面积上限。
    // 无值（非 const，V1 读不到）时静默跳过，由 tiling 阶段兜底。
    if (in.cropSizeHasValue) {
        if (in.cropH <= 0 || in.cropW <= 0) {
            return rejected("crop_h/crop_w must be positive, got [" + std::to_string(in.cropH) + ", " +
                            std::to_string(in.cropW) + "]");
        }
        if (in.cropH > CROP_DIM_MAX || in.cropW > CROP_DIM_MAX) {
            return rejected("max(crop_h, crop_w) must be <= " + std::to_string(CROP_DIM_MAX) + ", got [" +
                            std::to_string(in.cropH) + ", " + std::to_string(in.cropW) + "]");
        }
        int64_t cropArea = static_cast<int64_t>(in.cropH) * in.cropW;
        if (cropArea > CROP_AREA_MAX) {
            return rejected("crop_h*crop_w must be <= " + std::to_string(CROP_AREA_MAX) + ", got " +
                            std::to_string(cropArea));
        }
    }
    return true;
}

// ge::TensorDesc 侧 unknown rank 判定；Ops::Base::IsUnknownRank 仅收 gert::Shape，
// ge::Operator 上下文无 gert 类型，仓内同场景惯例为 GetDims 与 UNKNOWN_RANK 比对
static bool IsUnknownRank(const ge::TensorDesc& desc) { return desc.GetShape().GetDims() == ge::UNKNOWN_RANK; }

static ge::graphStatus CheckIfAICoreSupported(const ge::Operator& op, ge::AscendString& result)
{
    CropAndResizeCheckInput in;
    // GetInputDescByName 按值返回 TensorDesc，各取一次复用
    ge::TensorDesc xDesc = op.GetInputDescByName("x");
    ge::TensorDesc boxesDesc = op.GetInputDescByName("boxes");
    ge::TensorDesc boxIndexDesc = op.GetInputDescByName("box_index");
    in.xDtype = xDesc.GetDataType();
    in.boxesDtype = boxesDesc.GetDataType();
    in.boxIndexDtype = boxIndexDesc.GetDataType();
    in.boxIndexUnknownRank = IsUnknownRank(boxIndexDesc);
    std::vector<int64_t> xDims = xDesc.GetShape().GetDims();
    std::vector<int64_t> boxesDims = boxesDesc.GetShape().GetDims();
    // unknown rank([-2]) 归一化：dims 清空 + 标志位置位（见结构体注释）
    if (IsUnknownRank(xDesc)) {
        in.xDims.clear();
    } else {
        in.xDims = std::move(xDims);
    }
    in.boxesUnknownRank = IsUnknownRank(boxesDesc);
    if (in.boxesUnknownRank) {
        in.boxesDims.clear();
    } else {
        in.boxesDims = std::move(boxesDims);
    }
    (void)op.GetAttr("method", in.method);
    // crop_size 值（V1 独有能力：const 时经 GetInputConstData 读前驱 Constant 节点 weights）
    ge::Tensor cropTensor;
    if (op.GetInputConstData("crop_size", cropTensor) == ge::GRAPH_SUCCESS && cropTensor.GetData() != nullptr &&
        cropTensor.GetSize() == CROP_SIZE_LEN * sizeof(int32_t)) {
        const int32_t* cropData = reinterpret_cast<const int32_t*>(cropTensor.GetData());
        in.cropSizeHasValue = true;
        in.cropH = cropData[0];
        in.cropW = cropData[1];
    }

    std::string rejectDetail;
    if (!CheckCropAndResizeConstraints(in, rejectDetail)) {
        // reason 带性能路由语义，提示该拒绝为路由行为而非错误
        std::string json = "{\"isSupported\": \"False\", \"dynamicCompileStatic\": \"True\", \"reason\": "
                           "\"AiCore path of CropAndResize on ascend950 does not support this input (" +
                           rejectDetail + "), performance/compatibility is better on AiCpu, fallback expected.\"}";
        result = ge::AscendString(json.c_str());
        return ge::GRAPH_FAILED;
    }
    result = ge::AscendString(
        "{\"isSupported\": \"True\", \"dynamicCompileStatic\": \"True\", \"reason\": \"AICore CheckSupport Passed.\"}");
    return ge::GRAPH_SUCCESS;
}

class CropAndResize : public OpDef {
public:
    explicit CropAndResize(const char* name) : OpDef(name)
    {
        // 输入 x: 4D (N, H, W, C), float16/float32
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 boxes: 2D (num_boxes, 4), float16/float32
        // NaN 检查已移至 kernel 运行时，tiling 不再读取 boxes 值，无需 ValueDepend
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 box_index: 1D (num_boxes,), int32
        this->Input("box_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 crop_size: 1D (2,), int32
        // ValueDepend(OPTIONAL): infershape/tiling 需读取 crop_size 值（与 MDE §3.3 一致）
        this->Input("crop_size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();

        // 输出 y: 4D (num_boxes, crop_h, crop_w, C), dtype = boxes dtype
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 属性: extrapolation_value (Float, 默认 0)
        this->Attr("extrapolation_value").AttrType(OPTIONAL).Float(0);

        // 属性: method (String, 默认 "bilinear", 本次仅支持 bilinear)
        this->Attr("method").AttrType(OPTIONAL).String("bilinear");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            // true: 引擎分配阶段回调 CheckIfAICoreSupported —— 见上方实现，V1 版，
            // 可读 desc/dtype/attrs/crop_size const 值。tiling 约束不满足时 AIcoreEngine
            // 拒绝并 fallback 到 AiCpu
            .NeedCheckSupportFlag(true)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "crop_and_resize_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().SetCheckSupport(CheckIfAICoreSupported);
    }
};

OP_ADD(CropAndResize);
} // namespace ops
