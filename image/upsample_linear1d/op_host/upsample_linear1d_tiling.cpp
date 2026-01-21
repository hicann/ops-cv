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
 * \file upsample_linear1d_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_linear1d_tiling.h"

namespace optiling {
constexpr uint32_t BEST_PERFORMANCE_SIZE_16 = 16;
constexpr uint32_t BEST_PERFORMANCE_SIZE_32 = 32;
constexpr uint32_t BEST_PERFORMANCE_SIZE_48 = 48;
constexpr uint32_t BEST_PERFORMANCE_SIZE_64 = 64;
constexpr uint32_t BEST_PERFORMANCE_SIZE_128 = 128;

constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_50 = 50;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_20 = 20;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_8 = 8;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_5 = 5;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_1 = 1;

constexpr float HALF_NUM = 0.5;

constexpr uint32_t BYTE = 8;
constexpr uint32_t BYTE_REPEAT = 256;  // The amount of data that can be processed by a repeat.
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;
constexpr int8_t DIM_ZERO = 0;
constexpr int8_t DIM_ONE = 1;
constexpr int8_t DIM_TWO = 2;
constexpr int8_t DIM_THREE = 3;

constexpr int8_t MODE_LINEAR = 1;
constexpr int8_t MODE_BILINEAR = 2;

constexpr uint32_t ALIGN_CORNERS_ATTR = 0;
constexpr uint32_t SCALES_ATTR = 1;

constexpr uint64_t DATE_TYPE_FLOAT16 = 1;
constexpr uint64_t DATE_TYPE_FLOAT = 2;
constexpr uint64_t DATE_TYPE_HALF = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint32_t DIM_LEN = 4;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;

constexpr int8_t NUM_ONE = 1;
constexpr int8_t NUM_TWO = 2;
constexpr int8_t NUM_FIVE = 5;
constexpr int64_t NUM_1024 = 1024;

constexpr float MAX_SUPPORT_SHRINK_SCALE = 50.0f;
constexpr float MAX_SUPPORT_ZOOM_SCALE = 800.0f;
constexpr float MAX_SUPPORT_ZOOM_SCALE_REV = 0.00125f;

constexpr float support = 1.0;
constexpr int64_t max_interp_size = 2;
constexpr int64_t max_interp_size_10 = 10;
constexpr uint8_t SCHEDULE_MODE = 1;

class UpsampleLinear1dTiling {
public:
    explicit UpsampleLinear1dTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling(const uint8_t modeNum);

private:
    void setScale(const uint8_t mode);
    void get_scale_from_out(const uint8_t mode);
    inline float compute_scale_value(
        const int64_t input_size, const int64_t output_size, const bool align_corner, const float scale) const;
    bool getWorkSpace(const uint32_t needCoreNum);
    void getShapes(const uint8_t mode);
    void setSlideSize(const uint32_t coreNumPlatFormInfo, const uint8_t mode);
    inline int64_t calculateSlideSize(const uint32_t coreNumPlatFormInfo, uint8_t direction);
    inline int64_t getSlideSizeByScale(const uint32_t coreNumPlatFormInfo, uint8_t direction, float real_scale);
    uint8_t GetDataTypeSize() const;
    uint64_t GetDataTypeVal() const;
    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatFormInfo, const uint8_t mode);
    uint32_t GetNeedCoreNumW(const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size);
    uint32_t GetNeedCoreNumH(const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size);
    void FillTilingData();
    void getTCubeTiling_w();
    void getTCubeTiling_h();
    inline bool CheckScales(
        const gert::TilingContext *context, const float scales_w, const float scales_h, const uint8_t mode) const;
    inline int64_t getSingleCoreK(const int64_t slideSize, const float scale, const bool alignCorners) const;
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    int64_t slide_size_w{16};
    int64_t slide_size_h{16};
    UpsampleLinear1dTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize{4};
    gert::Shape input_shape;
    gert::Shape output_shape;
    const bool *align_corners{nullptr};
    float scale_h = 0.0f;
    float scale_w = 0.0f;
    float realScale_h{0.0f};
    float realScale_w{0.0f};

    int64_t output_shapes[4] = {0};
    int64_t input_shapes[4] = {0};

    int64_t slide_size_list[5] = {BEST_PERFORMANCE_SIZE_16,
        BEST_PERFORMANCE_SIZE_32,
        BEST_PERFORMANCE_SIZE_48,
        BEST_PERFORMANCE_SIZE_64,
        BEST_PERFORMANCE_SIZE_128};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
    int64_t singleCoreK_w = 0;
    int64_t singleCoreK_h = 0;
    uint32_t coreNumPlatForm = 20;
};

void UpsampleLinear1dTiling::setScale(const uint8_t mode)
{
    if (mode == MODE_BILINEAR) {
        realScale_h = compute_scale_value(input_shapes[H_INDEX], output_shapes[H_INDEX], *align_corners, scale_h);
        realScale_w = compute_scale_value(input_shapes[W_INDEX], output_shapes[W_INDEX], *align_corners, scale_w);
    } else {
        realScale_h = 1.0;
        realScale_w = compute_scale_value(input_shapes[W_INDEX], output_shapes[W_INDEX], *align_corners, scale_w);
    }

    tilingData.set_scale_h(realScale_h);
    tilingData.set_scale_w(realScale_w);
}

void UpsampleLinear1dTiling::get_scale_from_out(const uint8_t mode)
{
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    align_corners = attrs->GetAttrPointer<bool>(ALIGN_CORNERS_ATTR);
    if (mode == MODE_LINEAR) {
        const float *scales = attrs->GetAttrPointer<float>(SCALES_ATTR);
        scale_h = 1.0f;
        scale_w = *scales;
    } else {
        const gert::ContinuousVector *scalesAttr = attrs->GetAttrPointer<gert::ContinuousVector>(SCALES_ATTR);
        const float *scalesArray = reinterpret_cast<const float *>(scalesAttr->GetData());
        scale_h = scalesArray[DIM_ZERO];
        scale_w = scalesArray[DIM_ONE];
    }
}

inline float UpsampleLinear1dTiling::compute_scale_value(
    const int64_t input_size, const int64_t output_size, const bool align_corner, const float scale) const
{
    if (output_size == input_size) {
        return static_cast<float>(1);
    }
    if (align_corner) {
        if (output_size > 1) {
            return static_cast<float>(input_size - 1) / (output_size - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        return (scale > 0) ? static_cast<float>(scale) : (static_cast<float>(input_size) / output_size);
    }
}

inline bool UpsampleLinear1dTiling::CheckScales(
    const gert::TilingContext *context, const float scales_w, const float scales_h, const uint8_t mode) const
{
    if (mode == MODE_LINEAR) {
        // 1D的放大支持800倍，缩小支持50倍
        float maxSupport = scales_w < 1 ? MAX_SUPPORT_ZOOM_SCALE : MAX_SUPPORT_SHRINK_SCALE;
        OP_CHECK_IF(((scales_w < 1 && scales_w < MAX_SUPPORT_ZOOM_SCALE_REV) || (scales_w > MAX_SUPPORT_SHRINK_SCALE)),
            OP_LOGE(context->GetNodeName(),
                "Scales should not exceed %f, but got scale (scales: %f) ",
                maxSupport,
                scales_w),
            return false);
    } else {
        // 2D都限制50倍
        OP_CHECK_IF((scales_h > MAX_SUPPORT_SHRINK_SCALE || scales_w > MAX_SUPPORT_SHRINK_SCALE),
            OP_LOGE(context->GetNodeName(),
                "Scales should not exceed 50, but got scale (scales_w: %f, scales_h: %f) ",
                scales_w,
                scales_h),
            return false);
    }
    return true;
}

inline bool FloatEqual(const float a, const float b)
{
    const float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

ge::graphStatus UpsampleLinear1dTiling::RunBigKernelTiling(const uint8_t modeNum)
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    get_scale_from_out(modeNum);

    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType srcDtype = ge::DT_UNDEFINED;
    srcDtype = temp->GetDataType();

    // Determine whether all data types are consistent.
    if (dataType == ge::DT_UNDEFINED) {
        dataType = srcDtype;
        dataTypeSize = GetDataTypeSize();
    } else if (srcDtype != dataType) {
        return ge::GRAPH_FAILED;
    }
    auto src_shape = tilingContext->GetInputShape(0);
    auto dst_shape = tilingContext->GetOutputShape(0);

    input_shape = src_shape->GetOriginShape();
    output_shape = dst_shape->GetOriginShape();

    auto compileInfo = reinterpret_cast<const UpsampleLinear1dCompileInfo *>(tilingContext->GetCompileInfo());
    if (compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t coreNumPlatFormInfo = compileInfo->coreNum;
    if (coreNumPlatFormInfo < 1) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetTilingKey(1);
    tilingData.set_mode(modeNum);
    tilingData.set_align_corners(*align_corners);
    getShapes(modeNum);
    setScale(modeNum);
    if (!CheckScales(tilingContext, realScale_w, realScale_h, modeNum)) {
        return ge::GRAPH_FAILED;
    }

    setSlideSize(coreNumPlatFormInfo, modeNum);
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatFormInfo, modeNum);
    // 计算workspace，每个核的系数矩阵（是否要乘2，避免doubelBuffer矩阵相互影响？），中间矩阵大小
    if (!getWorkSpace(needCoreNum)) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetBlockDim(needCoreNum);
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

uint32_t UpsampleLinear1dTiling::GetNeedCoreNum(const uint32_t coreNumPlatFormInfo, const uint8_t mode)
{
    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    if (!FloatEqual(realScale_w, 1.0) || mode == MODE_LINEAR) {
        int64_t kMinValue = getSingleCoreK(0, realScale_w, *align_corners);
        int64_t kMaxValue = getSingleCoreK(slide_size_w, realScale_w, *align_corners);
        singleCoreK_w = kMaxValue - kMinValue + Ceil(max_interp_size_10);
        if (singleCoreK_w > input_shapes[W_INDEX]) {
            singleCoreK_w = input_shapes[W_INDEX];
        }
        needCoreNumW = GetNeedCoreNumW(coreNumPlatFormInfo, NUM_TWO, slide_size_w);
        getTCubeTiling_w();
    }

    if (mode == MODE_BILINEAR && (!FloatEqual(realScale_h, 1.0) || FloatEqual(realScale_w, 1.0))) {
        singleCoreK_h = Ceil(slide_size_h * realScale_h) + Ceil(max_interp_size);
        if (singleCoreK_h > input_shapes[H_INDEX]) {
            singleCoreK_h = input_shapes[H_INDEX];
        }
        needCoreNumH = GetNeedCoreNumH(coreNumPlatFormInfo, NUM_TWO, slide_size_h);
        getTCubeTiling_h();
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

void UpsampleLinear1dTiling::getTCubeTiling_w()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_w.SetOrgShape(input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shapes[H_INDEX],
        output_shapes[W_INDEX],
        input_shapes[W_INDEX]);
    mmTiling_w.SetShape(
        input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shapes[H_INDEX], slide_size_w, singleCoreK_w);
    if (mmTiling_w.GetTiling(tilingData.matmulTiling_w) == -1) {
        return;
    }
}

void UpsampleLinear1dTiling::getTCubeTiling_h()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_h;
    mmTiling_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_h.SetOrgShape(output_shapes[H_INDEX], output_shapes[W_INDEX], input_shapes[W_INDEX]);
    mmTiling_h.SetShape(slide_size_h, output_shapes[W_INDEX], singleCoreK_h);

    if (mmTiling_h.GetTiling(tilingData.matmulTiling_h) == -1) {
        return;
    }
}

// 先只算w方向
bool UpsampleLinear1dTiling::getWorkSpace(const uint32_t needCoreNum)
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return false;
    }
    // 中间tensor
    uint64_t intermediate_matrix_size =
        output_shapes[0] * output_shapes[1] * input_shapes[2] * output_shapes[3] * dataTypeSize;
    intermediate_matrix_size = (intermediate_matrix_size + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    // 每个核的系数矩阵，每个核申请两个workspace空间，避免相互覆盖
    int64_t singleCoreK = singleCoreK_w > singleCoreK_h ? singleCoreK_w : singleCoreK_h;
    int64_t slide_size = std::max(slide_size_w, slide_size_h);
    uint32_t radioMatrixWorkspaceSize = slide_size * singleCoreK * dataTypeSize;
    workspaces[0] = intermediate_matrix_size + radioMatrixWorkspaceSize * needCoreNum + WORK_SPACE_SIZE;
    tilingData.set_radio_matrix_size_w(slide_size_w * singleCoreK_w);
    tilingData.set_radio_matrix_size_h(slide_size_h * singleCoreK_h);
    tilingData.set_intermediate_matrix_size(intermediate_matrix_size);
    return true;
}

void UpsampleLinear1dTiling::getShapes(const uint8_t mode)
{
    for (int8_t i = 0; i < SHAPE_SIZE; i++) {
        input_shapes[i] = input_shape.GetDim(i);
        output_shapes[i] = input_shape.GetDim(i);
        if (i > C_INDEX && mode == MODE_BILINEAR) {
            output_shapes[i] = output_shape.GetDim(i);
        } else {
            output_shapes[DIM_TWO] = 1;
            output_shapes[DIM_THREE] = output_shape.GetDim(DIM_TWO);
        }
    }

    tilingData.set_input_shapes(input_shapes);
    tilingData.set_output_shapes(output_shapes);
}

void UpsampleLinear1dTiling::setSlideSize(const uint32_t coreNumPlatFormInfo, const uint8_t mode)
{
    slide_size_w = getSlideSizeByScale(coreNumPlatFormInfo, NUM_ONE, realScale_w);
    if (mode == MODE_BILINEAR) {
        slide_size_h = getSlideSizeByScale(coreNumPlatFormInfo, NUM_TWO, realScale_h);
    }
    tilingData.set_slide_size_w(slide_size_w);
    tilingData.set_slide_size_h(slide_size_h);
}

inline int64_t UpsampleLinear1dTiling::getSlideSizeByScale(
    const uint32_t coreNumPlatFormInfo, uint8_t direction, float real_scale)
{
    int64_t slide_size = 16;

    int64_t slideSizeBysize = calculateSlideSize(coreNumPlatFormInfo, direction);
    if (input_shapes[DIM_TWO] == NUM_ONE && input_shapes[DIM_THREE] == NUM_ONE) {
        slideSizeBysize = NUM_1024;
    }
    if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_1) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_128), slideSizeBysize);
    } else if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_5) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_64), slideSizeBysize);
    } else if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_8) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_48), slideSizeBysize);
    } else if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_20) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_32), slideSizeBysize);
    } else {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_16), slideSizeBysize);
    }
    return slide_size;
}

inline int64_t UpsampleLinear1dTiling::calculateSlideSize(const uint32_t coreNumPlatFormInfo, uint8_t direction)
{
    int64_t slide_size = BEST_PERFORMANCE_SIZE_16;
    uint32_t neeCoreNumMax = 0;

    for (uint32_t coreIndex = 0; coreIndex < NUM_FIVE; coreIndex++) {
        uint32_t res = 0;
        if (direction == NUM_ONE) {
            res = GetNeedCoreNumW(coreNumPlatFormInfo, NUM_ONE, slide_size_list[coreIndex]);
        } else {
            res = GetNeedCoreNumH(coreNumPlatFormInfo, NUM_ONE, slide_size_list[coreIndex]);
        }
        slide_size = res >= neeCoreNumMax ? slide_size_list[coreIndex] : slide_size;
        neeCoreNumMax = std::max(res, neeCoreNumMax);
    }
    return slide_size;
}

template <typename T1, typename T2>
inline auto UpsampleLinear1dTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleLinear1dTiling::Ceil(T1 x) const
{
    int32_t floor_x = int32_t(x);
    if (x == static_cast<T1>(floor_x)) {
        return floor_x;
    }
    return floor_x + 1;
}

inline int64_t UpsampleLinear1dTiling::getSingleCoreK(
    const int64_t slideSize, const float scale, const bool alignCorners) const
{
    float rel_idx = 0;
    float calNum = std::max(0.0f, static_cast<float>(slideSize - 1));
    if (alignCorners) {
        rel_idx = scale * calNum;
    } else {
        rel_idx = std::max(static_cast<float>((calNum + HALF_NUM) * scale - HALF_NUM), static_cast<float>(0.0));
    }
    int64_t floor_rel_idx = int32_t(rel_idx);
    return floor_rel_idx;
};

uint8_t UpsampleLinear1dTiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

uint64_t UpsampleLinear1dTiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return DATE_TYPE_FLOAT;
        case ge::DT_FLOAT16:
            return DATE_TYPE_FLOAT16;
        case ge::DT_BF16:
            return DATE_TYPE_HALF;
        default:
            return 0;
    }
}

uint32_t UpsampleLinear1dTiling::GetNeedCoreNumW(
    const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size)
{
    int64_t outputSize = output_shapes[3];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // H维度总数
    int64_t input_h = input_shapes[0] * input_shapes[1] * input_shapes[2];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRows = slide_size;

    if (remainder != 0) {
        // 获取最小分行数
        int64_t minAvergingRows = slide_size;
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRows = std::max(CeilA2B(input_h, groupCoreNum), minAvergingRows);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(input_h, tailAvergingRows));
    }
    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    } else if (remainder != 0) {
        for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
            groupCoreNum = groupCoreNum == 0 ? 1 : groupCoreNum;
            // 尾块处理
            int64_t groupIndex = coreIndex / groupCoreNum;
            if (groupIndex < remainder) {
                needCoreNum++;
            }
        }
    }

    if (isCalculate == NUM_TWO) {
        tilingData.set_eachCoreSlideNumW(eachCoreSlideNum);
        tilingData.set_tailStartSlideNumW(tailStartSlideNum);
        tilingData.set_slideNumW(slideNum);
        tilingData.set_groupCoreNumW(groupCoreNum);
        tilingData.set_tailAvergingRowsW(tailAvergingRows);
        tilingData.set_remainderW(remainder);
        tilingData.set_need_core_num_w(needCoreNum);
    }

    return needCoreNum;
}

uint32_t UpsampleLinear1dTiling::GetNeedCoreNumH(
    const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size)
{
    int64_t outputSize = output_shapes[2];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // Batch和W维度总数
    int64_t batch = input_shapes[0] * input_shapes[1];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingBatch = slide_size;
    if (remainder != 0) {
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingBatch = CeilA2B(batch, groupCoreNum);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(batch, tailAvergingBatch));
    }

    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    } else if (remainder != 0) {
        for (uint32_t coreIndexH = 0; coreIndexH < coreNumPlatform; coreIndexH++) {
            groupCoreNum = groupCoreNum == 0 ? 1 : groupCoreNum;
            // 尾块处理, 核数不全都一样
            int64_t groupIndex = coreIndexH / groupCoreNum;
            if (groupIndex < remainder) {
                needCoreNum++;
            }
        }
    }
    if (isCalculate == NUM_TWO) {
        tilingData.set_eachCoreSlideNumH(eachCoreSlideNum);
        tilingData.set_tailStartSlideNumH(tailStartSlideNum);
        tilingData.set_slideNumH(slideNum);
        tilingData.set_groupCoreNumH(groupCoreNum);
        tilingData.set_tailAvergingRowsH(tailAvergingBatch);
        tilingData.set_remainderH(remainder);
        tilingData.set_need_core_num_h(needCoreNum);
    }
    return needCoreNum;
}

void UpsampleLinear1dTiling::FillTilingData()
{
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleLinear1dTiling(gert::TilingContext *context)
{
    UpsampleLinear1dTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling(MODE_LINEAR);
}

static ge::graphStatus tiling4UpsampleBilinear2dTiling(gert::TilingContext *context)
{
    UpsampleLinear1dTiling tilingObject(context);
    context->SetScheduleMode(SCHEDULE_MODE);
    return tilingObject.RunBigKernelTiling(MODE_BILINEAR);
}

static ge::graphStatus tilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<UpsampleLinear1dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(compileInfo->coreNum <= 0,
        OP_LOGE(
            context->GetNodeName(), "UpsampleLinear1d GetHardwareInfo Failed, vectorCoreNum:%u", compileInfo->coreNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleLinear1d)
    .Tiling(tiling4UpsampleLinear1dTiling)
    .TilingParse<UpsampleLinear1dCompileInfo>(tilingPrepareTiling);

IMPL_OP_OPTILING(UpsampleBilinear2d)
    .Tiling(tiling4UpsampleBilinear2dTiling)
    .TilingParse<UpsampleLinear1dCompileInfo>(tilingPrepareTiling);
}  // namespace optiling
