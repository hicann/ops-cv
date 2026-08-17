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
 * \file crop_and_resize_simt.h
 * \brief SIMT kernel implementation for crop_and_resize operator
 *
 * Grid-Stride over (b, cy, cx) positions, each position loops C channels.
 * All intermediate computation uses float32 for precision alignment with TF.
 *
 * 索引位宽模板化 IDX_T(uint32_t/uint64_t)，数据量 <= INT32_MAX 走 32 位高效路径。
 * __launch_bounds__ 按 IDX_T 位宽模板化，uint32 路径开 1024 线程，uint64 路径开 512 线程。
 *
 * 标量参数通过 UB 传递，解决 VF 标量参数过多问题。
 *
 * 精度对齐方案（与 TF image.crop_and_resize bit-exact）：
 *   1. volatile — 阻止 NPU 编译器将 a*b+c 收缩为 fma(a,b,c)（单舍入），
 *      匹配 TF Eigen tensor 的 separate mul+add（双舍入）。
 *      -ffp-contract=off 和 #pragma clang fp contract(off) 在 NPU 编译器上均不可靠，
 *      volatile 是经实证验证的唯一有效 FMA 防护手段。
 *   2. DsDiv — fmaf 修正除法残差，消除 NPU 与 numpy 除法的 1-ULP 坐标差异，
 *      使 OOB 判定完全一致。
 *   3. IEEE 754 自然传播 — 不使用 t==0 guard，0*Inf=NaN 自然传播匹配 TF。
 */

#ifndef CROP_AND_RESIZE_SIMT_H_
#define CROP_AND_RESIZE_SIMT_H_

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/device_sync_functions.h"
#include "crop_and_resize_tiling_data.h"

namespace NsCropAndResize {
using namespace AscendC;

// 线程数模板：uint32 路径开 1024 线程，uint64 路径开 512 线程
template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == sizeof(uint32_t)) ? 1024 : 512;

// UB 参数布局（scalar 端写入，VF 端读取，解决 VF 标量参数过多问题）
static constexpr uint32_t UB_OFF_BATCH = 0;
static constexpr uint32_t UB_OFF_IMG_H = 1;
static constexpr uint32_t UB_OFF_IMG_W = 2;
static constexpr uint32_t UB_OFF_DEPTH = 3;
static constexpr uint32_t UB_OFF_CROP_H = 4;
static constexpr uint32_t UB_OFF_CROP_W = 5;
static constexpr uint32_t UB_OFF_EXT_BITS = 6; // extrapolationValue 的 float 位模式（存为 int32）
static constexpr uint32_t UB_PARAM_COUNT = 7;

// 每个 box 的坐标数 [y1, x1, y2, x2]
static constexpr int32_t BOX_COORDS_PER_BOX = 4;

// MapCoordinate cropSize==1 时的中点权重
static constexpr float MIDPOINT_WEIGHT = 0.5f;

// 以下 union 位转换依赖编译器 union 双关扩展（Clang/GCC 支持），非 ISO C++ 标准行为。
// int32_t/uint32_t/float 均为 32 位同尺寸类型，无数据损坏风险。
static_assert(sizeof(int32_t) == sizeof(float), "int32_t and float must have same size");
static_assert(sizeof(uint32_t) == sizeof(float), "uint32_t and float must have same size");

// 从 UB 加载的标量参数集合
struct ScalarParams {
    int32_t batch;
    int32_t imageHeight;
    int32_t imageWidth;
    int32_t depth;
    int32_t cropHeight;
    int32_t cropWidth;
    float extrapolationValue;
};

// GM 指针集合
template <typename T_X, typename T_BOXES, typename T_Y>
struct GmPointers {
    __gm__ T_X* xGm;
    __gm__ T_BOXES* boxesGm;
    __gm__ int32_t* boxIndexGm;
    __gm__ T_Y* yGm;
};

// 预计算的几何常量
struct Geometry {
    float imgHMinus1;
    float imgWMinus1;
    int64_t imgHWC;  // imageHeight * imageWidth * depth
    int64_t cropHWC; // cropHeight * cropWidth * depth
};

// 插值几何参数（邻居索引 + 权重 + 偏移）
struct InterpGeometry {
    int32_t topY;
    int32_t bottomY;
    int32_t leftX;
    int32_t rightX;
    float wy1;
    float wx1;
    int64_t imgBase;
    int64_t topRowOff;
    int64_t bottomRowOff;
    int64_t leftColOff;
    int64_t rightColOff;
};

// dtype 转换
template <typename T>
__simt_callee__ inline float CastToFloat(T val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else {
        return static_cast<float>(val);
    }
}

template <typename T>
__simt_callee__ inline T CastFromFloat(float val)
{
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(val);
    } else {
        return static_cast<T>(val);
    }
}

// float -> int32 位转换（__aicore__ 环境，无 trap）
// __aicore__ 环境（LaunchVf）使用 union 直接转换，避免 __simt_callee__ 调用开销。
__aicore__ inline int32_t FloatToBitsAicore(float val)
{
    union {
        int32_t i;
        float f;
    } c;
    c.f = val;
    return c.i;
}

// int32 -> float 位转换（__simt_callee__ 环境，与 FloatToBitsAicore 对称）
__simt_callee__ inline float BitsToFloatSimt(int32_t bits)
{
    union {
        int32_t i;
        float f;
    } c;
    c.i = bits;
    return c.f;
}

// IEEE 754 特殊值构造（union 位转换，无 trap）
__simt_callee__ inline float MakeNan()
{
    union {
        uint32_t u;
        float f;
    } c;
    c.u = 0x7FC00000U; // quiet NaN
    return c.f;
}

// DsDiv: 精确除法，消除 NPU 与 numpy 的 1-ULP 差异
// 普通除法 a/b 在 NPU 和 numpy 上可能因舍入方向不同产生 1-ULP 差异，
// 导致坐标映射的 OOB 判定不一致。DsDiv 用 fmaf 计算残差并修正：
//   q = a/b, residual = fmaf(q, b, -a), result = q - residual/b
// 调用方保证 b != 0（cropSize > 1 时除数 cropSize-1 >= 1）。
__simt_callee__ inline float DsDiv(float a, float b)
{
    float q = a / b;
    float residual = fmaf(q, b, -a);
    return q - residual / b;
}

// 将整数索引 clamp 到 [0, hi] 范围
__simt_callee__ inline int32_t ClampInt32(int32_t val, int32_t hi) { return min(hi, max(val, 0)); }

// 计算插值几何参数：邻居索引 + 权重 + 地址偏移
__simt_callee__ inline InterpGeometry ComputeInterpGeometry(float inY, float inX, int32_t boxIdx,
                                                            const ScalarParams& params, const Geometry& geom)
{
    int32_t depth = params.depth;
    InterpGeometry ig;
    ig.topY = ClampInt32(static_cast<int32_t>(floorf(inY)), params.imageHeight - 1);
    ig.bottomY = ClampInt32(static_cast<int32_t>(ceilf(inY)), params.imageHeight - 1);
    ig.leftX = ClampInt32(static_cast<int32_t>(floorf(inX)), params.imageWidth - 1);
    ig.rightX = ClampInt32(static_cast<int32_t>(ceilf(inX)), params.imageWidth - 1);
    ig.wy1 = inY - static_cast<float>(ig.topY);
    ig.wx1 = inX - static_cast<float>(ig.leftX);
    ig.imgBase = static_cast<int64_t>(boxIdx) * geom.imgHWC;
    ig.topRowOff = static_cast<int64_t>(ig.topY) * params.imageWidth * depth;
    ig.bottomRowOff = static_cast<int64_t>(ig.bottomY) * params.imageWidth * depth;
    ig.leftColOff = static_cast<int64_t>(ig.leftX) * depth;
    ig.rightColOff = static_cast<int64_t>(ig.rightX) * depth;
    return ig;
}

// 单通道双线性插值
// 像素布局: v00---v01 (top row), v10---v11 (bottom row)
// lerp 形式: a + t*(b-a)，volatile 阻止 FMA contraction，匹配 TF separate mul+add。
template <typename T_X>
__simt_callee__ inline float BilinearSampleChannel(const __gm__ T_X* xGm, const InterpGeometry& ig, int32_t d)
{
    float v00 = CastToFloat(xGm[ig.imgBase + ig.topRowOff + ig.leftColOff + d]);
    float v01 = CastToFloat(xGm[ig.imgBase + ig.topRowOff + ig.rightColOff + d]);
    float v10 = CastToFloat(xGm[ig.imgBase + ig.bottomRowOff + ig.leftColOff + d]);
    float v11 = CastToFloat(xGm[ig.imgBase + ig.bottomRowOff + ig.rightColOff + d]);

    volatile float diff0 = v01 - v00;
    volatile float prod0 = ig.wx1 * diff0;
    float row0 = v00 + prod0;

    volatile float diff1 = v11 - v10;
    volatile float prod1 = ig.wx1 * diff1;
    float row1 = v10 + prod1;

    volatile float diff2 = row1 - row0;
    volatile float prod2 = ig.wy1 * diff2;
    return row0 + prod2;
}

// 坐标映射：将 crop 坐标映射到原图坐标
// TF 公式：
//   if cropSize > 1: scale = (boxEnd - boxStart) * (imgSize-1) / (cropSize-1)
//                    coord = boxStart * (imgSize-1) + cropIdx * scale
//   else:             coord = 0.5 * (boxStart + boxEnd) * (imgSize-1)
// DsDiv 修正 NPU plain division 与 x86 的 1-ULP 差异，匹配 TF 除法结果。
// volatile 阻止 FMA contraction。
__simt_callee__ inline float MapCoordinate(float boxStart, float boxEnd, int32_t cropIdx, int32_t cropSize,
                                           float imgSizeMinus1)
{
    if (cropSize > 1) {
        volatile float boxDiff = boxEnd - boxStart;
        volatile float num = boxDiff * imgSizeMinus1;
        volatile float scale = DsDiv(num, static_cast<float>(cropSize - 1));
        volatile float prod = static_cast<float>(cropIdx) * scale;
        volatile float base = boxStart * imgSizeMinus1;
        volatile float coord = base + prod;
        return coord;
    } else {
        float sum = boxStart + boxEnd;
        float coord = MIDPOINT_WEIGHT * sum * imgSizeMinus1;
        return coord;
    }
}

// 从 UB 加载标量参数
__simt_callee__ inline ScalarParams LoadParamsFromUb(const __ubuf__ int32_t* ub)
{
    ScalarParams params;
    params.batch = ub[UB_OFF_BATCH];
    params.imageHeight = ub[UB_OFF_IMG_H];
    params.imageWidth = ub[UB_OFF_IMG_W];
    params.depth = ub[UB_OFF_DEPTH];
    params.cropHeight = ub[UB_OFF_CROP_H];
    params.cropWidth = ub[UB_OFF_CROP_W];
    params.extrapolationValue = BitsToFloatSimt(ub[UB_OFF_EXT_BITS]);
    return params;
}

// 预计算几何常量
__simt_callee__ inline Geometry ComputeGeometry(const ScalarParams& params)
{
    Geometry geom;
    geom.imgHMinus1 = static_cast<float>(params.imageHeight - 1);
    geom.imgWMinus1 = static_cast<float>(params.imageWidth - 1);
    geom.imgHWC = static_cast<int64_t>(params.imageHeight) * params.imageWidth * params.depth;
    geom.cropHWC = static_cast<int64_t>(params.cropHeight) * params.cropWidth * params.depth;
    return geom;
}

// 统一填充：所有通道写同一个值
template <typename T_Y>
__simt_callee__ inline void FillAllChannels(int32_t depth, int64_t outBase, float val, __gm__ T_Y* yGm)
{
    for (int32_t d = 0; d < depth; d++) {
        yGm[outBase + d] = CastFromFloat<T_Y>(val);
    }
}

// 从 TilingData 构造 ScalarParams
__aicore__ inline ScalarParams MakeScalarParams(const CropAndResizeTilingData* t)
{
    ScalarParams params;
    params.batch = t->batch;
    params.imageHeight = t->imageHeight;
    params.imageWidth = t->imageWidth;
    params.depth = t->depth;
    params.cropHeight = t->cropHeight;
    params.cropWidth = t->cropWidth;
    params.extrapolationValue = t->extrapolationValue;
    return params;
}

// 处理单个输出位置 (box, crop_y, crop_x)
// 完整流程：读取 box -> NaN 检测 -> 坐标映射 -> OOB 检查 -> 双线性插值
template <typename T_X, typename T_BOXES, typename T_Y, typename IDX_T>
__simt_callee__ inline void ProcessOnePosition(IDX_T b, IDX_T cy, IDX_T cx, const ScalarParams& params,
                                               const Geometry& geom, GmPointers<T_X, T_BOXES, T_Y> gms)
{
    int32_t depth = params.depth;
    int64_t batchOff = static_cast<int64_t>(b) * geom.cropHWC;
    int64_t rowOff = static_cast<int64_t>(cy) * params.cropWidth * depth;
    int64_t outBase = batchOff + rowOff + static_cast<int64_t>(cx) * depth;

    // box_index 越界：防御性填充 0（TF 会抛 InvalidArgumentError，测试用例应避免越界）
    int32_t boxIdx = gms.boxIndexGm[static_cast<int64_t>(b)];
    if (boxIdx < 0 || boxIdx >= params.batch) {
        FillAllChannels<T_Y>(depth, outBase, 0.0f, gms.yGm);
        return;
    }

    // 读取 box 归一化坐标 [y1, x1, y2, x2]（范围 0~1）
    // int64_t 避免 b * 4 在 int32 域溢出（b > 536M 时回绕为负数导致越界读）
    int64_t boxOffset = static_cast<int64_t>(b) * BOX_COORDS_PER_BOX;
    float y1 = CastToFloat(gms.boxesGm[boxOffset + 0]);
    float x1 = CastToFloat(gms.boxesGm[boxOffset + 1]);
    float y2 = CastToFloat(gms.boxesGm[boxOffset + 2]);
    float x2 = CastToFloat(gms.boxesGm[boxOffset + 3]);

    // NaN/Inf 检测：boxes 含 NaN 或 Inf 时输出 NaN（与 TF 行为一致）
    // NaN 会导致 floor(NaN) 产生未定义索引；Inf 经 MapCoordinate 会产生 NaN（Inf-Inf=NaN），
    // 且 NaN bypass OOB 检查后 static_cast<int32_t>(NaN) 为 UB，需提前拦截。
    // isnan()/isinf() 在 __simt_callee__ 中可靠工作。
    if (isnan(y1) || isnan(x1) || isnan(y2) || isnan(x2) || isinf(y1) || isinf(x1) || isinf(y2) || isinf(x2)) {
        FillAllChannels<T_Y>(depth, outBase, MakeNan(), gms.yGm);
        return;
    }

    // 坐标映射：crop 坐标 -> 原图坐标
    float inY = MapCoordinate(y1, y2, static_cast<int32_t>(cy), params.cropHeight, geom.imgHMinus1);
    float inX = MapCoordinate(x1, x2, static_cast<int32_t>(cx), params.cropWidth, geom.imgWMinus1);

    // OOB 检查：严格 < 0 和 > size-1，与 TF crop_and_resize_op 一致。
    // 纵深防御：MapCoordinate 可能在极端浮点输入下产生 NaN，NaN 比较为 false 会 bypass OOB 检查，
    // 导致后续 static_cast<int32_t>(floorf(NaN)) 为 UB，因此在此再次拦截。
    if (isnan(inY) || isnan(inX)) {
        FillAllChannels<T_Y>(depth, outBase, MakeNan(), gms.yGm);
        return;
    }
    if (inY < 0.0f || inY > geom.imgHMinus1 || inX < 0.0f || inX > geom.imgWMinus1) {
        FillAllChannels<T_Y>(depth, outBase, params.extrapolationValue, gms.yGm);
        return;
    }

    // 邻居索引 + 权重 + 偏移计算
    InterpGeometry ig = ComputeInterpGeometry(inY, inX, boxIdx, params, geom);

// 逐通道双线性插值（展开 4 次以提升 ILP，常见 depth=3 时全部展开）
#pragma unroll 4
    for (int32_t d = 0; d < depth; d++) {
        float result = BilinearSampleChannel<T_X>(gms.xGm, ig, d);
        gms.yGm[outBase + d] = CastFromFloat<T_Y>(result);
    }
}

// VF 核心计算：Grid-Stride 循环遍历输出位置
// 每个线程处理一个 (box, crop_y, crop_x) 位置，循环 depth 个通道。
// 标量参数通过 UB 传递，避免 VF 标量参数过多导致传递不可靠。
template <typename T_X, typename T_BOXES, typename T_Y, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void OpCropAndResizeSimtKernel(
    IDX_T totalPositions, __ubuf__ int32_t* ub, __gm__ T_X* xGm, __gm__ T_BOXES* boxesGm, __gm__ int32_t* boxIndexGm,
    __gm__ T_Y* yGm)
{
    ScalarParams params = LoadParamsFromUb(ub);
    Geometry geom = ComputeGeometry(params);
    GmPointers<T_X, T_BOXES, T_Y> gms = {xGm, boxesGm, boxIndexGm, yGm};

    IDX_T cropHW = static_cast<IDX_T>(params.cropHeight) * static_cast<IDX_T>(params.cropWidth);
    IDX_T cropWidthT = static_cast<IDX_T>(params.cropWidth);

    // 除零防御：host tiling 不一致（cropHeight=0 或 cropWidth=0）时避免整数除零异常
    if (cropHW == 0 || cropWidthT == 0) {
        return;
    }

    for (IDX_T pos = static_cast<IDX_T>(blockIdx.x) * static_cast<IDX_T>(blockDim.x) + static_cast<IDX_T>(threadIdx.x);
         pos < totalPositions; pos += static_cast<IDX_T>(blockDim.x) * static_cast<IDX_T>(gridDim.x)) {
        IDX_T b = pos / cropHW;
        IDX_T rem = pos - b * cropHW;
        IDX_T cy = rem / cropWidthT;
        IDX_T cx = rem - cy * cropWidthT;
        ProcessOnePosition<T_X, T_BOXES, T_Y, IDX_T>(b, cy, cx, params, geom, gms);
    }
}

// VF 启动函数：标量参数写入 UB，调用 VF kernel
template <typename T_X, typename T_BOXES, typename T_Y, typename IDX_T>
__aicore__ inline void LaunchVf(IDX_T totalPositions, const ScalarParams& params, GmPointers<T_X, T_BOXES, T_Y> gms)
{
    // 标量参数写入 UB
    LocalMemAllocator<AscendC::Hardware::UB> ubAlloc;
    LocalTensor<int32_t> ub = ubAlloc.Alloc<int32_t>(UB_PARAM_COUNT);
    ub.SetValue(UB_OFF_BATCH, params.batch);
    ub.SetValue(UB_OFF_IMG_H, params.imageHeight);
    ub.SetValue(UB_OFF_IMG_W, params.imageWidth);
    ub.SetValue(UB_OFF_DEPTH, params.depth);
    ub.SetValue(UB_OFF_CROP_H, params.cropHeight);
    ub.SetValue(UB_OFF_CROP_W, params.cropWidth);
    ub.SetValue(UB_OFF_EXT_BITS, FloatToBitsAicore(params.extrapolationValue));

    DataSyncBarrier<MemDsbT::UB>();

    __ubuf__ int32_t* ubPtr = (__ubuf__ int32_t*)(ub.GetPhyAddr());

    constexpr uint32_t TIDX = THREADS<IDX_T>;
    asc_vf_call<OpCropAndResizeSimtKernel<T_X, T_BOXES, T_Y, IDX_T>>(dim3(TIDX), totalPositions, ubPtr, gms.xGm,
                                                                     gms.boxesGm, gms.boxIndexGm, gms.yGm);
}

// Process 入口函数：数据量判断分发
// 数据量 <= INT32_MAX 走 uint32_t 高效路径（1024 线程），否则走 uint64_t 路径（512 线程）
template <typename T_X, typename T_BOXES, typename T_Y>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR boxes, GM_ADDR boxIndex, GM_ADDR cropSize, GM_ADDR y,
                               const CropAndResizeTilingData* tilingData)
{
    (void)cropSize;

    if (tilingData == nullptr) {
        return;
    }

    GmPointers<T_X, T_BOXES, T_Y> gms;
    gms.xGm = (__gm__ T_X*)x;
    gms.boxesGm = (__gm__ T_BOXES*)boxes;
    gms.boxIndexGm = (__gm__ int32_t*)boxIndex;
    gms.yGm = (__gm__ T_Y*)y;

    ScalarParams params = MakeScalarParams(tilingData);

    int64_t totalPositions = tilingData->totalPositions;
    if (totalPositions <= static_cast<int64_t>(INT32_MAX)) {
        LaunchVf<T_X, T_BOXES, T_Y, uint32_t>(static_cast<uint32_t>(totalPositions), params, gms);
    } else {
        LaunchVf<T_X, T_BOXES, T_Y, uint64_t>(static_cast<uint64_t>(totalPositions), params, gms);
    }
}
} // namespace NsCropAndResize
#endif // CROP_AND_RESIZE_SIMT_H_
