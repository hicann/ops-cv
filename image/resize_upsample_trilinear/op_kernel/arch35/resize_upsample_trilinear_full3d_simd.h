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
 * \file resize_upsample_trilinear_full3d_simd.h
 * \brief Separable SIMD path for aligned A950 full-3D upsampling shapes.
 */

#ifndef RESIZE_UPSAMPLE_TRILINEAR_FULL3D_SIMD_H_
#define RESIZE_UPSAMPLE_TRILINEAR_FULL3D_SIMD_H_

#include <type_traits>
#include "kernel_operator.h"
#include "./resize_upsample_trilinear_tiling_data.h"

namespace ResizeUpsampleTrilinear {
using namespace AscendC;

constexpr uint32_t FULL3D_IN_D = 8;
constexpr uint32_t FULL3D_IN_H = 128;
constexpr uint32_t FULL3D_IN_W = 128;
constexpr uint32_t FULL3D_OUT_D = 256;
constexpr uint32_t FULL3D_OUT_H = 256;
constexpr uint32_t FULL3D_OUT_W = 256;
constexpr uint32_t FULL3D_D_BATCH = 128;
constexpr uint32_t FULL3D_MIN_IN_D = 4;
constexpr uint32_t FULL3D_MIN_IN_H = 64;
constexpr uint32_t FULL3D_MIN_IN_W = 64;
constexpr uint32_t FULL3D_MIN_OUT_D = 128;
constexpr uint32_t FULL3D_MIN_OUT_H = 128;
constexpr uint32_t FULL3D_MIN_OUT_W = 128;
constexpr uint32_t FULL3D_WIDTH_ALIGN = 16;

constexpr uint32_t FULL3D_RAW_ELEMENTS = FULL3D_IN_D * 2 * FULL3D_IN_W;
constexpr uint32_t FULL3D_WIDTH_ROW_ELEMENTS = FULL3D_IN_D * 2 * FULL3D_OUT_W;
constexpr uint32_t FULL3D_BILINEAR_ELEMENTS = FULL3D_IN_D * FULL3D_OUT_W;
constexpr uint32_t FULL3D_OUTPUT_ELEMENTS = FULL3D_D_BATCH * FULL3D_OUT_W;
constexpr uint32_t FULL3D_WIDTH_TABLE_ELEMENTS = 2 * FULL3D_OUT_W;
template <typename T>
constexpr uint32_t Full3dSimdUbBytes()
{
    constexpr uint32_t conversionBytes = std::is_same_v<T, float> ?
                                             0 :
                                             (FULL3D_RAW_ELEMENTS * sizeof(float) + FULL3D_OUTPUT_ELEMENTS * sizeof(T));
    return FULL3D_RAW_ELEMENTS * sizeof(T) + conversionBytes +
           (FULL3D_WIDTH_ROW_ELEMENTS + FULL3D_BILINEAR_ELEMENTS + FULL3D_OUTPUT_ELEMENTS +
            FULL3D_WIDTH_TABLE_ELEMENTS + FULL3D_WIDTH_TABLE_ELEMENTS) *
               sizeof(float);
}

static_assert(Full3dSimdUbBytes<float>() <= 248 * 1024, "FP32 FULL_3D_SIMD buffers exceed A950 UB");
static_assert(Full3dSimdUbBytes<half>() <= 248 * 1024, "FP16 FULL_3D_SIMD buffers exceed A950 UB");
static_assert(Full3dSimdUbBytes<bfloat16_t>() <= 248 * 1024, "BF16 FULL_3D_SIMD buffers exceed A950 UB");

template <typename T, bool FixedShape>
class ResizeUpsampleTrilinearFull3dSimd {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const ResizeUpsampleTrilinearRegBaseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t InD() const;
    __aicore__ inline uint32_t InH() const;
    __aicore__ inline uint32_t InW() const;
    __aicore__ inline uint32_t OutD() const;
    __aicore__ inline uint32_t OutH() const;
    __aicore__ inline uint32_t OutW() const;
    __aicore__ inline void InitWidthTables(const LocalTensor<float>& scratch, const LocalTensor<float>& weight,
                                           const LocalTensor<int32_t>& offset, float scaleW, int32_t alignCorners);
    __aicore__ inline void LoadInputRows(const LocalTensor<T>& raw, uint32_t nc, uint32_t inH0, uint32_t inH1);
    __aicore__ inline void ComputeWidthRows(const LocalTensor<float>& raw, const LocalTensor<float>& widthRows,
                                            const LocalTensor<float>& scratch, const LocalTensor<float>& weight,
                                            const LocalTensor<uint32_t>& offset);

    __aicore__ inline void ProcessNc(uint32_t nc, const LocalTensor<T>& raw, const LocalTensor<float>& rawFp,
                                     const LocalTensor<float>& widthRows, const LocalTensor<float>& bilinear,
                                     const LocalTensor<float>& outputFp, const LocalTensor<T>& outputRaw,
                                     const LocalTensor<float>& weight, const LocalTensor<uint32_t>& offset,
                                     event_t vToMte2, event_t mte2ToV, event_t vToMte3, event_t mte3ToV);

    __aicore__ inline void ComputeHeight(const LocalTensor<float>& widthRows, const LocalTensor<float>& bilinear,
                                         float weightH0, float weightH1);

    __aicore__ inline void ComputeDepthRow(const LocalTensor<float>& bilinear, const LocalTensor<float>& output,
                                           uint32_t outputRow, uint32_t inputD0, uint32_t inputD1, float weightD0,
                                           float weightD1);
    __aicore__ inline void CopyOutputBatch(const LocalTensor<T>& output, uint32_t nc, uint32_t outH,
                                           uint32_t batchStart, uint32_t batchCount, event_t vToMte3, event_t mte3ToV);

    __aicore__ inline void ProcessOutputH(const LocalTensor<float>& widthRows, const LocalTensor<float>& bilinear,
                                          const LocalTensor<float>& outputFp, const LocalTensor<T>& outputRaw,
                                          uint32_t nc, uint32_t outH, float weightH0, float weightH1, event_t vToMte3,
                                          event_t mte3ToV);

private:
    const ResizeUpsampleTrilinearRegBaseTilingData* tilingData_ = nullptr;
    uint32_t blockIdx_ = 0;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> rawFpBuf_;
    TBuf<TPosition::VECCALC> widthRowBuf_;
    TBuf<TPosition::VECCALC> bilinearBuf_;
    TBuf<TPosition::VECCALC> outputFpBuf_;
    TBuf<TPosition::VECCALC> outputRawBuf_;
    TBuf<TPosition::VECCALC> widthOffsetBuf_;
    TBuf<TPosition::VECCALC> widthWeightBuf_;
};

template <typename T, bool FixedShape>
__aicore__ inline uint32_t ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::InD() const
{
    if constexpr (FixedShape) {
        return FULL3D_IN_D;
    }
    return static_cast<uint32_t>(tilingData_->inD);
}

template <typename T, bool FixedShape>
__aicore__ inline uint32_t ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::InH() const
{
    if constexpr (FixedShape) {
        return FULL3D_IN_H;
    }
    return static_cast<uint32_t>(tilingData_->inH);
}

template <typename T, bool FixedShape>
__aicore__ inline uint32_t ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::InW() const
{
    if constexpr (FixedShape) {
        return FULL3D_IN_W;
    }
    return static_cast<uint32_t>(tilingData_->inW);
}

template <typename T, bool FixedShape>
__aicore__ inline uint32_t ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::OutD() const
{
    if constexpr (FixedShape) {
        return FULL3D_OUT_D;
    }
    return static_cast<uint32_t>(tilingData_->outD);
}

template <typename T, bool FixedShape>
__aicore__ inline uint32_t ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::OutH() const
{
    if constexpr (FixedShape) {
        return FULL3D_OUT_H;
    }
    return static_cast<uint32_t>(tilingData_->outH);
}

template <typename T, bool FixedShape>
__aicore__ inline uint32_t ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::OutW() const
{
    if constexpr (FixedShape) {
        return FULL3D_OUT_W;
    }
    return static_cast<uint32_t>(tilingData_->outW);
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::Init(
    GM_ADDR input, GM_ADDR output, const ResizeUpsampleTrilinearRegBaseTilingData* __restrict tilingData)
{
    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
    outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    tilingData_ = tilingData;
    pipe_.InitBuffer(rawBuf_, FULL3D_RAW_ELEMENTS * sizeof(T));
    if constexpr (!std::is_same_v<T, float>) {
        pipe_.InitBuffer(rawFpBuf_, FULL3D_RAW_ELEMENTS * sizeof(float));
        pipe_.InitBuffer(outputRawBuf_, FULL3D_OUTPUT_ELEMENTS * sizeof(T));
    }
    pipe_.InitBuffer(widthRowBuf_, FULL3D_WIDTH_ROW_ELEMENTS * sizeof(float));
    pipe_.InitBuffer(bilinearBuf_, FULL3D_BILINEAR_ELEMENTS * sizeof(float));
    pipe_.InitBuffer(outputFpBuf_, FULL3D_OUTPUT_ELEMENTS * sizeof(float));
    pipe_.InitBuffer(widthOffsetBuf_, FULL3D_WIDTH_TABLE_ELEMENTS * sizeof(uint32_t));
    pipe_.InitBuffer(widthWeightBuf_, FULL3D_WIDTH_TABLE_ELEMENTS * sizeof(float));
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::InitWidthTables(
    const LocalTensor<float>& scratch, const LocalTensor<float>& weight, const LocalTensor<int32_t>& offset,
    float scaleW, int32_t alignCorners)
{
    uint32_t outW = OutW();
    uint32_t inW = InW();
    LocalTensor<float> source = scratch;
    LocalTensor<float> sourceFloor = scratch[outW];
    LocalTensor<float> weightW0 = weight;
    LocalTensor<float> weightW1 = weight[outW];
    LocalTensor<int32_t> offsetW0 = offset;
    LocalTensor<int32_t> offsetW1 = offset[outW];

    Arange(source, 0.0f, 1.0f, outW);
    PipeBarrier<PIPE_V>();
    if (alignCorners == 1) {
        Muls(source, source, scaleW, outW);
        PipeBarrier<PIPE_V>();
    } else {
        Adds(source, source, 0.5f, outW);
        PipeBarrier<PIPE_V>();
        Muls(source, source, scaleW, outW);
        PipeBarrier<PIPE_V>();
        Adds(source, source, -0.5f, outW);
        PipeBarrier<PIPE_V>();
        Maxs(source, source, 0.0f, outW);
        PipeBarrier<PIPE_V>();
    }
    Floor(sourceFloor, source, outW);
    PipeBarrier<PIPE_V>();
    Sub(weightW1, source, sourceFloor, outW);
    PipeBarrier<PIPE_V>();
    Muls(weightW0, weightW1, -1.0f, outW);
    PipeBarrier<PIPE_V>();
    Adds(weightW0, weightW0, 1.0f, outW);
    PipeBarrier<PIPE_V>();
    Cast(offsetW0, sourceFloor, RoundMode::CAST_RINT, outW);
    PipeBarrier<PIPE_V>();
    Adds(offsetW1, offsetW0, 1, outW);
    PipeBarrier<PIPE_V>();
    Mins(offsetW0, offsetW0, static_cast<int32_t>(inW - 1), outW);
    Mins(offsetW1, offsetW1, static_cast<int32_t>(inW - 1), outW);
    PipeBarrier<PIPE_V>();
    Muls(offsetW0, offsetW0, static_cast<int32_t>(sizeof(float)), outW);
    Muls(offsetW1, offsetW1, static_cast<int32_t>(sizeof(float)), outW);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::LoadInputRows(const LocalTensor<T>& raw,
                                                                                       uint32_t nc, uint32_t inH0,
                                                                                       uint32_t inH1)
{
    uint32_t inD = InD();
    uint32_t inW = InW();
    uint32_t inputPlaneElements = InH() * inW;
    uint32_t inputNcOffset = nc * inD * inputPlaneElements;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (inH1 == inH0 + 1U) {
        DataCopyExtParams copyParams{static_cast<uint16_t>(inD), static_cast<uint32_t>(2 * inW * sizeof(T)),
                                     static_cast<int64_t>((inputPlaneElements - 2 * inW) * sizeof(T)), 0, 0};
        DataCopyPad(raw, inputGm_[inputNcOffset + inH0 * inW], copyParams, padParams);
    } else {
        DataCopyExtParams copyParams{static_cast<uint16_t>(inD), static_cast<uint32_t>(inW * sizeof(T)),
                                     static_cast<int64_t>((inputPlaneElements - inW) * sizeof(T)),
                                     static_cast<int64_t>(inW * sizeof(T) / 32), 0};
        uint32_t gmOffset = inputNcOffset + inH0 * inW;
        DataCopyPad(raw, inputGm_[gmOffset], copyParams, padParams);
        DataCopyPad(raw[inW], inputGm_[gmOffset], copyParams, padParams);
    }
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::ComputeWidthRows(
    const LocalTensor<float>& raw, const LocalTensor<float>& widthRows, const LocalTensor<float>& scratch,
    const LocalTensor<float>& weight, const LocalTensor<uint32_t>& offset)
{
    uint32_t inDCount = InD();
    uint32_t inW = InW();
    uint32_t outW = OutW();
    LocalTensor<float> temp01 = scratch[outW];
    LocalTensor<float> temp11 = scratch[3 * outW];
    LocalTensor<float> weightW0 = weight;
    LocalTensor<float> weightW1 = weight[outW];
    LocalTensor<uint32_t> offsetW0 = offset;
    LocalTensor<uint32_t> offsetW1 = offset[outW];

    for (uint32_t inD = 0; inD < inDCount; ++inD) {
        uint32_t rawBase = inD * 2 * inW;
        uint32_t row0Base = inD * outW;
        uint32_t row1Base = (inDCount + inD) * outW;
        Gather(widthRows[row0Base], raw[rawBase], offsetW0, 0, outW);
        Gather(temp01, raw[rawBase], offsetW1, 0, outW);
        Gather(widthRows[row1Base], raw[rawBase + inW], offsetW0, 0, outW);
        Gather(temp11, raw[rawBase + inW], offsetW1, 0, outW);
        PipeBarrier<PIPE_V>();
        Mul(temp01, temp01, weightW1, outW);
        Mul(temp11, temp11, weightW1, outW);
        PipeBarrier<PIPE_V>();
        // PyTorch CUDA evaluates W as fma(w0, x0, round(w1 * x1)).
        // Match that rounding point instead of doing two rounded multiplies
        // followed by an add, which can differ by one FP16/BF16 ULP.
        FusedMulAdd(widthRows[row0Base], weightW0, temp01, outW);
        PipeBarrier<PIPE_V>();
        FusedMulAdd(widthRows[row1Base], weightW0, temp11, outW);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::ComputeHeight(
    const LocalTensor<float>& widthRows, const LocalTensor<float>& bilinear, float weightH0, float weightH1)
{
    uint32_t inDCount = InD();
    uint32_t outW = OutW();
    for (uint32_t inD = 0; inD < inDCount; ++inD) {
        uint32_t outputBase = inD * outW;
        Muls(bilinear[outputBase], widthRows[(inDCount + inD) * outW], weightH1, outW);
        PipeBarrier<PIPE_V>();
        Axpy(bilinear[outputBase], widthRows[inD * outW], weightH0, outW);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::ComputeDepthRow(
    const LocalTensor<float>& bilinear, const LocalTensor<float>& output, uint32_t outputRow, uint32_t inputD0,
    uint32_t inputD1, float weightD0, float weightD1)
{
    uint32_t outW = OutW();
    uint32_t outputBase = outputRow * outW;
    Muls(output[outputBase], bilinear[inputD1 * outW], weightD1, outW);
    PipeBarrier<PIPE_V>();
    Axpy(output[outputBase], bilinear[inputD0 * outW], weightD0, outW);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::CopyOutputBatch(
    const LocalTensor<T>& output, uint32_t nc, uint32_t outH, uint32_t batchStart, uint32_t batchCount, event_t vToMte3,
    event_t mte3ToV)
{
    uint32_t outW = OutW();
    uint32_t outputPlaneElements = OutH() * outW;
    uint32_t outputNcElements = OutD() * outputPlaneElements;
    uint32_t gmOffset = nc * outputNcElements + batchStart * outputPlaneElements + outH * outW;
    DataCopyExtParams copyParams{static_cast<uint16_t>(batchCount), static_cast<uint32_t>(outW * sizeof(T)), 0,
                                 static_cast<int64_t>((outputPlaneElements - outW) * sizeof(T)), 0};
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopyPad(outputGm_[gmOffset], output, copyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::ProcessOutputH(
    const LocalTensor<float>& widthRows, const LocalTensor<float>& bilinear, const LocalTensor<float>& outputFp,
    const LocalTensor<T>& outputRaw, uint32_t nc, uint32_t outH, float weightH0, float weightH1, event_t vToMte3,
    event_t mte3ToV)
{
    ComputeHeight(widthRows, bilinear, weightH0, weightH1);
    uint32_t outDCount = OutD();
    uint32_t inDCount = InD();
    uint32_t outW = OutW();
    for (uint32_t batchStart = 0; batchStart < outDCount; batchStart += FULL3D_D_BATCH) {
        uint32_t batchCount = min(FULL3D_D_BATCH, outDCount - batchStart);
        for (uint32_t localD = 0; localD < batchCount; ++localD) {
            uint32_t outD = batchStart + localD;
            float sourceD = static_cast<float>(outD) * tilingData_->scaleD;
            if (tilingData_->alignCorners != 1) {
                sourceD = (static_cast<float>(outD) + 0.5f) * tilingData_->scaleD - 0.5f;
                sourceD = sourceD < 0.0f ? 0.0f : sourceD;
            }
            uint32_t inputD0 = min(static_cast<uint32_t>(sourceD), inDCount - 1U);
            uint32_t inputD1 = min(inputD0 + 1U, inDCount - 1U);
            float weightD1 = min(max(sourceD - static_cast<float>(inputD0), 0.0f), 1.0f);
            float weightD0 = 1.0f - weightD1;
            ComputeDepthRow(bilinear, outputFp, localD, inputD0, inputD1, weightD0, weightD1);
        }
        if constexpr (!std::is_same_v<T, float>) {
            if constexpr (std::is_same_v<T, half>) {
                Cast(outputRaw, outputFp, RoundMode::CAST_NONE, batchCount * outW);
            } else {
                Cast(outputRaw, outputFp, RoundMode::CAST_RINT, batchCount * outW);
            }
            PipeBarrier<PIPE_V>();
        }
        CopyOutputBatch(outputRaw, nc, outH, batchStart, batchCount, vToMte3, mte3ToV);
    }
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::ProcessNc(
    uint32_t nc, const LocalTensor<T>& raw, const LocalTensor<float>& rawFp, const LocalTensor<float>& widthRows,
    const LocalTensor<float>& bilinear, const LocalTensor<float>& outputFp, const LocalTensor<T>& outputRaw,
    const LocalTensor<float>& weight, const LocalTensor<uint32_t>& offset, event_t vToMte2, event_t mte2ToV,
    event_t vToMte3, event_t mte3ToV)
{
    uint32_t inHCount = InH();
    uint32_t outHCount = OutH();
    uint32_t cachedH0 = inHCount;
    uint32_t cachedH1 = inHCount;
    for (uint32_t outH = 0; outH < outHCount; ++outH) {
        float sourceH = static_cast<float>(outH) * tilingData_->scaleH;
        if (tilingData_->alignCorners != 1) {
            sourceH = (static_cast<float>(outH) + 0.5f) * tilingData_->scaleH - 0.5f;
            sourceH = sourceH < 0.0f ? 0.0f : sourceH;
        }
        uint32_t inH0 = min(static_cast<uint32_t>(sourceH), inHCount - 1U);
        uint32_t inH1 = min(inH0 + 1U, inHCount - 1U);
        float weightH1 = min(max(sourceH - static_cast<float>(inH0), 0.0f), 1.0f);
        float weightH0 = 1.0f - weightH1;
        if (inH0 != cachedH0 || inH1 != cachedH1) {
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            LoadInputRows(raw, nc, inH0, inH1);
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            if constexpr (!std::is_same_v<T, float>) {
                Cast(rawFp, raw, RoundMode::CAST_NONE, InD() * 2 * InW());
                PipeBarrier<PIPE_V>();
            }
            ComputeWidthRows(rawFp, widthRows, outputFp, weight, offset);
            cachedH0 = inH0;
            cachedH1 = inH1;
        }
        ProcessOutputH(widthRows, bilinear, outputFp, outputRaw, nc, outH, weightH0, weightH1, vToMte3, mte3ToV);
    }
}

template <typename T, bool FixedShape>
__aicore__ inline void ResizeUpsampleTrilinearFull3dSimd<T, FixedShape>::Process()
{
    uint32_t totalNc = static_cast<uint32_t>(tilingData_->lenN * tilingData_->lenC);
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockIdx_ >= blockNum || blockNum == 0U) {
        return;
    }

    LocalTensor<T> raw = rawBuf_.Get<T>();
    LocalTensor<float> rawFp;
    if constexpr (std::is_same_v<T, float>) {
        rawFp = raw;
    } else {
        rawFp = rawFpBuf_.Get<float>();
    }
    LocalTensor<float> widthRows = widthRowBuf_.Get<float>();
    LocalTensor<float> bilinear = bilinearBuf_.Get<float>();
    LocalTensor<float> outputFp = outputFpBuf_.Get<float>();
    LocalTensor<T> outputRaw;
    if constexpr (std::is_same_v<T, float>) {
        outputRaw = outputFp;
    } else {
        outputRaw = outputRawBuf_.Get<T>();
    }
    LocalTensor<int32_t> offsetSigned = widthOffsetBuf_.Get<int32_t>();
    LocalTensor<uint32_t> offset = widthOffsetBuf_.Get<uint32_t>();
    LocalTensor<float> weight = widthWeightBuf_.Get<float>();

    InitWidthTables(outputFp, weight, offsetSigned, tilingData_->scaleW, tilingData_->alignCorners);

    event_t vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

    uint32_t ncBase = totalNc / blockNum;
    uint32_t tailBlockNum = totalNc % blockNum;
    uint32_t ncCount = ncBase + (blockIdx_ < tailBlockNum ? 1U : 0U);
    uint32_t ncStart = blockIdx_ * ncBase + min(blockIdx_, tailBlockNum);
    for (uint32_t ncOffset = 0; ncOffset < ncCount; ++ncOffset) {
        ProcessNc(ncStart + ncOffset, raw, rawFp, widthRows, bilinear, outputFp, outputRaw, weight, offset, vToMte2,
                  mte2ToV, vToMte3, mte3ToV);
    }
}
} // namespace ResizeUpsampleTrilinear

#endif // RESIZE_UPSAMPLE_TRILINEAR_FULL3D_SIMD_H_
