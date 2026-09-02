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
 * \file grid_sampler_2d_base.h
 * \brief
 */

#ifndef _ASCENDC_GRID_SAMPLER_2D_BASE_H_
#define _ASCENDC_GRID_SAMPLER_2D_BASE_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"

namespace GridSample {

using namespace AscendC;

template <typename T>
class GridSampler2D {
public:
    __aicore__ inline GridSampler2D(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace,
                                const GridSampleTilingData* tilingData, TPipe pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeWeightSub(LocalTensor<float> w1Ub, LocalTensor<float> w2Ub, LocalTensor<float> x1Ub,
                                            LocalTensor<float> x2Ub, LocalTensor<float> y1Ub, LocalTensor<float> y2Ub);
    __aicore__ inline void ParseTilingData(const GridSampleTilingData* tilingData);
    __aicore__ inline void PerLoopCompute(int32_t nIdx, int32_t hwIdx, int32_t calHWElems);
    __aicore__ inline void ClipCoordinates(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
                                           LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb,
                                           LocalTensor<int32_t> coorUb, LocalTensor<uint8_t> weightMaskUb);
    __aicore__ inline void CoordinatesFrameRange(LocalTensor<int32_t> iIntUb, int32_t upBound);
    __aicore__ inline void CoordinatesSelectScalar(LocalTensor<float> iFpUb, LocalTensor<float> oFpUb,
                                                   LocalTensor<uint8_t> maskUb, const float scalarVal,
                                                   const uint32_t calNum);
    __aicore__ inline void CoordinatesGetMaskWithRange(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
                                                       LocalTensor<uint8_t> maskXUb, LocalTensor<uint8_t> maskYUb,
                                                       LocalTensor<uint8_t> maskTmpXUb,
                                                       LocalTensor<uint8_t> maskTmpYUb);
    __aicore__ inline void CoordinatesSelectTensor(LocalTensor<float> src0, LocalTensor<float> src1,
                                                   LocalTensor<float> coorUb, LocalTensor<uint8_t> maskUb);
    __aicore__ inline void BorderClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void ReflectClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb);
    __aicore__ inline void ReflectCoordinatesGeneral(LocalTensor<float> iFpUb, LocalTensor<float> coorSubUb,
                                                     LocalTensor<float> extraFpUb, LocalTensor<float> fmodFpUb,
                                                     LocalTensor<uint8_t> maskUb, LocalTensor<float> tmpFpUb,
                                                     LocalTensor<int32_t> tmpIntUb, const int64_t twiceLow,
                                                     const int64_t twiceHigh);
    __aicore__ inline void MTE2ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
                                       int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb,
                                       LocalTensor<float> xLocal);
    __aicore__ inline void MTE2ForNHWC(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
                                       int32_t loopOffset, int32_t loopElems, LocalTensor<int32_t> coorUb,
                                       LocalTensor<float> xLocal);
    __aicore__ inline void OutTranspose(int32_t channelAlign, LocalTensor<float> xLocal, LocalTensor<float> outValueUb);
    __aicore__ inline void MTE3ForNCHW(int32_t nIdx, int32_t cIdx, int32_t calCElems, int32_t channelAlign,
                                       int32_t hwIdx, int32_t loopOffset, int32_t loopElems, int64_t outBaseOffset,
                                       LocalTensor<float> weightUb, LocalTensor<float> outValueUb, bool isAutomicAdd);
    __aicore__ inline void PointBilinear(int32_t nIdx, int32_t hwIdx, int32_t calHWElems,
                                         LocalTensor<int32_t> coordinatesUb, LocalTensor<float> weightUb,
                                         LocalTensor<uint8_t> weightMaskUb, LocalTensor<float> outValueUb,
                                         bool isAutomicAdd);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> gridQueue_;

    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;
    TBuf<QuePosition::VECCALC> inputXYFPBuf_;
    TBuf<QuePosition::VECCALC> inputXFpBuf_;
    TBuf<QuePosition::VECCALC> inputYFpBuf_;
    TBuf<QuePosition::VECCALC> weightBuf_;
    TBuf<QuePosition::VECCALC> weightTmpBuf_;
    TBuf<QuePosition::VECCALC> weightTmp1Buf_;
    TBuf<QuePosition::VECCALC> weightTmp2Buf_;
    TBuf<QuePosition::VECCALC> weightTmp3Buf_;
    TBuf<QuePosition::VECCALC> coorBuf_;
    TBuf<QuePosition::VECCALC> coorTmpBuf_;
    TBuf<QuePosition::VECCALC> intTmpBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> weightMaskBuf_;
    TBuf<QuePosition::VECCALC> modBuf_;
    TBuf<QuePosition::VECCALC> extraBuf_;
    TBuf<QuePosition::VECCALC> outTmpBuf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<T> gmY_;

    const int64_t TRANSE_REP_STRIDE = 128;
    const int64_t B32_MASK = 64;
    const int64_t CHANNEL_BLOCK = 64;
    const int32_t TRANSE_MUL_WEGHT_LOOPS = 2;

    const int64_t X_UB_SIZE_4_GENERAL = 32768;
    const int64_t GRID_UB_SIZE_4_GENERAL = 4096;
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;
    const int64_t OUT_VAL_NUM = 4096;
    const int64_t X_UB_OFFSET = 512;
    const int64_t CAL_H_W_BLOCK = 512;
    const int64_t MASK_UB_SIZE = CAL_H_W_BLOCK / 8;

    int64_t blockIDX = 0;

    // tiling params
    int64_t coreNum_ = 0;
    int64_t inputN_ = 0;
    int64_t inputC_ = 0;
    int64_t inputH_ = 0;
    int64_t inputW_ = 0;
    int64_t outputH_ = 0;
    int64_t outputW_ = 0;
    int64_t paddingMode_ = 0;
    int64_t interpolationMode_ = 0;
    int64_t alignCorners_ = 0;
    int64_t channelLast_ = 0;
    int64_t needCoreNum_ = 0;

    int64_t gridHW_ = 0;
    int64_t lastLoopHW_ = 0;
    int64_t preNUbLoop_ = 0;
    int64_t totalUbLoop_ = 0;
    int64_t preCoreLoop_ = 0;
    int64_t lastCoreLoop_ = 0;
    int64_t channelLoop_ = 0;
    int64_t perLoopChannel_ = 0;
    int64_t lastLoopChannel_ = 0;

    // const define
    constexpr static int64_t REFLECT_RATIO = 2;
    constexpr static int64_t PADDING_MODE_ZEROS = 0;
    constexpr static int64_t PADDING_MODE_BORDER = 1;
    constexpr static int64_t PADDING_MODE_REFLECTION = 2;
    constexpr static int64_t LAYOUT_NHWC = 1;

    constexpr static uint64_t B32_VECTOR_MASK = 64;
    constexpr static uint64_t B32_BLOCK_STRIDE = 1;
    constexpr static uint64_t B32_REPEAT_STRIDE = 8;
};

} // namespace GridSample

#endif // GIRD_SAMPLER_2D
