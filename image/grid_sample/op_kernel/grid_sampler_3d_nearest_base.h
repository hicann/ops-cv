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
 * \file grid_sampler_3d_nearest_base.h
 * \brief
 */

#ifndef _ASCENDC_GRID_SAMPLER_3D_NEAREST_BASE_H_
#define _ASCENDC_GRID_SAMPLER_3D_NEAREST_BASE_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "grid_sampler_3d_common.h"

namespace GridSample {

using namespace AscendC;

template <typename T>
class GridSampler3DNearest {
public:
    __aicore__ inline GridSampler3DNearest(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gird, GM_ADDR y, GM_ADDR workspace,
                                const GridSampleTilingData* tilingData, TPipe pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PerLoopCompute(ProcessParam processParam);
    __aicore__ inline void Clip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb);
    __aicore__ inline void ZeroClip(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb, LocalTensor<float> iZFpUb);
    __aicore__ inline void MTE2ForNCHW(int32_t nIdx, PointParam pointNearestParam, LocalTensor<int32_t> coorUb,
                                       LocalTensor<T> xLocal);
    __aicore__ inline void MTE2ForNHWC(int32_t nIdx, PointParam pointNearestParam, LocalTensor<int32_t> coorUb,
                                       LocalTensor<T> xLocal);
    __aicore__ inline void OutTransposeFp16(int32_t channelAlign, LocalTensor<T> xLocal, LocalTensor<T> outValueUb);

    __aicore__ inline void MTE3ForNCHWFp16(ProcessParam processParam, PointParam pointNearestParam,
                                           LocalTensor<float> weightUb, LocalTensor<float> outValueUb);

    __aicore__ inline void PointNearestEachChannel(ProcessParam processParam, LocalTensor<uint64_t> maskUbTmp,
                                                   PointParam pointNearestParam, LocalTensor<T> xLocal);

    __aicore__ inline void MTE3ForNCHWFp32(ProcessParam processParam, PointParam pointNearestParam,
                                           LocalTensor<float> weightUb, LocalTensor<float> outValueU);

    __aicore__ inline void PointNearest(ProcessParam processParam);

    __aicore__ inline void CalculateGrid(ProcessParam processParam, LocalTensor<float> inputXFpLocal,
                                         LocalTensor<float> inputYFpLocal, LocalTensor<float> inputZFpLocal);

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xBuf_;

    TBuf<QuePosition::VECCALC> gridFp32Buf_;
    TBuf<QuePosition::VECCALC> inputXIntBuf_;
    TBuf<QuePosition::VECCALC> inputYIntBuf_;
    TBuf<QuePosition::VECCALC> inputZIntBuf_;
    TBuf<QuePosition::VECCALC> weightBuf_;
    TBuf<QuePosition::VECCALC> coorBuf_;
    TBuf<QuePosition::VECCALC> outValueBuf_;
    TBuf<QuePosition::VECCALC> bufferMaskXBuf_;
    TBuf<QuePosition::VECCALC> bufferMaskYBuf_;
    TBuf<QuePosition::VECCALC> bufferMaskZBuf_;

    TBuf<QuePosition::VECCALC> gridFp16Buf_;
    TBuf<QuePosition::VECCALC> yFp16Buf_;
    TBuf<QuePosition::VECCALC> outValueFp16Buf_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmGrid_;
    GlobalTensor<float> gmWorkspace_;
    GlobalTensor<T> gmY_;

    LocalTensor<int32_t> coordinatesLocal;
    LocalTensor<float> weightLocal;
    LocalTensor<float> outValueLocal;
    LocalTensor<uint8_t> weightMaskUb;

    const int64_t X_UB_SIZE_4_GENERAL = 32768;   // 32KB
    const int64_t X_UB_SIZE_4_FP16 = 16384;      // 16KB
    const int64_t GRID_UB_SIZE_4_GENERAL = 6144; //  6KB
    const int64_t GRID_UB_SIZE_4_FP16 = 3072;    //  3KB
    const int64_t XYZ_UB_SIZE_4_GENERAL = 4096;  //  4KB
    const int64_t Y_UB_SIZE_4_GENERAL = 2048;    //  2KB

    int64_t blockIDX = 0;
    uint64_t rsvdCnt = 0;
    uint32_t mask = 192;
    uint16_t repeatTime = CAL_D_H_W_BLOCK * 3 / 192;

    GridSampleCommonParam commonParam{};
    IndexBuffer indexBuffer{};
};

} // namespace GridSample

#endif // GIRD_SAMPLER_3D_NEAREST
