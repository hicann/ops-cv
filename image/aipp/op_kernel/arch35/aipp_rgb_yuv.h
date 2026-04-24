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
 * \file aipp_rgb_yuv.h
 * \brief aipp rgb to yuv kernel
 */

#ifndef AIPP_OP_KERNEL_ARCH35_RGB_YUV_H
#define AIPP_OP_KERNEL_ARCH35_RGB_YUV_H

#include "aipp_base.h"
#include "simt_api/math_functions.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippRgbYuv : public AippBase<T, DataType> {
public:
    __aicore__ inline AippRgbYuv(){};
    __aicore__ inline void Init(const AippTilingData& tilingData);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    uint16_t blockDimX_ = MAX_THREAD_NUM;
};

template <typename T, typename DataType>
__aicore__ inline void AippRgbYuv<T, DataType>::Init(const AippTilingData& tilingData)
{
    this->BaseInit(tilingData);
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aicore__ void SimtComputeRgb2Yuv(
    __gm__ uint8_t* rgbGM, __gm__ T* yuvGM, const AippTilingData tD,
    uint32_t blockIdx, uint32_t blockNum, uint64_t batchSize)
{
    uint32_t outputSizeH = tD.outputSizeH;
    uint32_t outputSizeW = tD.outputSizeW;
    float padValue = tD.paddingParam.padValue;

    for (DataType idx = threadIdx.x + blockIdx * blockDim.x; idx < batchSize;
         idx += blockNum * blockDim.x) {
        CoordPack<DataType> coord;
        ComputeCoordFromIndex(idx, outputSizeH, outputSizeW, coord);

        RgbPack<DataType> dstYuvIdx;
        RgbComputeDstIdx(dstYuvIdx, coord, tD);
        bool isPadding = IsPixelInPadding(coord.hIdx, coord.wIdx, tD);

        if (isPadding) {
            AssignPadValue(yuvGM[dstYuvIdx.r], padValue);
            AssignPadValue(yuvGM[dstYuvIdx.g], padValue);
            AssignPadValue(yuvGM[dstYuvIdx.b], padValue);
        } else {
            RgbPack<DataType> srcRgbIdx;
            RgbComputeSrcIdx(srcRgbIdx, coord, tD, (DataType)tD.srcChannelOffset);

            RgbPack<uint8_t> dstYuv;
            ApplyCscMatrix(dstYuv, rgbGM[srcRgbIdx.r], rgbGM[srcRgbIdx.g], rgbGM[srcRgbIdx.b], tD.cscParam);
            DataConversion(yuvGM[dstYuvIdx.r], dstYuv.r, tD.dtcParam, CHANNEL_NUM_0);
            DataConversion(yuvGM[dstYuvIdx.g], dstYuv.g, tD.dtcParam, CHANNEL_NUM_1);
            DataConversion(yuvGM[dstYuvIdx.b], dstYuv.b, tD.dtcParam, CHANNEL_NUM_2);
        }
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippRgbYuv<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    asc_vf_call<Aipp_Kernel::SimtComputeRgb2Yuv<T, DataType>>(dim3(this->blockDimX_),
        (__gm__ uint8_t*)x, (__gm__ T*)y, this->tilingData_, this->blockIdx_, this->blockNum_, this->totalNum_);
}
} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_RGB_YUV_H