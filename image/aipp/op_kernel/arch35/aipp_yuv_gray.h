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
 * \file aipp_yuv_gray.h
 * \brief aipp yuv to gray
 *
 * Supports YUV420SP_U8 -> GRAY and YUV400_U8 -> GRAY
 * YUV420SP: YUV420SP format with size H*W*1.5, extracts Y component from interleaved storage
 * YUV400: Single-channel grayscale format with size H*W, direct access
 */

#ifndef AIPP_OP_KERNEL_ARCH35_YUV_GRAY_H
#define AIPP_OP_KERNEL_ARCH35_YUV_GRAY_H

#include "aipp_base.h"
#include "simt_api/math_functions.h"

namespace Aipp_Kernel {
using namespace AscendC;

template <typename T, typename DataType>
class AippYuvGray : public AippBase<T, DataType> {
public:
    __aicore__ inline AippYuvGray(){};
    __aicore__ inline void Init(const AippTilingData& tilingData);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y);

private:
    uint16_t blockDimX_ = MAX_THREAD_NUM;
};

template <typename T, typename DataType>
__aicore__ inline void AippYuvGray<T, DataType>::Init(const AippTilingData& tilingData)
{
    this->BaseInit(tilingData);
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aicore__ void SimtComputeYuvGray(
    __gm__ uint8_t* yuvGM, __gm__ T* outputGM, const AippTilingData tD,
    uint32_t blockIdx, uint32_t blockNum, uint64_t batchSize)
{
    uint32_t outputSizeH = tD.outputSizeH;
    uint32_t outputSizeW = tD.outputSizeW;
    float padValue = tD.paddingParam.padValue;
    bool isYuv400 = (tD.imageFormat == YUV400_U8_FORMAT);

    for (DataType idx = threadIdx.x + blockIdx * blockDim.x; idx < batchSize;
         idx += blockNum * blockDim.x) {
        CoordPack<DataType> coord;
        RgbPack<DataType> dstGrayIdx;
        ComputeCoordFromIndex(idx, outputSizeH, outputSizeW, coord);
        RgbComputeDstIdx(dstGrayIdx, coord, tD);
        bool isPadding = IsPixelInPadding(coord.hIdx, coord.wIdx, tD);

        if (isPadding) {
            AssignPadValue(outputGM[dstGrayIdx.r], padValue);
            AssignPadValue(outputGM[dstGrayIdx.g], padValue);
            AssignPadValue(outputGM[dstGrayIdx.b], padValue);
        } else {
            DataType srcIdx;
            if (isYuv400) {
                srcIdx = coord.nIdx * tD.inputSizeH * tD.inputSizeW +
                    (tD.cropParam.cropStartPosH + coord.hIdx - tD.paddingParam.topPaddingSize) * tD.inputSizeW +
                    tD.cropParam.cropStartPosW + coord.wIdx - tD.paddingParam.leftPaddingSize;
            } else {
                srcIdx = coord.nIdx * tD.inputSizeH * tD.inputSizeW * DIGIT_3 / DIGIT_2 +
                    (tD.cropParam.cropStartPosH + coord.hIdx - tD.paddingParam.topPaddingSize) * tD.inputSizeW +
                    tD.cropParam.cropStartPosW + coord.wIdx - tD.paddingParam.leftPaddingSize;
            }
            RgbPack<uint8_t> result;
            ApplyCscMatrix(result, yuvGM[srcIdx], 0, 0, tD.cscParam);
            DataConversion(outputGM[dstGrayIdx.r], result.r, tD.dtcParam, CHANNEL_NUM_0);
            DataConversion(outputGM[dstGrayIdx.g], 0, tD.dtcParam, CHANNEL_NUM_1);
            DataConversion(outputGM[dstGrayIdx.b], 0, tD.dtcParam, CHANNEL_NUM_2);
        }
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippYuvGray<T, DataType>::Process(GM_ADDR x, GM_ADDR y)
{
    uint64_t batchSize = static_cast<uint64_t>(this->tilingData_.batchNum) *
        this->tilingData_.outputSizeH * this->tilingData_.outputSizeW;

    asc_vf_call<Aipp_Kernel::SimtComputeYuvGray<T, DataType>>(dim3(this->blockDimX_),
        (__gm__ uint8_t*)x, (__gm__ T*)y, this->tilingData_, this->blockIdx_, this->blockNum_, batchSize);
}

} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_YUV_GRAY_H