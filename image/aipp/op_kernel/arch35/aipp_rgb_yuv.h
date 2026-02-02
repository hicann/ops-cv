/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AippTilingData& tilingData);
    __aicore__ inline void Process(GM_ADDR tiling);

private:
    uint16_t blockDimX_ = MAX_TGHREAD_NUM;
    uint32_t mBlockIdx_;
};

template <typename T, typename DataType>
__aicore__ inline void AippRgbYuv<T, DataType>::Init(GM_ADDR x, GM_ADDR y, const AippTilingData& tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    this->BaseInit(x, y, tilingData);
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_TGHREAD_NUM) __aicore__ void SimtComputeRgb2YuvNHWC(__gm__ uint8_t* inputGM,
                                                                                __gm__ T* outputGM, GM_ADDR tiling,
                                                                                uint32_t mBlockIdx_,uint32_t blockNum,
                                                                                DataType batchSize, DataType
                                                                                perChannelOutDataSize) {
    GET_TILING_DATA_PTR_WITH_STRUCT(AippTilingData, tD, tiling);
    for (DataType idx = Simt::GetThreadIdx() + block_idx * Simt::GetThreadNum(); idx < batchSize;
       idx += block_num * Simt::GetThreadNum()) {
        DataType nIdx = idx / perChannelOutDataSize;
        DataType newIdx = idx - nIdx * perChannelOutDataSize;
        DataType hIdx = newIdx / tD->cropSizeW;
        DataType wIdx = newIdx - hIdx * tD->cropSizeW;

        DataType dstYIdx = nIdx * (perChannelOutDataSize * YUV_CHANNEL_NUM) +
                            hIdx * tD->cropSizeW * YUV_CHANNEL_NUM  + wIdx * YUV_CHANNEL_NUM;
        DataType dstUIdx = dstYIdx + DIGIT_ONE;
        DataType dstVIdx = dstYIdx + DIGIT_TWO;

        DataType srcHIdx = hIdx;
        DataType srcWIdx = wIdx;
        DataType rgbBaseIdx = nIdx * tD->inputSizeH * tD->inputSizeW * RGB_CHANNEL_NUM +
                                srcHIdx * tD->inputSizeW * RGB_CHANNEL_NUM + srcWIdx * RGB_CHANNEL_NUM;
        DataType r = rgbBaseIdx;
        DataType g = rgbBaseIdx + DIGIT_ONE;
        DataType b = rgbBaseIdx + DIGIT_TWO;

        uint8_t srcR = inputGM[r];
        uint8_t srcG = inputGM[g];
        uint8_t srcB = inputGM[b];

        uint8_t dstY = 0;
        uint8_t dstU = 0;
        uint8_t dstV = 0;

        int16_t rSinged = static_cast<int16_t>(srcR);
        int16_t gSinged = static_cast<int16_t>(srcG);
        int16_t bSinged = static_cast<int16_t>(srcB);

        dstY = static_cast<uint8_t>(roundf(static_cast<float>(tD->cscMatrix00 * rSinged +
            tD->cscMatrix01 * gSinged +tD->cscMatrix02 * bSinged) / 256)) + tD->outBias0;
        dstU = static_cast<uint8_t>(roundf(static_cast<float>(tD->cscMatrix10 * rSinged +
            tD->cscMatrix11 * gSinged +tD->cscMatrix12 * bSinged) / 256)) + tD->outBias1;
        dstV = static_cast<uint8_t>(roundf(static_cast<float>(tD->cscMatrix20 * rSinged +
            tD->cscMatrix21 * gSinged +tD->cscMatrix22 * bSinged) / 256)) + tD->outBias2;

        if constexpr (sizeof(T) == DIGIT_TWO) {
            outputGM[dstYIdx] = (static_cast<T>(dstY - tD->dtcPixelMeanChn0) - static_cast<T>(tD->dtcPixelMinChn0)) *
                static_cast<T>(tD->dtcPixelVarReciChn0);
            outputGM[dstUIdx] = (static_cast<T>(dstU - tD->dtcPixelMeanChn1) - static_cast<T>(tD->dtcPixelMinChn1)) *
                static_cast<T>(tD->dtcPixelVarReciChn1);
            outputGM[dstVIdx] = (static_cast<T>(dstV - tD->dtcPixelMeanChn2) - static_cast<T>(tD->dtcPixelMinChn2)) *
                static_cast<T>(tD->dtcPixelVarReciChn2);
        } else {
            outputGM[dstYIdx] = dstY;
            outputGM[dstUIdx] = dstU;
            outputGM[dstVIdx] = dstV;
        }
    }
}

template <typename T, typename DataType>
__simt_vf__ LAUNCH_BOUND(MAX_TGHREAD_NUM) __aicore__ void SimtComputeRgb2YuvNCHW(__gm__ uint8_t* inputGM,
                                                                                __gm__ T* outputGM, GM_ADDR tiling,
                                                                                uint32_t mBlockIdx_,
                                                                                uint32_t blockNum,
                                                                                DataType batchSize,
                                                                                DataType perChannelOutDataSize) {
    GET_TILING_DATA_PTR_WITH_STRUCT(AippTilingData, tD, tiling);
    for (DataType idx = Simt::GetThreadIdx() + block_idx * Simt::GetThreadNum(); idx < batchSize;
       idx += block_num * Simt::GetThreadNum()) {
        DataType nIdx = idx / perChannelOutDataSize;
        DataType newIdx = idx - nIdx * perChannelOutDataSize;
        DataType hIdx = newIdx / tD->cropSizeW;
        DataType wIdx = newIdx - hIdx * tD->cropSizeW;

        DataType dstYIdx = nIdx * (perChannelOutDataSize * YUV_CHANNEL_NUM) + hIdx * tD->cropSizeW + wIdx;
        DataType dstUIdx = dstYIdx + perChannelOutDataSize;
        DataType dstVIdx = dstYIdx + DIGIT_TWO * perChannelOutDataSize;

        DataType srcHIdx = hIdx;
        DataType srcWIdx = wIdx;
        DataType rgbBaseIdx = nIdx * tD->inputSizeH * tD->inputSizeW * RGB_CHANNEL_NUM +
                                srcHIdx * tD->inputSizeW + srcWIdx;
        DataType r = rgbBaseIdx;
        DataType g = rgbBaseIdx + tD->inputSizeW * tD->inputSizeH;
        DataType b = rgbBaseIdx + DIGIT_TWO * tD->inputSizeW * tD->inputSizeH;

        uint8_t srcR = inputGM[r];
        uint8_t srcG = inputGM[g];
        uint8_t srcB = inputGM[b];

        uint8_t dstY = 0;
        uint8_t dstU = 0;
        uint8_t dstV = 0;

        int16_t rSinged = static_cast<int16_t>(srcR);
        int16_t gSinged = static_cast<int16_t>(srcG);
        int16_t bSinged = static_cast<int16_t>(srcB);

        dstY = static_cast<uint8_t>(roundf(static_cast<float>(tD->cscMatrix00 * rSinged +
            tD->cscMatrix01 * gSinged +tD->cscMatrix02 * bSinged) / 256)) + tD->outBias0;
        dstU = static_cast<uint8_t>(roundf(static_cast<float>(tD->cscMatrix10 * rSinged +
            tD->cscMatrix11 * gSinged +tD->cscMatrix12 * bSinged) / 256)) + tD->outBias1;
        dstV = static_cast<uint8_t>(roundf(static_cast<float>(tD->cscMatrix20 * rSinged +
            tD->cscMatrix21 * gSinged +tD->cscMatrix22 * bSinged) / 256)) + tD->outBias2;

        if constexpr (sizeof(T) == DIGIT_TWO) {
            outputGM[dstYIdx] = (static_cast<T>(dstY - tD->dtcPixelMeanChn0) - static_cast<T>(tD->dtcPixelMinChn0)) *
                static_cast<T>(tD->dtcPixelVarReciChn0);
            outputGM[dstUIdx] = (static_cast<T>(dstU - tD->dtcPixelMeanChn1) - static_cast<T>(tD->dtcPixelMinChn1)) *
                static_cast<T>(tD->dtcPixelVarReciChn1);
            outputGM[dstVIdx] = (static_cast<T>(dstV - tD->dtcPixelMeanChn2) - static_cast<T>(tD->dtcPixelMinChn2)) *
                static_cast<T>(tD->dtcPixelVarReciChn2);
        } else {
            outputGM[dstYIdx] = dstY;
            outputGM[dstUIdx] = dstU;
            outputGM[dstVIdx] = dstV;
        }
    }
}

template <typename T, typename DataType>
__aicore__ inline void AippRgbYuv<T, DataType>::Process(GM_ADDR tiling) {
    DataType perChannelOutDataSize = this->cropSizeH_ * this->cropSizeW_;
    uint32_t blockNum = GetBlockNum();

    if (this->inputFormat_ == NHWC_FORMAT_INDEX) {
        Simt::VF_CALL<SimtComputeRgb2YuvNHWC<T, DataType>>(
            Simt::Dim3(blockDimX_), (__gm__ uint8_t*)(this->xGm_.GetPhyAddr()),
            (__gm__ T*)(this->yGm_.GetPhyAddr()), tiling, mBlockIdx_,
            blockNum, this->totalNum_, perChannelOutDataSize);
    } else {
        Simt::VF_CALL<SimtComputeRgb2YuvNCHW<T, DataType>>(
            Simt::Dim3(blockDimX_), (__gm__ uint8_t*)(this->xGm_.GetPhyAddr()),
            (__gm__ T*)(this->yGm_.GetPhyAddr()), tiling, mBlockIdx_,
            blockNum, this->totalNum_, perChannelOutDataSize);
    }
}
}  // namespace Aipp
#endif  // AIPP_OP_KERNEL_ARCH35_RGB_YUV_H