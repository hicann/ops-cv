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
 * \file aipp_base.h
 * \brief aipp kernel base
 */
#ifndef AIPP_OP_KERNEL_ARCH35_BASE_H
#define AIPP_OP_KERNEL_ARCH35_BASE_H

#include "kernel_operator.h"
#include "aipp_struct.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"

#define CLIP3(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

namespace Aipp_Kernel {
using namespace AscendC;
constexpr uint32_t MAX_THREAD_NUM = 256;
constexpr uint32_t CSC_MATRIX_SCALE = 512;
constexpr uint32_t MAX_UINT8 = 255;
constexpr uint32_t CSC_IDENTITY_SCALE = 256;
constexpr uint8_t CHANNEL_THREE = 3;
constexpr uint8_t CHANNEL_COUNT_3 = 3;
constexpr uint8_t CHANNEL_COUNT_4 = 4;
constexpr uint8_t YUV_PER_DEAL_NUM = 4;
constexpr uint8_t YUV_DEAL_NUM_0 = 0;
constexpr uint8_t YUV_DEAL_NUM_1 = 1;
constexpr uint8_t YUV_DEAL_NUM_2 = 2;
constexpr uint8_t YUV_DEAL_NUM_3 = 3;
constexpr uint8_t CHANNEL_NUM_0 = 0;
constexpr uint8_t CHANNEL_NUM_1 = 1;
constexpr uint8_t CHANNEL_NUM_2 = 2;
constexpr uint8_t DIGIT_2 = 2;
constexpr uint8_t DIGIT_3 = 3;
constexpr uint8_t NCHW_FORMAT_INDEX = 1;
constexpr uint8_t NHWC_FORMAT_INDEX = 2;
constexpr uint8_t AIPP_RGB_PASS_THROUGH = 1;
constexpr uint8_t AIPP_YUV_PASS_THROUGH = 2;
constexpr uint8_t AIPP_RGB_TO_YUV = 3;
constexpr uint8_t AIPP_RGB_TO_GRAY = 4;
constexpr uint8_t AIPP_YUV_TO_RGB = 5;
constexpr uint8_t AIPP_YUV_TO_GRAY = 6;
constexpr uint8_t IMAGE_FORMAT_YUV420SP_U8 = 1;
constexpr uint8_t IMAGE_FORMAT_XRGB8888_U8 = 2;
constexpr uint8_t IMAGE_FORMAT_RGB888_U8 = 5;
constexpr uint8_t IMAGE_FORMAT_YUV400_U8 = 10;
constexpr int16_t FP16_MAN_HIDE_BIT = 0x0400;
constexpr int16_t FP16_MAX_EXP = 0x001F;
constexpr uint32_t FP32_EXP_BIAS = 127U;
constexpr uint32_t FP16_EXP_BIAS = 15U;
constexpr uint32_t FP16_MAN_MASK = 0x03FFU;
constexpr uint32_t FP32_MAN_LEN = 23U;
constexpr uint32_t FP16_MAN_LEN = 10U;
constexpr uint32_t FP32_MAX = 0x7FFFFFU;
constexpr uint32_t FP32_SIGN_INDEX = 31U;

class AippDynamicParam {
public:
    __aicore__ inline AippDynamicParam(tagAippDynamicParaHeader* header,
        const __gm__ uint8_t* gmParams)
        : header_(header), gmParams_(gmParams) {}

    __aicore__ inline tagAippDynamicParaHeader& Header() { return *header_; }
    __aicore__ inline const tagAippDynamicParaHeader& Header() const { return *header_; }

    // Get GM pointer for passing into SIMT function
    __aicore__ inline const __gm__ uint8_t* GetGMParamsPtr() const { return gmParams_; }

    __aicore__ inline int8_t BatchNum() const { return header_->batchNum; }

private:
    tagAippDynamicParaHeader* header_;
    const __gm__ uint8_t* gmParams_;
};

union TypeUnion {
    float fVal;
    uint32_t uVal;
};

#define FP16_EXTRAC_SIGN(x)            (((x) >> 15U) & 1U)
#define FP16_EXTRAC_EXP(x)             (((x) >> 10U) & FP16_MAX_EXP)
#define FP16_EXTRAC_MAN(x)             ((((x) >> 0U) & 0x3FFU) |          \
                                       ((((((x) >> 10U) & 0x1FU) > 0U) ? 1U : 0U) * 0x400U))
#define FP32_CONSTRUCTOR(s, e, m)        (((s) << FP32_SIGN_INDEX) |      \
                                          ((e) << FP32_MAN_LEN) |         \
                                          ((m) & FP32_MAX))

__simt_callee__ __attribute__((always_inline)) inline void ExtractFP16(
    const uint16_t val, uint16_t *const s, int16_t *const e, uint16_t *const m)
{
    // 1.Extract
    *s = FP16_EXTRAC_SIGN(val);
    *e = static_cast<int16_t>(FP16_EXTRAC_EXP(val));
    *m = FP16_EXTRAC_MAN(val);

    // Denormal
    if ((*e) == 0) {
        *e = 1;
    }
}

__simt_callee__ __attribute__((always_inline)) inline float Fp16ToFloat(const uint16_t val)
{
    uint16_t hfSign;
    uint16_t hfMan;
    int16_t hfExp;
    ExtractFP16(val, &hfSign, &hfExp, &hfMan);

    while ((hfMan != 0U) && ((hfMan & FP16_MAN_HIDE_BIT) == 0U)) {
        hfMan <<= 1U;
        hfExp--;
    }

    uint32_t eRet;
    uint32_t mRet;
    if (hfMan == 0U) {
        eRet = 0U;
        mRet = 0U;
    } else {
        eRet = static_cast<uint32_t>(hfExp + static_cast<int16_t>(FP32_EXP_BIAS - FP16_EXP_BIAS));
        mRet = static_cast<uint32_t>(hfMan & FP16_MAN_MASK);
        mRet = mRet << (FP32_MAN_LEN - FP16_MAN_LEN);
    }

    const uint32_t sRet = hfSign;
    TypeUnion u;
    u.uVal = FP32_CONSTRUCTOR(sRet, eRet, mRet);
    const auto ret = u.fVal;
    return ret;
}

template <class ByteCountT>
__inline__ __attribute__((always_inline)) __aicore__ void InitHeaderParamData(
    const __gm__ uint8_t *p_tilingdata, uint8_t *tilingdata, ByteCountT all_bytes)
{
    const uint64_t copy_bytes = static_cast<uint64_t>(all_bytes);
#if defined(ASCENDC_CPU_DEBUG)
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)p_tilingdata;
    uint32_t *dst = (uint32_t *)tilingdata;
    for (uint64_t i = 0; i < (copy_bytes + 3) / 4; i++) {
        *(dst + i) = *(src + i);
    }
#elif defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__) || \
      defined(__GET_CODE_CHANNEL__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
    copy_data_align64(tilingdata, (__gm__ uint8_t *)p_tilingdata, copy_bytes);
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    uint32_t len_burst = (copy_bytes + 31) / 32;
    copy_gm_to_ubuf_align_v2(tilingdata_in_ub, (__gm__ uint8_t *)p_tilingdata,
        0, 1, len_burst * 32, 0, 0, false, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64(tilingdata, tilingdata_in_ub, copy_bytes);
#endif
    pipe_barrier(PIPE_ALL);
}

__inline__ __attribute__((always_inline)) __aicore__ void InitDynamicAippHeader(
    const __gm__ uint8_t *p_tilingdata, tagAippDynamicParaHeader *tilingdata, bool dynamic_flag)
{
    if (!dynamic_flag) {
        return;
    }
    constexpr uint64_t header_size = sizeof(tagAippDynamicParaHeader);
    InitHeaderParamData(p_tilingdata, reinterpret_cast<uint8_t*>(tilingdata), header_size);
}

#define GET_PARAM_DATA_WITH_STRUCT_TBUF(tiling_data, tiling_arg, dynamic_flag)                            \
    tagAippDynamicParaHeader tiling_data##_header;                                          \
    InitDynamicAippHeader(tiling_arg, &tiling_data##_header, dynamic_flag);                               \
    AippDynamicParam tiling_data(&tiling_data##_header, tiling_arg)

__aicore__ __attribute__((always_inline)) inline void SwapMatrixVal(int16_t& val1, int16_t& val2)
{
    int16_t temp = val1;
    val1 = val2;
    val2 = temp;
}

template <typename T, typename DataType>
class AippBase {
public:
    __aicore__ inline AippBase(){};
    __aicore__ inline void BaseInit(const AippTilingData& tilingData,
        const tagAippDynamicParaHeader& tilingParamHeader,
        const __gm__ uint8_t* gmParams,
        uint8_t dynamicTilingKey);

public:
    AippTilingData tilingData_ = {};
    tagAippDynamicParaHeader tilingParamHeader_ = {};
    const __gm__ uint8_t* gmParams_ = nullptr;

    uint64_t totalNum_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 0;
    uint8_t dynamicTilingKey_ = 0;
};

template <typename T, typename DataType>
__aicore__ inline void AippBase<T, DataType>::BaseInit(const AippTilingData& tilingData,
    const tagAippDynamicParaHeader& tilingParamHeader,
    const __gm__ uint8_t* gmParams,
    uint8_t dynamicTilingKey)
{
    tilingData_ = tilingData;
    tilingParamHeader_ = tilingParamHeader;
    gmParams_ = gmParams;
    dynamicTilingKey_ = dynamicTilingKey;

#if defined(ASCENDC_CPU_DEBUG)
    blockIdx_ = static_cast<uint32_t>(::get_block_idx());
    blockNum_ = static_cast<uint32_t>(::get_block_num());
#else
    blockNum_ = gridDim.x;
    blockIdx_ = blockIdx.x;
#endif
    totalNum_ = tilingData_.batchNum * tilingData_.outputSizeH * tilingData_.outputSizeW;
}

__aicore__ __attribute__((always_inline)) inline void SetDynamicGrayFlag(const AippTilingData& tD, bool& isGray)
{
    if (tD.imageFormat == IMAGE_FORMAT_YUV400_U8 && !static_cast<bool>(tD.cscParam.cscSwitch)) {
        isGray = true;
        return;
    }

    bool anyMatrix1NotZero = (tD.cscParam.cscMatrix10 != 0) || (tD.cscParam.cscMatrix11 != 0) ||
                             (tD.cscParam.cscMatrix12 != 0);
    bool anyMatrix2NotZero = (tD.cscParam.cscMatrix20 != 0) || (tD.cscParam.cscMatrix21 != 0) ||
                             (tD.cscParam.cscMatrix22 != 0);
    if (anyMatrix1NotZero || anyMatrix2NotZero) {
        return;
    }
    if (tD.imageFormat == IMAGE_FORMAT_RGB888_U8 || tD.imageFormat == IMAGE_FORMAT_XRGB8888_U8) {
        if ((tD.cscParam.outBias0 == 0) && (tD.cscParam.outBias1 == 0) && (tD.cscParam.outBias2 == 0)) {
            isGray = true;
        }
    }
    if (tD.imageFormat == IMAGE_FORMAT_YUV420SP_U8 && tD.cscParam.cscMatrix01 == 0 && tD.cscParam.cscMatrix02 == 0) {
        if ((tD.cscParam.inBias0 == 0) && (tD.cscParam.inBias1 == 0) && (tD.cscParam.inBias2 == 0)) {
            isGray = true;
        }
    }
}

__aicore__ __attribute__((always_inline)) inline void SwapDynamicChannel(AippTilingData& tD)
{
    if ((tD.imageFormat == IMAGE_FORMAT_RGB888_U8 || tD.imageFormat == IMAGE_FORMAT_XRGB8888_U8) &&
        tD.cscParam.rbuvSwapSwitch == 1) {
        SwapMatrixVal(tD.cscParam.cscMatrix00, tD.cscParam.cscMatrix02);
        SwapMatrixVal(tD.cscParam.cscMatrix10, tD.cscParam.cscMatrix12);
        SwapMatrixVal(tD.cscParam.cscMatrix20, tD.cscParam.cscMatrix22);
    }
    if (tD.imageFormat == IMAGE_FORMAT_YUV420SP_U8 && tD.cscParam.rbuvSwapSwitch == 1) {
        SwapMatrixVal(tD.cscParam.cscMatrix01, tD.cscParam.cscMatrix02);
        SwapMatrixVal(tD.cscParam.cscMatrix11, tD.cscParam.cscMatrix12);
        SwapMatrixVal(tD.cscParam.cscMatrix21, tD.cscParam.cscMatrix22);
    }
}

__aicore__ __attribute__((always_inline)) inline void ResetDynamicTilingKey(AippTilingData& tD,
                                                                            uint8_t& dynamicTilingKey)
{
    bool isGray = false;
    SetDynamicGrayFlag(tD, isGray);
    SwapDynamicChannel(tD);
    const bool cscSwitch = static_cast<bool>(tD.cscParam.cscSwitch);
    const bool isRgbFormat = (tD.imageFormat == IMAGE_FORMAT_RGB888_U8 || tD.imageFormat == IMAGE_FORMAT_XRGB8888_U8);
    const bool isYuvFormat = (tD.imageFormat == IMAGE_FORMAT_YUV420SP_U8 || tD.imageFormat == IMAGE_FORMAT_YUV400_U8);

    if (isGray) {
        if (isRgbFormat) {
            dynamicTilingKey = AIPP_RGB_TO_GRAY;
            return;
        } else if (isYuvFormat) {
            dynamicTilingKey = AIPP_YUV_TO_GRAY;
            return;
        }
    } else if (isRgbFormat) {
        dynamicTilingKey = cscSwitch ? AIPP_RGB_TO_YUV : AIPP_RGB_PASS_THROUGH;
        return;
    } else if (isYuvFormat) {
        dynamicTilingKey = cscSwitch ? AIPP_YUV_TO_RGB : AIPP_YUV_PASS_THROUGH;
        return;
    }
}

__aicore__ __attribute__((always_inline)) inline void resetRealPara(AippTilingData& tD,
                                                                    const tagAippDynamicParaHeader& tP)
{
    if (tP.inputFormat == IMAGE_FORMAT_YUV420SP_U8) {
        tD.cscParam.outBias0 = 0;
        tD.cscParam.outBias1 = 0;
        tD.cscParam.outBias2 = 0;
        tD.cscParam.inBias0 = static_cast<int16_t>(tP.cscInputBiasR0);
        tD.cscParam.inBias1 = static_cast<int16_t>(tP.cscInputBiasR1);
        tD.cscParam.inBias2 = static_cast<int16_t>(tP.cscInputBiasR2);
    } else {
        tD.cscParam.inBias0 = 0;
        tD.cscParam.inBias1 = 0;
        tD.cscParam.inBias2 = 0;
        tD.cscParam.outBias0 = static_cast<int16_t>(tP.cscOutputBiasR0);
        tD.cscParam.outBias1 = static_cast<int16_t>(tP.cscOutputBiasR1);
        tD.cscParam.outBias2 = static_cast<int16_t>(tP.cscOutputBiasR2);
    }
    if (tP.inputFormat == IMAGE_FORMAT_XRGB8888_U8 && (tD.cscParam.axSwapSwitch == 1)) {
        tD.srcChannelOffset = 1;
    }
    if (tP.cscSwitch == 0) {
        tD.cscParam.cscMatrix00 = CSC_IDENTITY_SCALE;
        tD.cscParam.cscMatrix01 = 0;
        tD.cscParam.cscMatrix02 = 0;
        tD.cscParam.cscMatrix10 = 0;
        tD.cscParam.cscMatrix11 = CSC_IDENTITY_SCALE;
        tD.cscParam.cscMatrix12 = 0;
        tD.cscParam.cscMatrix20 = 0;
        tD.cscParam.cscMatrix21 = 0;
        tD.cscParam.cscMatrix22 = CSC_IDENTITY_SCALE;
    }
}

__aicore__ __attribute__((always_inline)) inline void UpdateRealPara(AippTilingData& tD,
    const tagAippDynamicParaHeader& tP, uint8_t dynamicTilingKey)
{
    tD.imageFormat = tP.inputFormat;
    tD.channelNum = tP.inputFormat == IMAGE_FORMAT_YUV420SP_U8 ? CHANNEL_COUNT_3 :
                    tP.inputFormat == IMAGE_FORMAT_XRGB8888_U8 ? CHANNEL_COUNT_4 :
                    tP.inputFormat == IMAGE_FORMAT_RGB888_U8   ? CHANNEL_COUNT_3 :
                    tP.inputFormat == IMAGE_FORMAT_YUV400_U8   ? 1 : 0;
    tD.cscParam.cscSwitch = static_cast<int16_t>(tP.cscSwitch);
    tD.cscParam.rbuvSwapSwitch = static_cast<int16_t>(tP.rbuvSwapSwitch);
    tD.cscParam.axSwapSwitch = static_cast<int16_t>(tP.axSwapSwitch);
    tD.batchNum = static_cast<uint32_t>(tP.batchNum);
    tD.inputSizeW = static_cast<uint32_t>(tP.srcImageSizeW);
    tD.inputSizeH = static_cast<uint32_t>(tP.srcImageSizeH);
    tD.cscParam.cscMatrix00 = tP.cscMatrixR0C0;
    tD.cscParam.cscMatrix01 = tP.cscMatrixR0C1;
    tD.cscParam.cscMatrix02 = tP.cscMatrixR0C2;
    tD.cscParam.cscMatrix10 = tP.cscMatrixR1C0;
    tD.cscParam.cscMatrix11 = tP.cscMatrixR1C1;
    tD.cscParam.cscMatrix12 = tP.cscMatrixR1C2;
    tD.cscParam.cscMatrix20 = tP.cscMatrixR2C0;
    tD.cscParam.cscMatrix21 = tP.cscMatrixR2C1;
    tD.cscParam.cscMatrix22 = tP.cscMatrixR2C2;
    resetRealPara(tD, tP);
}

template <typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void UpdateDynamicBatchPara(
    const CoordPack<DataType>& coord, AippTilingData& tD,
    const __gm__ uint8_t* gmParams)
{
    constexpr uint64_t header_size = sizeof(tagAippDynamicParaHeader);
    const __gm__ kAippDynamicBatchPara* gmBatchArr =
        reinterpret_cast<const __gm__ kAippDynamicBatchPara*>(gmParams + header_size);
    const __gm__ kAippDynamicBatchPara& para = gmBatchArr[coord.nIdx];
    tD.cropParam.cropSwitch = static_cast<int16_t>(para.cropSwitch);
    tD.paddingParam.paddingSwitch = static_cast<int32_t>(para.paddingSwitch);
    tD.cropParam.cropStartPosW = static_cast<uint32_t>(para.cropStartPosW);
    tD.cropParam.cropStartPosH = static_cast<uint32_t>(para.cropStartPosH);
    tD.cropParam.cropSizeW = tD.cropParam.cropSwitch == 0 ? tD.inputSizeW : static_cast<uint32_t>(para.cropSizeW);
    tD.cropParam.cropSizeH = tD.cropParam.cropSwitch == 0 ? tD.inputSizeH : static_cast<uint32_t>(para.cropSizeH);
    tD.paddingParam.topPaddingSize = para.paddingSizeTop;
    tD.paddingParam.bottomPaddingSize = para.paddingSizeBottom;
    tD.paddingParam.leftPaddingSize = para.paddingSizeLeft;
    tD.paddingParam.rightPaddingSize = para.paddingSizeRight;
    tD.dtcParam.dtcPixelMeanChn0 = para.dtcPixelMeanChn0;
    tD.dtcParam.dtcPixelMeanChn1 = para.dtcPixelMeanChn1;
    tD.dtcParam.dtcPixelMeanChn2 = para.dtcPixelMeanChn2;
    tD.dtcParam.dtcPixelMeanChn3 = para.dtcPixelMeanChn3;
    tD.dtcParam.dtcPixelMinChn0 = Fp16ToFloat(para.dtcPixelMinChn0);
    tD.dtcParam.dtcPixelMinChn1 = Fp16ToFloat(para.dtcPixelMinChn1);
    tD.dtcParam.dtcPixelMinChn2 = Fp16ToFloat(para.dtcPixelMinChn2);
    tD.dtcParam.dtcPixelMinChn3 = Fp16ToFloat(para.dtcPixelMinChn3);
    tD.dtcParam.dtcPixelVarReciChn0 = Fp16ToFloat(para.dtcPixelVarReciChn0);
    tD.dtcParam.dtcPixelVarReciChn1 = Fp16ToFloat(para.dtcPixelVarReciChn1);
    tD.dtcParam.dtcPixelVarReciChn2 = Fp16ToFloat(para.dtcPixelVarReciChn2);
    tD.dtcParam.dtcPixelVarReciChn3 = Fp16ToFloat(para.dtcPixelVarReciChn3);
}

template <typename T>
__simt_callee__ __attribute__((always_inline)) inline void DataConversion(
    T& dst, uint8_t src, const DtcParam& dtcParam, int32_t channelIndex)
{
    if constexpr (sizeof(T) == DIGIT_2) {
        if (channelIndex == 0) {
            dst = (static_cast<T>(src - dtcParam.dtcPixelMeanChn0) - static_cast<T>(dtcParam.dtcPixelMinChn0)) *
                  static_cast<T>(dtcParam.dtcPixelVarReciChn0);
        } else if (channelIndex == 1) {
            dst = (static_cast<T>(src - dtcParam.dtcPixelMeanChn1) - static_cast<T>(dtcParam.dtcPixelMinChn1)) *
                  static_cast<T>(dtcParam.dtcPixelVarReciChn1);
        } else if (channelIndex == 2) {
            dst = (static_cast<T>(src - dtcParam.dtcPixelMeanChn2) - static_cast<T>(dtcParam.dtcPixelMinChn2)) *
                  static_cast<T>(dtcParam.dtcPixelVarReciChn2);
        } else if (channelIndex == 3) {
            dst = (static_cast<T>(src - dtcParam.dtcPixelMeanChn3) - static_cast<T>(dtcParam.dtcPixelMinChn3)) *
                  static_cast<T>(dtcParam.dtcPixelVarReciChn3);
        }
    } else {
        dst = src;
    }
}

template <typename T>
__simt_callee__ __attribute__((always_inline)) inline void AssignPadValue(T& dst, float padValue)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        dst = static_cast<T>(CLIP3(padValue, 0.0f, 255.0f));
    } else {
        dst = static_cast<T>(padValue);
    }
}

template <typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void RgbComputeDstIdx(
    RgbPack<DataType> &dstIdx, const CoordPack<DataType>& coord, const AippTilingData& tD)
{
    if (tD.outputFormat == NCHW_FORMAT_INDEX) {
        dstIdx.r = coord.nIdx * tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE +
                   coord.hIdx * tD.outputSizeW  + coord.wIdx;
        dstIdx.g = dstIdx.r + tD.outputSizeH * tD.outputSizeW;
        dstIdx.b = dstIdx.g + tD.outputSizeH * tD.outputSizeW;
    } else {
        dstIdx.r = coord.nIdx * tD.outputSizeH * tD.outputSizeW * CHANNEL_THREE +
                   coord.hIdx * tD.outputSizeW * CHANNEL_THREE + coord.wIdx * CHANNEL_THREE;
        dstIdx.g = dstIdx.r + 1;
        dstIdx.b = dstIdx.g + 1;
    }
}

template <typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void RgbComputeSrcIdx(
    RgbPack<DataType> &srcIdx, const CoordPack<DataType>& coord, const AippTilingData& tD,
    const DataType offset = 0)
{
    srcIdx.r = coord.nIdx * tD.inputSizeH * tD.inputSizeW * tD.channelNum +
        (tD.cropParam.cropStartPosH + coord.hIdx - tD.paddingParam.topPaddingSize) * tD.inputSizeW * tD.channelNum +
        (tD.cropParam.cropStartPosW + coord.wIdx - tD.paddingParam.leftPaddingSize) * tD.channelNum + offset;
    srcIdx.g = srcIdx.r + 1;
    srcIdx.b = srcIdx.g + 1;
}

template <typename DataType>
__simt_callee__ __attribute__((always_inline)) inline bool IsPixelInPadding(
    DataType hIdx, DataType wIdx, const AippTilingData& tD)
{
    if (tD.paddingParam.paddingSwitch == 0) {
        return false;
    }
    int32_t leftPaddingSize = tD.paddingParam.leftPaddingSize;
    int32_t topPaddingSize = tD.paddingParam.topPaddingSize;
    uint32_t cropSizeH = tD.cropParam.cropSizeH;
    uint32_t cropSizeW = tD.cropParam.cropSizeW;
    return (hIdx < topPaddingSize) || (hIdx >= topPaddingSize + cropSizeH) ||
           (wIdx < leftPaddingSize) || (wIdx >= leftPaddingSize + cropSizeW);
}

__simt_callee__ __attribute__((always_inline)) inline bool IsPixelInPaddingForYuv(
    uint32_t pixelH, uint32_t pixelW, const AippTilingData& tD,
    bool allEvenPadding, bool blockAllInPadding)
{
    if (tD.paddingParam.paddingSwitch == 0) {
        return false;
    }
    if (allEvenPadding) {
        return blockAllInPadding;
    }
    return (pixelH < (uint32_t)tD.paddingParam.topPaddingSize) ||
           (pixelH >= (uint32_t)tD.paddingParam.topPaddingSize + tD.cropParam.cropSizeH) ||
           (pixelW < (uint32_t)tD.paddingParam.leftPaddingSize) ||
           (pixelW >= (uint32_t)tD.paddingParam.leftPaddingSize + tD.cropParam.cropSizeW);
}

__simt_callee__ __attribute__((always_inline)) inline void ApplyCscMatrix(
    RgbPack<uint8_t>& dst, uint8_t ch0, uint8_t ch1, uint8_t ch2, const CscParam& cscParam)
{
    auto t0 = static_cast<int16_t>(ch0) - cscParam.inBias0;
    auto t1 = static_cast<int16_t>(ch1) - cscParam.inBias1;
    auto t2 = static_cast<int16_t>(ch2) - cscParam.inBias2;
    auto r = static_cast<int16_t>(roundf(static_cast<float>(
        cscParam.cscMatrix00 * t0 * 2 + cscParam.cscMatrix01 * t1 * 2 +
        cscParam.cscMatrix02 * t2 * 2 + 1) / CSC_MATRIX_SCALE));
    auto g = static_cast<int16_t>(roundf(static_cast<float>(
        cscParam.cscMatrix10 * t0 * 2 + cscParam.cscMatrix11 * t1 * 2 +
        cscParam.cscMatrix12 * t2 * 2 + 1) / CSC_MATRIX_SCALE));
    auto b = static_cast<int16_t>(roundf(static_cast<float>(
        cscParam.cscMatrix20 * t0 * 2 + cscParam.cscMatrix21 * t1 * 2 +
        cscParam.cscMatrix22 * t2 * 2 + 1) / CSC_MATRIX_SCALE));
    dst.r = static_cast<uint8_t>(CLIP3(r + cscParam.outBias0, 0, MAX_UINT8));
    dst.g = static_cast<uint8_t>(CLIP3(g + cscParam.outBias1, 0, MAX_UINT8));
    dst.b = static_cast<uint8_t>(CLIP3(b + cscParam.outBias2, 0, MAX_UINT8));
}

template <typename DataType>
__simt_callee__ __attribute__((always_inline)) inline void ComputeCoordFromIndex(
    DataType idx, uint32_t outputSizeH, uint32_t outputSizeW,
    CoordPack<DataType>& coord)
{
    coord.nIdx = idx / (outputSizeH * outputSizeW);
    DataType newIdx = idx - coord.nIdx * outputSizeH * outputSizeW;
    coord.hIdx = newIdx / outputSizeW;
    coord.wIdx = newIdx - coord.hIdx * outputSizeW;
}

} // namespace Aipp_Kernel
#endif // AIPP_OP_KERNEL_ARCH35_BASE_H