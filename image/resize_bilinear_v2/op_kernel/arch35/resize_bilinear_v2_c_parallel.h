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
 * \file resize_bilinear_v2_c_parallel.h
 * \brief resize_bilinear_v2_c_parallel
 */
#ifndef RESIZE_BILINEAR_V2_C_PARALLEL_H
#define RESIZE_BILINEAR_V2_C_PARALLEL_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace ResizeBilinearV2 {
using namespace AscendC;
using AscendC::MicroAPI::RegTensor;

constexpr int32_t POS_NW = 0;
constexpr int32_t POS_NE = 1;
constexpr int32_t POS_SW = 2;
constexpr int32_t POS_SE = 3;
constexpr int32_t POS_NUM = 4;

template <typename T_X, typename T_Y>
class ResizeBilinearV2CParallel : public ResizeBilinearV2Base {
public:
    __aicore__ inline ResizeBilinearV2CParallel(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe, const ResizeBilinearV2TilingData* data);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyInSinglePoint(LocalTensor<uint8_t> xTensor, int64_t queIdx, int64_t hPos, int64_t wPos);
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    __aicore__ inline void Compute();
    __aicore__ inline float CalcInputPos(int64_t pos, float scale);

    const ResizeBilinearV2TilingData* tilingData_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;

    int64_t bufferLen_;

    DataCopyPadExtParams<uint8_t> padParams_ = {false, 0, 0, 0};

    constexpr static MicroAPI::CastTrait castTrait0 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::UNKNOWN}; // bf16 --float

    constexpr static MicroAPI::CastTrait castTrait1 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT}; // float---bf16

    int64_t nStrideX_;
    int64_t hwStrideX_;

    int64_t nStrideY_;
    int64_t hwStrideY_;

    int64_t nOffset_;
    int64_t hwOffset_;
    int64_t cOffset_;

    int64_t nLength_;
    int64_t cLength_;

    float delta_[POS_NUM];
};

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2CParallel<T_X, T_Y>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe, const ResizeBilinearV2TilingData* data)
{
    this->BaseInit(x, size, y, pipe);

    tilingData_ = data;

    bufferLen_ = tilingData_->ubNFactor * tilingData_->ubCFactor;
    // bufferLen需要对齐，因为x的buffer是手动分隔4块，每块起始位置必须是block对齐的。
    bufferLen_ = Ops::Base::CeilAlign<int64_t>(bufferLen_, ONE_BLOCK_BYTE / sizeof(T_X));
    this->pipe_->InitBuffer(xQue_, 2, POS_NUM * bufferLen_ * sizeof(T_X));
    this->pipe_->InitBuffer(yQue_, 2, bufferLen_ * sizeof(T_Y));
}

template <typename T_X, typename T_Y>
__aicore__ inline float ResizeBilinearV2CParallel<T_X, T_Y>::CalcInputPos(int64_t pos, float scale)
{
    float posFp;

    if (tilingData_->halfPixelCenters > 0) {
        posFp = ((float)pos + 0.5f) * scale - 0.5f;
    } else {
        posFp = ((float)pos) * scale;
    }

    return posFp;
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2CParallel<T_X, T_Y>::CopyInSinglePoint(
    LocalTensor<uint8_t> xTensor, int64_t queIdx, int64_t hPos, int64_t wPos)
{
    int64_t xOffsetInGM = nOffset_ * nStrideX_ + (hPos * tilingData_->lenSrcW + wPos) * hwStrideX_ + cOffset_;

    DataCopyExtParams gm2ubParams;
    gm2ubParams.blockCount = nLength_;
    gm2ubParams.blockLen = cLength_ * sizeof(T_X);
    gm2ubParams.srcStride = (nStrideX_ - cLength_) * sizeof(T_X);
    gm2ubParams.dstStride = 0;

    DataCopyPad<uint8_t, PaddingMode::Compact>(
        xTensor[queIdx * bufferLen_ * sizeof(T_X)], xGM_[xOffsetInGM * sizeof(T_X)], gm2ubParams, padParams_);
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2CParallel<T_X, T_Y>::CopyIn()
{
    int64_t hPos = hwOffset_ / tilingData_->lenDesW;
    int64_t wPos = hwOffset_ - tilingData_->lenDesW * hPos;

    float hFp = CalcInputPos(hPos, tilingData_->scaleH);
    float hLerp = hFp - Floor(hFp);
    int64_t top = (hFp > 0.0f) ? Floor(hFp) : 0;
    top = min(tilingData_->lenSrcH - 1, top);
    int64_t bot = (hLerp > 0.0f) ? top + 1 : top;
    bot = min(tilingData_->lenSrcH - 1, bot);

    float wFp = CalcInputPos(wPos, tilingData_->scaleW);
    float wLerp = wFp - Floor(wFp);
    int64_t left = (wFp > 0.0f) ? Floor(wFp) : 0;
    left = min(tilingData_->lenSrcW - 1, left);
    int64_t right = (wLerp > 0.0f) ? left + 1 : left;
    right = min(tilingData_->lenSrcW - 1, right);

    delta_[POS_NW] = (1.0f - hLerp) * (1.0f - wLerp);
    delta_[POS_NE] = (1.0f - hLerp) * wLerp;
    delta_[POS_SW] = hLerp * (1.0f - wLerp);
    delta_[POS_SE] = hLerp * wLerp;

    LocalTensor<uint8_t> xTensor = xQue_.AllocTensor<uint8_t>();
    CopyInSinglePoint(xTensor, POS_NW, top, left);
    CopyInSinglePoint(xTensor, POS_NE, top, right);
    CopyInSinglePoint(xTensor, POS_SW, bot, left);
    CopyInSinglePoint(xTensor, POS_SE, bot, right);
    xQue_.EnQue(xTensor);
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2CParallel<T_X, T_Y>::CopyOut()
{
    int64_t yOffsetInGM = nOffset_ * nStrideY_ + hwOffset_ * hwStrideY_ + cOffset_;

    LocalTensor<uint8_t> yTensor = yQue_.DeQue<uint8_t>();

    DataCopyExtParams ub2gmParams;
    ub2gmParams.blockCount = nLength_;
    ub2gmParams.blockLen = cLength_ * sizeof(T_Y);
    ub2gmParams.srcStride = 0;
    ub2gmParams.dstStride = (nStrideY_ - cLength_) * sizeof(T_Y);
    DataCopyPad<uint8_t, PaddingMode::Compact>(yGM_[yOffsetInGM * sizeof(T_Y)], yTensor, ub2gmParams);

    yQue_.FreeTensor(yTensor);
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2CParallel<T_X, T_Y>::Compute()
{
    LocalTensor<T_X> xTensor = xQue_.DeQue<T_X>();
    LocalTensor<T_Y> yTensor = yQue_.AllocTensor<T_Y>();

    __ubuf__ T_X* xAddr = (__ubuf__ T_X*)xTensor.GetPhyAddr();
    __ubuf__ T_X* xAddrNW = xAddr + POS_NW * bufferLen_;
    __ubuf__ T_X* xAddrNE = xAddr + POS_NE * bufferLen_;
    __ubuf__ T_X* xAddrSW = xAddr + POS_SW * bufferLen_;
    __ubuf__ T_X* xAddrSE = xAddr + POS_SE * bufferLen_;
    __ubuf__ T_Y* yAddr = (__ubuf__ T_Y*)yTensor.GetPhyAddr();

    float weightNW = delta_[POS_NW];
    float weightNE = delta_[POS_NE];
    float weightSW = delta_[POS_SW];
    float weightSE = delta_[POS_SE];

    int64_t oneRepeat = Ops::Base::GetVRegSize() / sizeof(float);
    uint32_t totalLen = tilingData_->ubNFactor * tilingData_->ubCFactor;
    int64_t repeatTimes = Ops::Base::CeilDiv<int64_t>(totalLen, oneRepeat);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFp32;
        MicroAPI::MaskReg pregFp16;
        MicroAPI::RegTensor<T_X> regNW;
        MicroAPI::RegTensor<T_X> regNE;
        MicroAPI::RegTensor<T_X> regSW;
        MicroAPI::RegTensor<T_X> regSE;
        MicroAPI::RegTensor<T_X> regTmp;
        MicroAPI::RegTensor<float> regNWFp32;
        MicroAPI::RegTensor<float> regNEFp32;
        MicroAPI::RegTensor<float> regSWFp32;
        MicroAPI::RegTensor<float> regSEFp32;
        MicroAPI::RegTensor<float> regSumTopFp32;
        MicroAPI::RegTensor<float> regSumBotFp32;
        MicroAPI::RegTensor<float> regSumFp32;
        MicroAPI::RegTensor<T_Y> regRst;

        for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
            pregFp32 = MicroAPI::UpdateMask<float>(totalLen);
            MicroAPI::DataCopy<T_X, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regNW, xAddrNW, (int32_t)oneRepeat);
            MicroAPI::DataCopy<T_X, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regNE, xAddrNE, (int32_t)oneRepeat);
            MicroAPI::DataCopy<T_X, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regSW, xAddrSW, (int32_t)oneRepeat);
            MicroAPI::DataCopy<T_X, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regSE, xAddrSE, (int32_t)oneRepeat);

            if constexpr (sizeof(T_X) == sizeof(int16_t)) {
                MicroAPI::UnPack((RegTensor<int32_t>&)regTmp, (RegTensor<int16_t>&)regNW);
                MicroAPI::Cast<float, T_X, castTrait0>(regNWFp32, regTmp, pregFp32);
                MicroAPI::UnPack((RegTensor<int32_t>&)regTmp, (RegTensor<int16_t>&)regNE);
                MicroAPI::Cast<float, T_X, castTrait0>(regNEFp32, regTmp, pregFp32);
                MicroAPI::UnPack((RegTensor<int32_t>&)regTmp, (RegTensor<int16_t>&)regSW);
                MicroAPI::Cast<float, T_X, castTrait0>(regSWFp32, regTmp, pregFp32);
                MicroAPI::UnPack((RegTensor<int32_t>&)regTmp, (RegTensor<int16_t>&)regSE);
                MicroAPI::Cast<float, T_X, castTrait0>(regSEFp32, regTmp, pregFp32);

                MicroAPI::Muls(regNWFp32, regNWFp32, weightNW, pregFp32);
                MicroAPI::Muls(regNEFp32, regNEFp32, weightNE, pregFp32);
                MicroAPI::Muls(regSWFp32, regSWFp32, weightSW, pregFp32);
                MicroAPI::Muls(regSEFp32, regSEFp32, weightSE, pregFp32);
            } else {
                MicroAPI::Muls(regNWFp32, regNW, weightNW, pregFp32);
                MicroAPI::Muls(regNEFp32, regNE, weightNE, pregFp32);
                MicroAPI::Muls(regSWFp32, regSW, weightSW, pregFp32);
                MicroAPI::Muls(regSEFp32, regSE, weightSE, pregFp32);
            }

            MicroAPI::Add(regSumTopFp32, regNWFp32, regNEFp32, pregFp32);
            MicroAPI::Add(regSumBotFp32, regSWFp32, regSEFp32, pregFp32);
            MicroAPI::Add(regSumFp32, regSumTopFp32, regSumBotFp32, pregFp32);

            if constexpr (sizeof(T_Y) == sizeof(int16_t)) {
                MicroAPI::Cast<T_Y, float, castTrait1>(regTmp, regSumFp32, pregFp32);
                MicroAPI::Pack((RegTensor<uint16_t>&)regRst, (RegTensor<uint32_t>&)regTmp);
                MicroAPI::MaskPack(pregFp16, pregFp32);
                MicroAPI::DataCopy<T_Y, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    yAddr, regRst, (int32_t)oneRepeat, pregFp16);
            } else {
                MicroAPI::DataCopy<T_Y, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    yAddr, regSumFp32, (int32_t)oneRepeat, pregFp32);
            }
        }
    }

    yQue_.EnQue(yTensor);
    xQue_.FreeTensor(xTensor);
}

template <typename T_X, typename T_Y>
__aicore__ inline void ResizeBilinearV2CParallel<T_X, T_Y>::Process()
{
    int64_t nNum = Ops::Base::CeilDiv(tilingData_->lenN, tilingData_->nFactor);
    int64_t hwNum = Ops::Base::CeilDiv(tilingData_->lenDesH * tilingData_->lenDesW, tilingData_->hwFactor);
    int64_t cNum = Ops::Base::CeilDiv(tilingData_->lenC, tilingData_->cFactor);
    int64_t nTail = tilingData_->lenN - (nNum - 1) * tilingData_->nFactor;
    int64_t hwTail = tilingData_->lenDesH * tilingData_->lenDesW - (hwNum - 1) * tilingData_->hwFactor;
    int64_t cTail = tilingData_->lenC - (cNum - 1) * tilingData_->cFactor;

    hwStrideX_ = tilingData_->lenC;
    nStrideX_ = hwStrideX_ * tilingData_->lenSrcH * tilingData_->lenSrcW;

    hwStrideY_ = tilingData_->lenC;
    nStrideY_ = hwStrideY_ * tilingData_->lenDesH * tilingData_->lenDesW;

    int64_t block = GetBlockIdx();
    if (block >= tilingData_->realCoreNum) {
        return;
    }

    int64_t cIdx = block % cNum;
    block /= cNum;
    int64_t hwIdx = block % hwNum;
    block /= hwNum;
    int64_t nIdx = block % nNum;

    int64_t nLength4Block = (nIdx < nNum - 1) ? tilingData_->nFactor : nTail;
    int64_t hwLength4Block = (hwIdx < hwNum - 1) ? tilingData_->hwFactor : hwTail;
    int64_t cLength4Block = (cIdx < cNum - 1) ? tilingData_->cFactor : cTail;

    int64_t nOffset4Block = nIdx * tilingData_->nFactor;
    int64_t hwOffset4Block = hwIdx * tilingData_->hwFactor;
    int64_t cOffset4Block = cIdx * tilingData_->cFactor;

    for (int64_t hwLoop = 0; hwLoop < hwLength4Block; hwLoop++) {
        hwOffset_ = hwOffset4Block + hwLoop;

        for (int64_t nLoop = 0; nLoop < nLength4Block; nLoop += tilingData_->ubNFactor) {
            nOffset_ = nOffset4Block + nLoop;
            nLength_ = this->Min(tilingData_->ubNFactor, nLength4Block - nLoop);
            for (int64_t cLoop = 0; cLoop < cLength4Block; cLoop += tilingData_->ubCFactor) {
                cOffset_ = cOffset4Block + cLoop;
                cLength_ = this->Min(tilingData_->ubCFactor, cLength4Block - cLoop);

                CopyIn();
                Compute();
                CopyOut();
            }
        }
    }
}

} // namespace ResizeBilinearV2

#endif // RESIZE_BILINEAR_V2_C_PARALLEL_H
