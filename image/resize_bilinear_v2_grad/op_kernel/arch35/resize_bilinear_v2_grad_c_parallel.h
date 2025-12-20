/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file resize_bilinear_v2_grad_c_parallel.h
 * \brief resize_bilinear_v2_grad_c_parallel
 */

#ifndef RESIZE_BILINEARV2_GRAD_C_PARALLEL_H
#define RESIZE_BILINEARV2_GRAD_C_PARALLEL_H
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace ResizeBilinearV2Grad {
using namespace AscendC;
using AscendC::MicroAPI::RegTensor;
constexpr int32_t BUFF_NUM = 2;
constexpr int32_t POS_LU = 0;
constexpr int32_t POS_RU = 1;
constexpr int32_t POS_LD = 2;
constexpr int32_t POS_RD = 3;
constexpr int32_t POS_TOTAL = 4;

struct OffsetDefSt {
    // for per core
    int64_t nStart = 0;
    int64_t hwStart = 0;
    int64_t cStart = 0;
    int64_t nLength = 0;
    int64_t hwLength = 0;
    int64_t cLength = 0;
    // loops for inner-core
    int32_t nOffset = 0;
    int32_t nDataLen = 0;
    int32_t cOffset = 0;
    int32_t cDataLen = 0;
    int64_t dstH = 0;
    int64_t dstW = 0;
    // coordinate
    int64_t yUpper = 0;
    int64_t yDown = 0;
    int64_t yRight = 0;
    int64_t yLeft = 0;
};

template <typename T_GRADS, typename T_OUT>
class ResizeBilinearV2GradNc : public ResizeBilinearV2GradBase {
public:
    __aicore__ inline ResizeBilinearV2GradNc(){};
    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, TPipe *pipe, const ResizeBilinearV2GradTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ClearOutputGm();
    __aicore__ inline void ComputeDivideN(OffsetDefSt &stOffset);
    __aicore__ inline void DataCopyPadGm2UbV2(
        TQue<QuePosition::VECIN, BUFF_NUM> &inQueue, GlobalTensor<T_GRADS> &srcGm, OffsetDefSt &stOffset);
    __aicore__ inline void DataCopyPadUb2GmV2(LocalTensor<T_OUT> &ubTensorOut, int64_t ubOffset,
        GlobalTensor<T_OUT> &dstGm, int64_t gmOffset, OffsetDefSt &stOffset);
    __aicore__ inline void Compute4SrcDotWithGrads(uint32_t totalLen);
    __aicore__ inline void ComputeSrcIdx(int64_t &dstIdx, float &srcIdx, float scale);
    __aicore__ inline void ComputeDeltaArgus(int64_t idxHw, OffsetDefSt &stOffset);
    __aicore__ inline void DataMoveUb2Gm(
        TQue<QuePosition::VECOUT, BUFF_NUM> &outQueue, GlobalTensor<T_OUT> &dstGm, OffsetDefSt &stOffset);
    __aicore__ inline void ComputeLengthOffset(OffsetDefSt &stOffset);

private:
    const ResizeBilinearV2GradTilingData *tilingDataPtr_;
    TQue<QuePosition::VECIN, BUFF_NUM> gradsQueue;
    TQue<QuePosition::VECOUT, BUFF_NUM> outQueue;
    GlobalTensor<T_GRADS> gradsGm_;
    GlobalTensor<T_OUT> yGm_;
    DataCopyPadExtParams<T_GRADS> padParams_ = {false, 0, 0, 0};
    int64_t blockIdx_ = 0;

    /* tiling data and other data compute from tiling */
    int64_t lenSrcHwc_ = 0;
    int64_t lenDstHw_ = 0;
    int64_t lenDstHwc_ = 0;
    int64_t ubCFactor_ = 0;
    int64_t dataBuffLen_ = 0;
    float delta_[POS_TOTAL];
    uint32_t oneRepeat_ = platform::GetVRegSize() / sizeof(float);
    constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};  // bf16 --float

    constexpr static AscendC::MicroAPI::CastTrait castTrait1 = {AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};  // float---bf16
};

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::Init(
    GM_ADDR grads, GM_ADDR originalImage, GM_ADDR y, TPipe *pipe, const ResizeBilinearV2GradTilingData *tilingData)
{
    this->BaseInit(grads, y, pipe);
    blockIdx_ = GetBlockIdx();
    tilingDataPtr_ = tilingData;
    lenSrcHwc_ = tilingDataPtr_->lenSrcW * tilingDataPtr_->lenSrcH * tilingDataPtr_->lenC;
    lenDstHw_ = tilingDataPtr_->lenDesH * tilingDataPtr_->lenDesW;
    lenDstHwc_ = lenDstHw_ * tilingDataPtr_->lenC;
    int64_t max_size = this->Max(sizeof(T_GRADS), sizeof(T_OUT));
    ubCFactor_ =
        (ops::CeilDiv(static_cast<uint32_t>(tilingDataPtr_->ubCFactor * max_size), platform::GetUbBlockSize()) *
            platform::GetUbBlockSize()) /
        max_size;
    dataBuffLen_ = ubCFactor_ * tilingDataPtr_->ubNFactor;
    this->pipe_->InitBuffer(gradsQueue, BUFF_NUM, dataBuffLen_ * sizeof(T_GRADS));
    this->pipe_->InitBuffer(outQueue, BUFF_NUM, dataBuffLen_ * sizeof(T_OUT) * POS_TOTAL);
    gradsGm_.SetGlobalBuffer((__gm__ T_GRADS *)grads);
    yGm_.SetGlobalBuffer((__gm__ T_OUT *)y);
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::Process()
{
    OffsetDefSt offsetSt;
    if (blockIdx_ < tilingDataPtr_->initYRealCoreNum) {
        ClearOutputGm();
    }
    if (blockIdx_ < tilingDataPtr_->realCoreNum) {
        ComputeLengthOffset(offsetSt);
        ComputeDivideN(offsetSt);
    }
    return;
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::DataCopyPadGm2UbV2(
    TQue<QuePosition::VECIN, BUFF_NUM> &inQueue, GlobalTensor<T_GRADS> &srcGm, OffsetDefSt &stOffset)
{
    int64_t gmOffset = (stOffset.nOffset * lenDstHw_ + stOffset.dstH * tilingDataPtr_->lenDesW + stOffset.dstW) *
                           tilingDataPtr_->lenC +
                       stOffset.cOffset;
    LocalTensor<T_GRADS> x_ub = inQueue.AllocTensor<T_GRADS>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = stOffset.nDataLen;
    copyParams.blockLen = stOffset.cDataLen * sizeof(T_GRADS);
    copyParams.srcStride = (lenDstHwc_ - stOffset.cDataLen) * sizeof(T_GRADS);
    copyParams.dstStride = 0;
    DataCopyPad(x_ub, srcGm[gmOffset], copyParams, padParams_);
    inQueue.EnQue(x_ub);
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::DataCopyPadUb2GmV2(LocalTensor<T_OUT> &ubTensorOut,
    int64_t ubOffset, GlobalTensor<T_OUT> &dstGm, int64_t gmOffset, OffsetDefSt &stOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = stOffset.nDataLen;
    copyParams.blockLen = stOffset.cDataLen * sizeof(T_OUT);
    copyParams.srcStride = 0;
    copyParams.dstStride = (lenSrcHwc_ - stOffset.cDataLen) * sizeof(T_OUT);
    DataCopyPad(dstGm[gmOffset], ubTensorOut[ubOffset], copyParams);
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::Compute4SrcDotWithGrads(uint32_t totalLen)
{
    LocalTensor<T_OUT> dstUb = outQueue.AllocTensor<T_OUT>();
    LocalTensor<T_GRADS> gradsUbTensor = gradsQueue.DeQue<T_GRADS>();
    auto gradsUbPtr = (__ubuf__ T_GRADS *)gradsUbTensor.GetPhyAddr();
    auto outputUbPtr = (__ubuf__ T_OUT *)dstUb.GetPhyAddr();
    auto outUbPtrLu = outputUbPtr + POS_LU * dataBuffLen_;
    auto outUbPtrRu = outputUbPtr + POS_RU * dataBuffLen_;
    auto outUbPtrLd = outputUbPtr + POS_LD * dataBuffLen_;
    auto outUbPtrRd = outputUbPtr + POS_RD * dataBuffLen_;
    uint16_t repeatTimes = ops::CeilDiv(totalLen, oneRepeat_);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFp16;
        MicroAPI::MaskReg pregFp32;
        MicroAPI::RegTensor<T_GRADS> reg_grads;
        MicroAPI::RegTensor<T_GRADS> reg_grads_i32;
        MicroAPI::RegTensor<float> reg_grads_f32;
        MicroAPI::RegTensor<float> reg_out_lu;
        MicroAPI::RegTensor<float> reg_out_ru;
        MicroAPI::RegTensor<float> reg_out_ld;
        MicroAPI::RegTensor<float> reg_out_rd;
        MicroAPI::RegTensor<T_OUT> regTmpOutput;
        MicroAPI::RegTensor<T_OUT> regOutLu;
        MicroAPI::RegTensor<T_OUT> regOutRu;
        MicroAPI::RegTensor<T_OUT> regOutLd;
        MicroAPI::RegTensor<T_OUT> regOutRd;

        for (uint16_t idx = 0; idx < repeatTimes; idx++) {
            pregFp32 = AscendC::MicroAPI::UpdateMask<float>(totalLen);
            MicroAPI::DataCopy<T_GRADS, MicroAPI::PostLiteral::POST_MODE_UPDATE>(reg_grads, gradsUbPtr, oneRepeat_);
            if constexpr (sizeof(T_GRADS) != sizeof(int32_t)) {
                MicroAPI::UnPack((RegTensor<int32_t> &)reg_grads_i32, (RegTensor<int16_t> &)reg_grads);
                MicroAPI::Cast<float, T_GRADS, castTrait0>(reg_grads_f32, reg_grads_i32, pregFp32);
                MicroAPI::Muls(reg_out_lu, reg_grads_f32, delta_[POS_LU], pregFp32);
                MicroAPI::Muls(reg_out_ru, reg_grads_f32, delta_[POS_RU], pregFp32);
                MicroAPI::Muls(reg_out_ld, reg_grads_f32, delta_[POS_LD], pregFp32);
                MicroAPI::Muls(reg_out_rd, reg_grads_f32, delta_[POS_RD], pregFp32);
            } else {
                MicroAPI::Muls(reg_out_lu, reg_grads, delta_[POS_LU], pregFp32);
                MicroAPI::Muls(reg_out_ru, reg_grads, delta_[POS_RU], pregFp32);
                MicroAPI::Muls(reg_out_ld, reg_grads, delta_[POS_LD], pregFp32);
                MicroAPI::Muls(reg_out_rd, reg_grads, delta_[POS_RD], pregFp32);
            }

            if constexpr (sizeof(T_OUT) == sizeof(int16_t)) {
                MicroAPI::Cast<T_OUT, float, castTrait1>(regTmpOutput, reg_out_lu, pregFp32);
                MicroAPI::Pack((RegTensor<uint16_t> &)regOutLu, (RegTensor<uint32_t> &)regTmpOutput);

                MicroAPI::Cast<T_OUT, float, castTrait1>(regTmpOutput, reg_out_ru, pregFp32);
                MicroAPI::Pack((RegTensor<uint16_t> &)regOutRu, (RegTensor<uint32_t> &)regTmpOutput);

                MicroAPI::Cast<T_OUT, float, castTrait1>(regTmpOutput, reg_out_ld, pregFp32);
                MicroAPI::Pack((RegTensor<uint16_t> &)regOutLd, (RegTensor<uint32_t> &)regTmpOutput);

                MicroAPI::Cast<T_OUT, float, castTrait1>(regTmpOutput, reg_out_rd, pregFp32);
                MicroAPI::Pack((RegTensor<uint16_t> &)regOutRd, (RegTensor<uint32_t> &)regTmpOutput);

                MicroAPI::MaskPack(pregFp16, pregFp32);
                MicroAPI::DataCopy<T_OUT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outUbPtrLu, regOutLu, oneRepeat_, pregFp16);
                MicroAPI::DataCopy<T_OUT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outUbPtrRu, regOutRu, oneRepeat_, pregFp16);
                MicroAPI::DataCopy<T_OUT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outUbPtrLd, regOutLd, oneRepeat_, pregFp16);
                MicroAPI::DataCopy<T_OUT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outUbPtrRd, regOutRd, oneRepeat_, pregFp16);
            } else {
                MicroAPI::DataCopy<T_OUT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outUbPtrLu, reg_out_lu, oneRepeat_, pregFp32);
                MicroAPI::DataCopy<T_OUT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outUbPtrRu, reg_out_ru, oneRepeat_, pregFp32);
                MicroAPI::DataCopy<T_OUT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outUbPtrLd, reg_out_ld, oneRepeat_, pregFp32);
                MicroAPI::DataCopy<T_OUT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    outUbPtrRd, reg_out_rd, oneRepeat_, pregFp32);
            }
        }
    }
    gradsQueue.FreeTensor<T_GRADS>(gradsUbTensor);
    outQueue.EnQue(dstUb);
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::ComputeLengthOffset(OffsetDefSt &stOffset)
{
    int64_t nDivNum = ops::CeilDiv(tilingDataPtr_->lenN, tilingDataPtr_->nFactor);
    int64_t hwDivNum = ops::CeilDiv(lenDstHw_, tilingDataPtr_->hwFactor);
    int64_t cDivNum = ops::CeilDiv(tilingDataPtr_->lenC, tilingDataPtr_->cFactor);
    int64_t tmpBlockIdx = blockIdx_;
    int64_t cIdx = tmpBlockIdx % cDivNum;
    tmpBlockIdx = tmpBlockIdx / cDivNum;
    int64_t hwIdx = tmpBlockIdx % hwDivNum;
    tmpBlockIdx = tmpBlockIdx / hwDivNum;
    int64_t nIdx = tmpBlockIdx % nDivNum;

    stOffset.nStart = nIdx * tilingDataPtr_->nFactor;
    stOffset.hwStart = hwIdx * tilingDataPtr_->hwFactor;
    stOffset.cStart = cIdx * tilingDataPtr_->cFactor;

    int64_t nTailLength = tilingDataPtr_->lenN - (nDivNum - 1) * tilingDataPtr_->nFactor;
    int64_t hwTailLength = lenDstHw_ - (hwDivNum - 1) * tilingDataPtr_->hwFactor;
    int64_t cTailLength = tilingDataPtr_->lenC - (cDivNum - 1) * tilingDataPtr_->cFactor;
    stOffset.nLength = (nIdx < nDivNum - 1) ? tilingDataPtr_->nFactor : nTailLength;
    stOffset.hwLength = (hwIdx < hwDivNum - 1) ? tilingDataPtr_->hwFactor : hwTailLength;
    stOffset.cLength = (cIdx < cDivNum - 1) ? tilingDataPtr_->cFactor : cTailLength;
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::ClearOutputGm()
{
    int64_t yOffset = 0;
    int64_t yLength = tilingDataPtr_->initYSplitBlockFactor;
    int64_t yBaseOffset = tilingDataPtr_->initYSplitBlockFactor * blockIdx_;
    if (blockIdx_ < tilingDataPtr_->initYSplitBlockTailFactor) {
        yLength = yLength + 1;
        yOffset = yBaseOffset + blockIdx_;
    } else {
        yOffset = yBaseOffset + tilingDataPtr_->initYSplitBlockTailFactor;
    }
    GlobalTensor<T_OUT> yInitGm = yGm_[yOffset];
    InitOutput<T_OUT>(yInitGm, yLength, static_cast<T_OUT>(0));
    SyncAll();
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::DataMoveUb2Gm(
    TQue<QuePosition::VECOUT, BUFF_NUM> &outQueue, GlobalTensor<T_OUT> &dstGm, OffsetDefSt &stOffset)
{
    int64_t ubOffset = 0;
    int64_t gmOffset = 0;
    LocalTensor<T_OUT> ubTensorOut = outQueue.DeQue<T_OUT>();

    ubOffset = POS_LU * dataBuffLen_;
    gmOffset = stOffset.nOffset * lenSrcHwc_ +
               (stOffset.yUpper * tilingDataPtr_->lenSrcW + stOffset.yLeft) * tilingDataPtr_->lenC + stOffset.cOffset;
    DataCopyPadUb2GmV2(ubTensorOut, ubOffset, dstGm, gmOffset, stOffset);

    ubOffset = POS_RU * dataBuffLen_;
    gmOffset = stOffset.nOffset * lenSrcHwc_ +
               (stOffset.yUpper * tilingDataPtr_->lenSrcW + stOffset.yRight) * tilingDataPtr_->lenC + stOffset.cOffset;
    DataCopyPadUb2GmV2(ubTensorOut, ubOffset, dstGm, gmOffset, stOffset);

    ubOffset = POS_LD * dataBuffLen_;
    gmOffset = stOffset.nOffset * lenSrcHwc_ +
               (stOffset.yDown * tilingDataPtr_->lenSrcW + stOffset.yLeft) * tilingDataPtr_->lenC + stOffset.cOffset;
    DataCopyPadUb2GmV2(ubTensorOut, ubOffset, dstGm, gmOffset, stOffset);

    ubOffset = POS_RD * dataBuffLen_;
    gmOffset = stOffset.nOffset * lenSrcHwc_ +
               (stOffset.yDown * tilingDataPtr_->lenSrcW + stOffset.yRight) * tilingDataPtr_->lenC + stOffset.cOffset;
    DataCopyPadUb2GmV2(ubTensorOut, ubOffset, dstGm, gmOffset, stOffset);

    outQueue.FreeTensor<T_OUT>(ubTensorOut);
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::ComputeDivideN(OffsetDefSt &stOffset)
{
    int64_t gradsH = 0;
    int64_t gradsW = 0;
    int64_t nStop = 0;

    SetAtomicAdd<T_OUT>();
    for (int64_t idxHw = stOffset.hwStart; idxHw < stOffset.hwStart + stOffset.hwLength; idxHw++) {
        gradsH = idxHw / tilingDataPtr_->lenDesW;
        gradsW = idxHw % tilingDataPtr_->lenDesW;
        ComputeDeltaArgus(idxHw, stOffset);
        nStop = stOffset.nStart + stOffset.nLength;
        for (int64_t idx_n = stOffset.nStart; idx_n < nStop; idx_n = idx_n + tilingDataPtr_->ubNFactor) {
            stOffset.nOffset = idx_n;
            stOffset.nDataLen = this->Min(tilingDataPtr_->ubNFactor, nStop - idx_n);
            for (int64_t idx_c = 0; idx_c < stOffset.cLength; idx_c = idx_c + ubCFactor_) {
                stOffset.cOffset = stOffset.cStart + idx_c;
                stOffset.cDataLen = this->Min(ubCFactor_, stOffset.cLength - idx_c);
                // Step1: copy data from input gm to ub
                DataCopyPadGm2UbV2(gradsQueue, gradsGm_, stOffset);
                // Step2: compute value throw MicroAPI
                Compute4SrcDotWithGrads(dataBuffLen_);
                // Step3: copy data from ub to output gm
                DataMoveUb2Gm(outQueue, yGm_, stOffset);
            }
        }
    }
    SetAtomicNone();
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::ComputeSrcIdx(
    int64_t &dstIdx, float &srcIdx, float scale)
{
    if ((tilingDataPtr_->alignCorners == 0) && (tilingDataPtr_->halfPixelCenters == 1)) {
        srcIdx = (dstIdx + 0.5f) * scale - 0.5f;
    } else {
        srcIdx = dstIdx * scale;
    }
}

template <typename T_GRADS, typename T_OUT>
__aicore__ inline void ResizeBilinearV2GradNc<T_GRADS, T_OUT>::ComputeDeltaArgus(int64_t idxHw, OffsetDefSt &stOffset)
{
    float hFp = 0.0f;
    stOffset.dstH = idxHw / tilingDataPtr_->lenDesW;
    ComputeSrcIdx(stOffset.dstH, hFp, tilingDataPtr_->scaleH);
    float hLerp = hFp - Floor(hFp);
    stOffset.yUpper = (hFp > 0.0f) ? Floor(hFp) : 0;
    stOffset.yUpper = (stOffset.yUpper >= tilingDataPtr_->lenSrcH) ? tilingDataPtr_->lenSrcH - 1 : stOffset.yUpper;
    stOffset.yDown = (hLerp > 0.0f) ? stOffset.yUpper + 1 : stOffset.yUpper;
    stOffset.yDown = (stOffset.yDown >= tilingDataPtr_->lenSrcH) ? tilingDataPtr_->lenSrcH - 1 : stOffset.yDown;

    float wFp = 0.0f;
    stOffset.dstW = idxHw % tilingDataPtr_->lenDesW;
    ComputeSrcIdx(stOffset.dstW, wFp, tilingDataPtr_->scaleW);
    float wLerp = wFp - Floor(wFp);
    stOffset.yLeft = (wFp > 0.0f) ? Floor(wFp) : 0;
    stOffset.yLeft = (stOffset.yLeft >= tilingDataPtr_->lenSrcW) ? tilingDataPtr_->lenSrcW - 1 : stOffset.yLeft;
    stOffset.yRight = (wLerp > 0.0f) ? stOffset.yLeft + 1 : stOffset.yLeft;
    stOffset.yRight = (stOffset.yRight >= tilingDataPtr_->lenSrcW) ? tilingDataPtr_->lenSrcW - 1 : stOffset.yRight;

    delta_[POS_LU] = (1.0f - hLerp) * (1.0f - wLerp);
    delta_[POS_RU] = (1.0f - hLerp) * wLerp;
    delta_[POS_LD] = hLerp * (1.0f - wLerp);
    delta_[POS_RD] = hLerp * wLerp;
}
}  // namespace ResizeBilinearV2Grad
#endif  // RESIZE_BILINEARV2_GRAD_C_PARALLEL_H
