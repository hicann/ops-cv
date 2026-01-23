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
 * \file resize_bilinear_v2_nc.h
 * \brief resize_bilinear_v2_nc
 */

#ifndef RESIZE_BILINEARV2_NC_H
#define RESIZE_BILINEARV2_NC_H
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace ResizeBilinearV2 {
using namespace AscendC;
using AscendC::MicroAPI::RegTensor;
constexpr int32_t BUFF_NUM = 2;
constexpr int32_t POS_LU = 0;
constexpr int32_t POS_RU = 1;
constexpr int32_t POS_LD = 2;
constexpr int32_t POS_RD = 3;
constexpr int32_t POS_TOTAL = 4;

using OffsetDefSt = struct {
    int64_t nStart;
    int64_t hwStart;
    int64_t cStart;
    int64_t nLength;
    int64_t hwLength;
    int64_t cLength;
};

const int32_t ONE_BLOCK_UB = Ops::Base::GetUbBlockSize();

template <typename Tin, typename Tout>
class ResizeBilinearV2Nc : public ResizeBilinearV2Base {
public:
    __aicore__ inline ResizeBilinearV2Nc(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe, const ResizeBilinearV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DivideNhwc();
    __aicore__ inline void ComputeSrcIdx(int64_t& dstIdx, float& srcIdx, float scale);
    __aicore__ inline void ComputeDeltaArgus(
        int64_t idxHw, int64_t& srcUpper, int64_t& srcDown, int64_t& srcLeft, int64_t& srcRight);
    __aicore__ inline void ComputeDstValueWith4SrcDot(LocalTensor<Tout>& dstUb, uint32_t totalLen);
    __aicore__ inline void DataCopyPadUb2Gm(
        TQue<QuePosition::VECOUT, BUFF_NUM>& outQueue, GlobalTensor<Tout>& dstGm, int64_t gmOffset, int32_t length);
    __aicore__ inline void DataCopyPadGm2Ub(
        TQue<QuePosition::VECIN, BUFF_NUM>& inQueue, GlobalTensor<Tin>& srcGm, int64_t gmOffset, int32_t length);
    __aicore__ inline void ComputLengthOffset(OffsetDefSt& stOffset);

private:
    const ResizeBilinearV2TilingData* tilingDataPtr_;
    TQue<QuePosition::VECIN, BUFF_NUM> inQueue0;
    TQue<QuePosition::VECIN, BUFF_NUM> inQueue1;
    TQue<QuePosition::VECIN, BUFF_NUM> inQueue2;
    TQue<QuePosition::VECIN, BUFF_NUM> inQueue3;
    TQue<QuePosition::VECOUT, BUFF_NUM> outQueue;
    GlobalTensor<Tin> xGm_;
    GlobalTensor<Tout> yGm_;
    int64_t blockIdx_ = 0;
    DataCopyPadExtParams<Tin> padParams_ = {false, 0, 0, 0};

    /* tiling data and other data compute from tiling */
    int64_t realCoreNum_ = 0;
    int64_t alignCorners_ = 0;
    int64_t halfPixelCenters_ = 0;
    int64_t lenN_ = 0;
    int64_t lenC_ = 0;
    int64_t lenSrcH_ = 0;
    int64_t lenSrcW_ = 0;
    int64_t lenSrcHw_ = 0;
    int64_t lenDstH_ = 0;
    int64_t lenDstW_ = 0;
    int64_t nFactor_ = 0;
    int64_t hFactor_ = 0;
    int64_t wFactor_ = 0;
    int64_t cFactor_ = 0;
    int64_t hwFactor_ = 0;
    int64_t ubNFactor_ = 0;
    int64_t ubHFactor_ = 0;
    int64_t ubWFactor_ = 0;
    int64_t ubCFactor_ = 0;
    int64_t dataUbSize_ = 0;
    float scaleW_ = 0.0;
    float scaleH_ = 0.0;
    int64_t lenDstHw_ = 0;
    float delta_[POS_TOTAL];
    uint32_t oneRepeat_ = Ops::Base::GetVRegSize() / sizeof(float);
    constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN}; // bf16 --float

    constexpr static AscendC::MicroAPI::CastTrait castTrait1 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT}; // float---bf16
};

template <typename Tin, typename Tout>
__aicore__ inline void ResizeBilinearV2Nc<Tin, Tout>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe, const ResizeBilinearV2TilingData* tilingData)
{
    this->BaseInit(x, size, y, pipe);
    blockIdx_ = GetBlockIdx();
    tilingDataPtr_ = tilingData;
    realCoreNum_ = tilingDataPtr_->realCoreNum;
    alignCorners_ = tilingDataPtr_->alignCorners;
    halfPixelCenters_ = tilingDataPtr_->halfPixelCenters;
    lenN_ = tilingDataPtr_->lenN;
    lenC_ = tilingDataPtr_->lenC;
    lenSrcH_ = tilingDataPtr_->lenSrcH;
    lenSrcW_ = tilingDataPtr_->lenSrcW;
    lenSrcHw_ = lenSrcW_ * lenSrcH_;
    lenDstH_ = tilingDataPtr_->lenDesH;
    lenDstW_ = tilingDataPtr_->lenDesW;
    nFactor_ = tilingDataPtr_->nFactor;
    hFactor_ = tilingDataPtr_->hFactor;
    wFactor_ = tilingDataPtr_->wFactor;
    cFactor_ = tilingDataPtr_->cFactor;
    hwFactor_ = tilingDataPtr_->hwFactor;
    ubNFactor_ = tilingDataPtr_->ubNFactor;
    ubHFactor_ = tilingDataPtr_->ubHFactor;
    ubWFactor_ = tilingDataPtr_->ubWFactor;
    ubCFactor_ = tilingDataPtr_->ubCFactor;
    dataUbSize_ = (sizeof(Tin) > sizeof(Tout)) ? ubCFactor_ * sizeof(Tin) : ubCFactor_ * sizeof(Tout);
    scaleW_ = static_cast<float>(tilingDataPtr_->scaleW);
    scaleH_ = static_cast<float>(tilingDataPtr_->scaleH);
    lenDstHw_ = lenDstH_ * lenDstW_;

    this->pipe_->InitBuffer(inQueue0, BUFF_NUM, dataUbSize_);
    this->pipe_->InitBuffer(inQueue1, BUFF_NUM, dataUbSize_);
    this->pipe_->InitBuffer(inQueue2, BUFF_NUM, dataUbSize_);
    this->pipe_->InitBuffer(inQueue3, BUFF_NUM, dataUbSize_);
    this->pipe_->InitBuffer(outQueue, BUFF_NUM, dataUbSize_);
    xGm_.SetGlobalBuffer((__gm__ Tin*)x);
    yGm_.SetGlobalBuffer((__gm__ Tout*)y);
}

template <typename Tin, typename Tout>
__aicore__ inline void ResizeBilinearV2Nc<Tin, Tout>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    DivideNhwc();
    return;
}

template <typename Tin, typename Tout>
__aicore__ inline void ResizeBilinearV2Nc<Tin, Tout>::DataCopyPadGm2Ub(
    TQue<QuePosition::VECIN, BUFF_NUM>& inQueue, GlobalTensor<Tin>& srcGm, int64_t gmOffset, int32_t length)
{
    LocalTensor<Tin> xUb = inQueue.AllocTensor<Tin>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = length * sizeof(Tin);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xUb, srcGm[gmOffset], copyParams, padParams_);
    inQueue.EnQue(xUb);
}

template <typename Tin, typename Tout>
__aicore__ inline void ResizeBilinearV2Nc<Tin, Tout>::DataCopyPadUb2Gm(
    TQue<QuePosition::VECOUT, BUFF_NUM>& outQueue, GlobalTensor<Tout>& dstGm, int64_t gmOffset, int32_t length)
{
    LocalTensor<Tout> ubTensorOut = outQueue.DeQue<Tout>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = length * sizeof(Tout);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(dstGm[gmOffset], ubTensorOut, copyParams);
    outQueue.FreeTensor<Tout>(ubTensorOut);
}

template <typename Tin, typename Tout>
__aicore__ inline void ResizeBilinearV2Nc<Tin, Tout>::ComputeDstValueWith4SrcDot(
    LocalTensor<Tout>& dstUb, uint32_t totalLen)
{
    LocalTensor<Tin> srcUbLu = inQueue0.DeQue<Tin>();
    LocalTensor<Tin> srcUbRu = inQueue1.DeQue<Tin>();
    LocalTensor<Tin> srcUbLd = inQueue2.DeQue<Tin>();
    LocalTensor<Tin> srcUbRd = inQueue3.DeQue<Tin>();

    uint16_t repeatTimes = Ops::Base::CeilDiv(totalLen, oneRepeat_);
    auto dstUbPtr = (__ubuf__ Tout*)dstUb.GetPhyAddr();
    auto srcUbLuPrt = (__ubuf__ Tin*)srcUbLu.GetPhyAddr();
    auto srcUbRuPrt = (__ubuf__ Tin*)srcUbRu.GetPhyAddr();
    auto srcUbLdPrt = (__ubuf__ Tin*)srcUbLd.GetPhyAddr();
    auto srcUbRdPrt = (__ubuf__ Tin*)srcUbRd.GetPhyAddr();
    float delta_lu = delta_[POS_LU];
    float delta_ru = delta_[POS_RU];
    float delta_ld = delta_[POS_LD];
    float delta_rd = delta_[POS_RD];
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFp16;
        MicroAPI::MaskReg pregFp32;
        MicroAPI::RegTensor<Tin> reg_srcLu;
        MicroAPI::RegTensor<Tin> reg_srcRu;
        MicroAPI::RegTensor<Tin> reg_srcLd;
        MicroAPI::RegTensor<Tin> reg_srcRd;
        MicroAPI::RegTensor<Tin> reg_srcLui32;
        MicroAPI::RegTensor<Tin> reg_srcRui32;
        MicroAPI::RegTensor<Tin> reg_srcLdi32;
        MicroAPI::RegTensor<Tin> reg_srcRdi32;
        MicroAPI::RegTensor<Tout> regOutputT2;
        MicroAPI::RegTensor<Tout> regTmpT2;
        MicroAPI::RegTensor<float> regSrcLuf32;
        MicroAPI::RegTensor<float> regSrcRuf32;
        MicroAPI::RegTensor<float> regSrcLdf32;
        MicroAPI::RegTensor<float> regSrcRdf32;
        MicroAPI::RegTensor<float> regDeltaLu;
        MicroAPI::RegTensor<float> regDeltaRu;
        MicroAPI::RegTensor<float> regDeltaLd;
        MicroAPI::RegTensor<float> regDeltaRd;
        MicroAPI::RegTensor<float> regSumUpperF32;
        MicroAPI::RegTensor<float> regSumDownF32;
        MicroAPI::RegTensor<float> regSumF32;

        for (uint16_t idx = 0; idx < repeatTimes; idx++) {
            pregFp32 = AscendC::MicroAPI::UpdateMask<float>(totalLen);
            MicroAPI::DataCopy<Tin, MicroAPI::PostLiteral::POST_MODE_UPDATE>(reg_srcLu, srcUbLuPrt, oneRepeat_);
            MicroAPI::DataCopy<Tin, MicroAPI::PostLiteral::POST_MODE_UPDATE>(reg_srcRu, srcUbRuPrt, oneRepeat_);
            MicroAPI::DataCopy<Tin, MicroAPI::PostLiteral::POST_MODE_UPDATE>(reg_srcLd, srcUbLdPrt, oneRepeat_);
            MicroAPI::DataCopy<Tin, MicroAPI::PostLiteral::POST_MODE_UPDATE>(reg_srcRd, srcUbRdPrt, oneRepeat_);
            if constexpr (sizeof(Tin) != sizeof(int32_t)) {
                MicroAPI::UnPack((RegTensor<int32_t>&)reg_srcLui32, (RegTensor<int16_t>&)reg_srcLu);
                MicroAPI::UnPack((RegTensor<int32_t>&)reg_srcRui32, (RegTensor<int16_t>&)reg_srcRu);
                MicroAPI::UnPack((RegTensor<int32_t>&)reg_srcLdi32, (RegTensor<int16_t>&)reg_srcLd);
                MicroAPI::UnPack((RegTensor<int32_t>&)reg_srcRdi32, (RegTensor<int16_t>&)reg_srcRd);
                MicroAPI::Cast<float, Tin, castTrait0>(regSrcLuf32, reg_srcLui32, pregFp32);
                MicroAPI::Cast<float, Tin, castTrait0>(regSrcRuf32, reg_srcRui32, pregFp32);
                MicroAPI::Cast<float, Tin, castTrait0>(regSrcLdf32, reg_srcLdi32, pregFp32);
                MicroAPI::Cast<float, Tin, castTrait0>(regSrcRdf32, reg_srcRdi32, pregFp32);
                MicroAPI::Muls(regSrcLuf32, regSrcLuf32, delta_lu, pregFp32);
                MicroAPI::Muls(regSrcRuf32, regSrcRuf32, delta_ru, pregFp32);
                MicroAPI::Muls(regSrcLdf32, regSrcLdf32, delta_ld, pregFp32);
                MicroAPI::Muls(regSrcRdf32, regSrcRdf32, delta_rd, pregFp32);
            } else {
                MicroAPI::Muls(regSrcLuf32, reg_srcLu, delta_lu, pregFp32);
                MicroAPI::Muls(regSrcRuf32, reg_srcRu, delta_ru, pregFp32);
                MicroAPI::Muls(regSrcLdf32, reg_srcLd, delta_ld, pregFp32);
                MicroAPI::Muls(regSrcRdf32, reg_srcRd, delta_rd, pregFp32);
            }
            MicroAPI::Add(regSumUpperF32, regSrcLuf32, regSrcRuf32, pregFp32);
            MicroAPI::Add(regSumDownF32, regSrcLdf32, regSrcRdf32, pregFp32);
            MicroAPI::Add(regSumF32, regSumUpperF32, regSumDownF32, pregFp32);

            if constexpr (sizeof(Tout) == sizeof(int16_t)) {
                MicroAPI::Cast<Tout, float, castTrait1>(regTmpT2, regSumF32, pregFp32);
                MicroAPI::Pack((RegTensor<uint16_t>&)regOutputT2, (RegTensor<uint32_t>&)regTmpT2);
                MicroAPI::MaskPack(pregFp16, pregFp32);
                MicroAPI::DataCopy<Tout, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dstUbPtr, regOutputT2, oneRepeat_, pregFp16);
            } else {
                MicroAPI::DataCopy<Tout, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dstUbPtr, regSumF32, oneRepeat_, pregFp32);
            }
        }
    }

    inQueue0.FreeTensor(srcUbLu);
    inQueue1.FreeTensor(srcUbRu);
    inQueue2.FreeTensor(srcUbLd);
    inQueue3.FreeTensor(srcUbRd);
}

template <typename Tin, typename Tout>
__aicore__ inline void ResizeBilinearV2Nc<Tin, Tout>::ComputLengthOffset(OffsetDefSt& stOffset)
{
    int64_t nDivNum = Ops::Base::CeilDiv(lenN_, nFactor_);
    int64_t hwDivNum = Ops::Base::CeilDiv(lenDstHw_, hwFactor_);
    int64_t cDivNum = Ops::Base::CeilDiv(lenC_, cFactor_);
    int64_t tmpBlockIdx = GetBlockIdx();
    int64_t cIdx = tmpBlockIdx % cDivNum;
    tmpBlockIdx = tmpBlockIdx / cDivNum;
    int64_t hwIdx = tmpBlockIdx % hwDivNum;
    tmpBlockIdx = tmpBlockIdx / hwDivNum;
    int64_t nIdx = tmpBlockIdx % nDivNum;

    stOffset.nStart = nIdx * nFactor_;
    stOffset.hwStart = hwIdx * hwFactor_;
    stOffset.cStart = cIdx * cFactor_;

    int64_t nTailLength = lenN_ - (nDivNum - 1) * nFactor_;
    int64_t hwTailLength = lenDstHw_ - (hwDivNum - 1) * hwFactor_;
    int64_t cTailLength = lenC_ - (cDivNum - 1) * cFactor_;
    stOffset.nLength = (nIdx < nDivNum - 1) ? nFactor_ : nTailLength;
    stOffset.hwLength = (hwIdx < hwDivNum - 1) ? hwFactor_ : hwTailLength;
    stOffset.cLength = (cIdx < cDivNum - 1) ? cFactor_ : cTailLength;
}

template <typename Tin, typename Tout>
__aicore__ inline void ResizeBilinearV2Nc<Tin, Tout>::DivideNhwc()
{
    int64_t dst_h = 0;
    int64_t dst_w = 0;
    int64_t src_offset = 0;
    int64_t base_src_offset = 0;
    int64_t dst_offset = 0;
    int64_t base_dst_offset = 0;
    int64_t src_upper = 0;
    int64_t src_down = 0;
    int64_t src_right = 0;
    int64_t src_left = 0;
    uint32_t process_length = 0;
    OffsetDefSt point_offset_st = {0};
    ComputLengthOffset(point_offset_st);
    for (int64_t idx_hw = point_offset_st.hwStart; idx_hw < point_offset_st.hwStart + point_offset_st.hwLength;
         idx_hw++) {
        dst_h = idx_hw / lenDstW_;
        dst_w = idx_hw % lenDstW_;
        ComputeDeltaArgus(idx_hw, src_upper, src_down, src_left, src_right);
        for (int64_t idx_n = point_offset_st.nStart; idx_n < point_offset_st.nStart + point_offset_st.nLength;
             idx_n++) {
            base_src_offset = idx_n * lenSrcHw_ * lenC_;
            base_dst_offset = idx_n * lenDstHw_ * lenC_ + (lenDstW_ * dst_h + dst_w) * lenC_;
            for (int64_t idx_c = 0; idx_c < point_offset_st.cLength; idx_c = idx_c + ubCFactor_) {
                process_length =
                    (idx_c + ubCFactor_) <= point_offset_st.cLength ? ubCFactor_ : point_offset_st.cLength - idx_c;
                // Step1: copy data from input gm to ub
                src_offset =
                    base_src_offset + (lenSrcW_ * src_upper + src_left) * lenC_ + idx_c + point_offset_st.cStart;
                DataCopyPadGm2Ub(inQueue0, xGm_, src_offset, process_length);
                src_offset =
                    base_src_offset + (lenSrcW_ * src_upper + src_right) * lenC_ + idx_c + point_offset_st.cStart;
                DataCopyPadGm2Ub(inQueue1, xGm_, src_offset, process_length);
                src_offset =
                    base_src_offset + (lenSrcW_ * src_down + src_left) * lenC_ + idx_c + point_offset_st.cStart;
                DataCopyPadGm2Ub(inQueue2, xGm_, src_offset, process_length);
                src_offset =
                    base_src_offset + (lenSrcW_ * src_down + src_right) * lenC_ + idx_c + point_offset_st.cStart;
                DataCopyPadGm2Ub(inQueue3, xGm_, src_offset, process_length);
                // Step2: compute value throw MicroAPI
                LocalTensor<Tout> ubTensorOut = outQueue.AllocTensor<Tout>();
                ComputeDstValueWith4SrcDot(ubTensorOut, process_length);
                outQueue.EnQue(ubTensorOut);
                // Step3: copy data from ub to output gm
                dst_offset = base_dst_offset + idx_c + point_offset_st.cStart;
                DataCopyPadUb2Gm(outQueue, yGm_, dst_offset, process_length);
            }
        }
    }
    return;
}

template <typename Tin, typename Tout>
__aicore__ inline void ResizeBilinearV2Nc<Tin, Tout>::ComputeSrcIdx(int64_t& dstIdx, float& srcIdx, float scale)
{
    if ((alignCorners_ == 0) && (halfPixelCenters_ == 1)) {
        srcIdx = (dstIdx + 0.5f) * scale - 0.5f;
    } else {
        srcIdx = dstIdx * scale;
    }
}

template <typename Tin, typename Tout>
__aicore__ inline void ResizeBilinearV2Nc<Tin, Tout>::ComputeDeltaArgus(
    int64_t idxHw, int64_t& srcUpper, int64_t& srcDown, int64_t& srcLeft, int64_t& srcRight)
{
    int64_t dst_h = idxHw / lenDstW_;
    float hFp = 0.0f;
    ComputeSrcIdx(dst_h, hFp, scaleH_);
    float hLerp = hFp - Floor(hFp);
    srcUpper = (hFp > 0.0f) ? Floor(hFp) : 0;
    srcUpper = min(srcUpper, lenSrcH_ - 1);
    srcDown = (hFp > 0.0f) ? Ceil(hFp) : 0;
    srcDown = (srcDown >= lenSrcH_) ? lenSrcH_ - 1 : srcDown;

    int64_t dst_w = idxHw % lenDstW_;
    float wFp = 0.0f;
    ComputeSrcIdx(dst_w, wFp, scaleW_);
    float wLerp = wFp - Floor(wFp);
    srcLeft = (wFp > 0.0f) ? Floor(wFp) : 0;
    srcLeft = min(srcLeft, lenSrcW_ - 1);
    srcRight = (wFp > 0.0f) ? Ceil(wFp) : 0;
    srcRight = (srcRight >= lenSrcW_) ? lenSrcW_ - 1 : srcRight;

    delta_[POS_LU] = (1.0f - hLerp) * (1.0f - wLerp);
    delta_[POS_RU] = (1.0f - hLerp) * wLerp;
    delta_[POS_LD] = hLerp * (1.0f - wLerp);
    delta_[POS_RD] = hLerp * wLerp;
}
} // namespace ResizeBilinearV2
#endif // namespace ResizeBilinearV2
