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
 * \file resize_bilinear_v2_broadcast_nchw.h
 * \brief resize_bilinear_v2_broadcast_nchw
 */
#ifndef RESIZE_BILINEAR_V2_BROADCAST_NCHW_H
#define RESIZE_BILINEAR_V2_BROADCAST_NCHW_H

#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"

namespace ResizeBilinearV2 {
using namespace AscendC;

constexpr static int32_t VECTOR_LEN = platform::GetVRegSize();

template <typename T_DATA>
class ResizeBilinearV2BroadcastNCHW : public ResizeBilinearV2Base {
public:
    __aicore__ inline ResizeBilinearV2BroadcastNCHW(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe *pipe, const ResizeBilinearV2TilingData *data);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    template <typename U>
    __aicore__ inline void Compute(LocalTensor<T_DATA> &xTensor);

    __aicore__ inline void SyncM2toS()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
    };

    __aicore__ inline void SyncStoV()
    {
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    };

    const ResizeBilinearV2TilingData *tilingData_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;

    DataCopyPadExtParams<uint8_t> padParams_ = {false, 0, 0, 0};

    int64_t nStrideX_;
    int64_t cStrideX_;
    int64_t hwStrideX_;

    int64_t nStrideY_;
    int64_t cStrideY_;
    int64_t hwStrideY_;

    int64_t nOffset_;
    int64_t cOffset_;
    int64_t hwOffset_;

    int64_t nLength_;
    int64_t cLength_;
    int64_t hwLength_;
    int64_t hwLengthAlign_;
};

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2BroadcastNCHW<T_DATA>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe *pipe, const ResizeBilinearV2TilingData *data)
{
    this->BaseInit(x, size, y, pipe);

    tilingData_ = data;

    int64_t bufferSizeX = tilingData_->ubNFactor * tilingData_->ubCFactor * sizeof(T_DATA);
    this->pipe_->InitBuffer(xQue_, 2, bufferSizeX);

    int64_t bufferSizeY = tilingData_->ubNFactor * tilingData_->ubCFactor *
                          ops::CeilAlign<int64_t>(tilingData_->ubHWFactor * sizeof(T_DATA), ONE_BLOCK_BYTE);
    this->pipe_->InitBuffer(yQue_, 2, bufferSizeY);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2BroadcastNCHW<T_DATA>::CopyIn()
{
    LocalTensor<uint8_t> xTensor = xQue_.AllocTensor<uint8_t>();

    int64_t xOffsetInGM = nOffset_ * nStrideX_ + cOffset_ * cStrideX_;

    DataCopyExtParams gm2ubParams;
    if (cLength_ >= tilingData_->lenC) {
        gm2ubParams.blockCount = 1;
        gm2ubParams.blockLen = nLength_ * cLength_ * sizeof(T_DATA);
        gm2ubParams.srcStride = 0;
        gm2ubParams.dstStride = 0;
    } else {
        gm2ubParams.blockCount = nLength_;
        gm2ubParams.blockLen = cLength_ * sizeof(T_DATA);
        gm2ubParams.srcStride = (nStrideX_ - cLength_) * sizeof(T_DATA);
        gm2ubParams.dstStride = 0;
    }

    DataCopyPad<uint8_t, PaddingMode::Compact>(xTensor, xGM_[xOffsetInGM * sizeof(T_DATA)], gm2ubParams, padParams_);

    xQue_.EnQue(xTensor);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2BroadcastNCHW<T_DATA>::CopyOut()
{
    LocalTensor<uint8_t> yTensor = yQue_.DeQue<uint8_t>();

    int64_t yOffsetInGM = nOffset_ * nStrideY_ + cOffset_ * cStrideY_ + hwOffset_;

    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1Size = nLength_;
    loopParams.loop1SrcStride = cLength_ * hwLengthAlign_ * sizeof(T_DATA);
    loopParams.loop1DstStride = nStrideY_ * sizeof(T_DATA);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams ub2gmParams;
    ub2gmParams.blockCount = cLength_;
    ub2gmParams.blockLen = hwLength_ * sizeof(T_DATA);
    ub2gmParams.srcStride = 0;
    ub2gmParams.dstStride = (cStrideY_ - hwLength_) * sizeof(T_DATA);

    DataCopyPad(yGM_[yOffsetInGM * sizeof(T_DATA)], yTensor, ub2gmParams);

    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    yQue_.FreeTensor(yTensor);
}

template <typename T_DATA>
template <typename U>
__aicore__ inline void ResizeBilinearV2BroadcastNCHW<T_DATA>::Compute(LocalTensor<T_DATA> &xTensor)
{
    LocalTensor<T_DATA> yTensor = yQue_.AllocTensor<T_DATA>();

    int64_t lenNC = nLength_ * cLength_;
    int64_t oneRepeat = VECTOR_LEN / sizeof(U);
    uint32_t repeatTimes = ops::CeilDiv<int64_t>(hwLength_, oneRepeat);
    uint32_t lineLen = hwLength_;
    uint32_t lineLenAlign = hwLengthAlign_;

    __ubuf__ U *xAddr = (__ubuf__ U *)xTensor.GetPhyAddr();
    __ubuf__ U *yAddr = (__ubuf__ U *)yTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<U> regData;

        for (uint16_t ncLoop = 0; ncLoop < (uint16_t)lenNC; ncLoop++) {
            U val = *xAddr;
            __ubuf__ U *yLineAddr = yAddr;
            uint32_t oneLineLen = lineLen;
            for (uint16_t inLoop = 0; inLoop < (uint16_t)repeatTimes; inLoop++) {
                preg = MicroAPI::UpdateMask<U>(oneLineLen);
                MicroAPI::Duplicate(regData, val, preg);
                MicroAPI::DataCopy<U, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    yLineAddr, regData, (int32_t)oneRepeat, preg);
            }

            xAddr++;
            yAddr += lineLenAlign;
        }
    }

    yQue_.EnQue(yTensor);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2BroadcastNCHW<T_DATA>::Process()
{
    int64_t nNum = ops::CeilDiv(tilingData_->lenN, tilingData_->nFactor);
    int64_t cNum = ops::CeilDiv(tilingData_->lenC, tilingData_->cFactor);
    int64_t hwNum = ops::CeilDiv(tilingData_->lenDesH * tilingData_->lenDesW, tilingData_->hwFactor);
    int64_t nTail = tilingData_->lenN - (nNum - 1) * tilingData_->nFactor;
    int64_t cTail = tilingData_->lenC - (cNum - 1) * tilingData_->cFactor;
    int64_t hwTail = tilingData_->lenDesH * tilingData_->lenDesW - (hwNum - 1) * tilingData_->hwFactor;

    cStrideY_ = tilingData_->lenDesH * tilingData_->lenDesW;
    nStrideY_ = cStrideY_ * tilingData_->lenC;

    cStrideX_ = tilingData_->lenSrcH * tilingData_->lenSrcW;
    nStrideX_ = cStrideX_ * tilingData_->lenC;

    int64_t block = GetBlockIdx();
    if (block >= tilingData_->realCoreNum) {
        return;
    }

    int64_t cIdx = block % cNum;
    block /= cNum;
    int64_t nIdx = block % nNum;
    block /= nNum;
    int64_t hwIdx = block % hwNum;

    int64_t nOffset4Block = nIdx * tilingData_->nFactor;
    int64_t cOffset4Block = cIdx * tilingData_->cFactor;
    int64_t hwOffset4Block = hwIdx * tilingData_->hwFactor;

    int64_t nLength4Block = (nIdx < nNum - 1) ? tilingData_->nFactor : nTail;
    int64_t cLength4Block = (cIdx < cNum - 1) ? tilingData_->cFactor : cTail;
    int64_t hwLength4Block = (hwIdx < hwNum - 1) ? tilingData_->hwFactor : hwTail;

    for (int64_t nLoop = 0; nLoop < nLength4Block; nLoop += tilingData_->ubNFactor) {
        nOffset_ = nOffset4Block + nLoop;
        nLength_ = this->Min(tilingData_->ubNFactor, nLength4Block - nLoop);
        for (int64_t cLoop = 0; cLoop < cLength4Block; cLoop += tilingData_->ubCFactor) {
            cOffset_ = cOffset4Block + cLoop;
            cLength_ = this->Min(tilingData_->ubCFactor, cLength4Block - cLoop);

            CopyIn();

            LocalTensor<T_DATA> xTensor = xQue_.DeQue<T_DATA>();
            for (int64_t hwLoop = 0; hwLoop < hwLength4Block; hwLoop += tilingData_->ubHWFactor) {
                hwOffset_ = hwOffset4Block + hwLoop;
                hwLength_ = this->Min(tilingData_->ubHWFactor, hwLength4Block - hwLoop);
                hwLengthAlign_ = ops::CeilAlign<int64_t>(hwLength_, ONE_BLOCK_BYTE / sizeof(T_DATA));

                if constexpr (sizeof(T_DATA) == sizeof(int16_t)) {
                    Compute<int16_t>(xTensor);
                } else {
                    Compute<int32_t>(xTensor);
                }

                CopyOut();
            }
            xQue_.FreeTensor(xTensor);
        }
    }
}

}  // namespace ResizeBilinearV2

#endif  // RESIZE_BILINEAR_V2_BROADCAST_NCHW_H
