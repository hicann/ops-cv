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
 * \file resize_bilinear_v2_broadcast_nhwc.h
 * \brief resize_bilinear_v2_broadcast_nhwc
 */
#ifndef RESIZE_BILINEAR_V2_BROADCAST_NHWC_H
#define RESIZE_BILINEAR_V2_BROADCAST_NHWC_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace ResizeBilinearV2 {
using namespace AscendC;

template <typename T_DATA>
class ResizeBilinearV2BroadcastNHWC : public ResizeBilinearV2Base {
public:
    __aicore__ inline ResizeBilinearV2BroadcastNHWC(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe, const ResizeBilinearV2TilingData* data);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    __aicore__ inline void Compute(LocalTensor<T_DATA> xTensor);
    __aicore__ inline void CopyLine(LocalTensor<T_DATA> dst, LocalTensor<T_DATA> src);

    const ResizeBilinearV2TilingData* tilingData_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;

    DataCopyPadExtParams<uint8_t> padParams_ = {false, 0, 0, 0};

    int64_t nStrideX_;
    int64_t hwStrideX_;

    int64_t nStrideY_;
    int64_t hwStrideY_;

    int64_t nOffset_;
    int64_t hwOffset_;
    int64_t cOffset_;

    int64_t nLength_;
    int64_t hwLength_;
    int64_t cLength_;
    int64_t cLengthAlign_;
};

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2BroadcastNHWC<T_DATA>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, TPipe* pipe, const ResizeBilinearV2TilingData* data)
{
    this->BaseInit(x, size, y, pipe);

    tilingData_ = data;

    int64_t bufferSizeX =
        tilingData_->ubNFactor * Ops::Base::CeilAlign<int64_t>(tilingData_->ubCFactor * sizeof(T_DATA), ONE_BLOCK_BYTE);
    this->pipe_->InitBuffer(xQue_, 2, bufferSizeX);

    int64_t bufferSizeY = tilingData_->ubNFactor * tilingData_->ubHWFactor *
                          Ops::Base::CeilAlign<int64_t>(tilingData_->ubCFactor * sizeof(T_DATA), ONE_BLOCK_BYTE);
    this->pipe_->InitBuffer(yQue_, 2, bufferSizeY);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2BroadcastNHWC<T_DATA>::CopyIn()
{
    LocalTensor<uint8_t> xTensor = xQue_.AllocTensor<uint8_t>();

    int64_t xOffsetInGM = nOffset_ * nStrideX_ + cOffset_;

    DataCopyExtParams gm2ubParams;
    gm2ubParams.blockCount = nLength_;
    gm2ubParams.blockLen = cLength_ * sizeof(T_DATA);
    gm2ubParams.srcStride = (nStrideX_ - cLength_) * sizeof(T_DATA);
    gm2ubParams.dstStride = 0;

    DataCopyPad(xTensor, xGM_[xOffsetInGM * sizeof(T_DATA)], gm2ubParams, padParams_);

    xQue_.EnQue(xTensor);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2BroadcastNHWC<T_DATA>::CopyOut()
{
    LocalTensor<uint8_t> yTensor = yQue_.DeQue<uint8_t>();

    int64_t yOffsetInGM = nOffset_ * nStrideY_ + hwOffset_ * hwStrideY_ + cOffset_;

    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1Size = nLength_;
    loopParams.loop1SrcStride = hwLength_ * cLengthAlign_ * sizeof(T_DATA);
    loopParams.loop1DstStride = nStrideY_ * sizeof(T_DATA);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams ub2gmParams;
    ub2gmParams.blockCount = hwLength_;
    ub2gmParams.blockLen = cLength_ * sizeof(T_DATA);
    ub2gmParams.srcStride = 0;
    ub2gmParams.dstStride = (hwStrideY_ - cLength_) * sizeof(T_DATA);

    DataCopyPad(yGM_[yOffsetInGM * sizeof(T_DATA)], yTensor, ub2gmParams);

    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    yQue_.FreeTensor(yTensor);
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2BroadcastNHWC<T_DATA>::Compute(LocalTensor<T_DATA> xTensor)
{
    LocalTensor<T_DATA> yTensor = yQue_.AllocTensor<T_DATA>();

    int64_t nStrideAlign = hwLength_ * cLengthAlign_;

    for (int64_t nLoop = 0; nLoop < nLength_; nLoop++) {
        int64_t xOffsetInUB = nLoop * cLengthAlign_;

        for (int64_t hwLoop = 0; hwLoop < hwLength_; hwLoop++) {
            int64_t yOffsetInUB = nLoop * nStrideAlign + hwLoop * cLengthAlign_;
            CopyLine(yTensor[yOffsetInUB], xTensor[xOffsetInUB]);
        }
    }

    yQue_.EnQue(yTensor);
}

template <typename T_DATA>
inline __aicore__ void ResizeBilinearV2BroadcastNHWC<T_DATA>::CopyLine(
    LocalTensor<T_DATA> yTensor, LocalTensor<T_DATA> xTensor)
{
    auto dst = yTensor.template ReinterpretCast<int16_t>();
    auto src = xTensor.template ReinterpretCast<int16_t>();

    Copy<int16_t, false>(dst, src, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
}

template <typename T_DATA>
__aicore__ inline void ResizeBilinearV2BroadcastNHWC<T_DATA>::Process()
{
    int64_t nNum = Ops::Base::CeilDiv(tilingData_->lenN, tilingData_->nFactor);
    int64_t hwNum = Ops::Base::CeilDiv(tilingData_->lenDesH * tilingData_->lenDesW, tilingData_->hwFactor);
    int64_t cNum = Ops::Base::CeilDiv(tilingData_->lenC, tilingData_->cFactor);
    int64_t nTail = tilingData_->lenN - (nNum - 1) * tilingData_->nFactor;
    int64_t hwTail = tilingData_->lenDesH * tilingData_->lenDesW - (hwNum - 1) * tilingData_->hwFactor;
    int64_t cTail = tilingData_->lenC - (cNum - 1) * tilingData_->cFactor;

    hwStrideY_ = tilingData_->lenC;
    nStrideY_ = hwStrideY_ * tilingData_->lenDesH * tilingData_->lenDesW;

    hwStrideX_ = tilingData_->lenC;
    nStrideX_ = hwStrideX_ * tilingData_->lenSrcH * tilingData_->lenSrcW;
    ;

    int64_t block = GetBlockIdx();
    if (block >= tilingData_->realCoreNum) {
        return;
    }

    int64_t cIdx = block % cNum;
    block /= cNum;
    int64_t hwIdx = block % hwNum;
    block /= hwNum;
    int64_t nIdx = block % nNum;

    int64_t nOffset4Block = nIdx * tilingData_->nFactor;
    int64_t hwOffset4Block = hwIdx * tilingData_->hwFactor;
    int64_t cOffset4Block = cIdx * tilingData_->cFactor;

    int64_t nLength4Block = (nIdx < nNum - 1) ? tilingData_->nFactor : nTail;
    int64_t hwLength4Block = (hwIdx < hwNum - 1) ? tilingData_->hwFactor : hwTail;
    int64_t cLength4Block = (cIdx < cNum - 1) ? tilingData_->cFactor : cTail;

    for (int64_t nLoop = 0; nLoop < nLength4Block; nLoop += tilingData_->ubNFactor) {
        nOffset_ = nOffset4Block + nLoop;
        nLength_ = this->Min(tilingData_->ubNFactor, nLength4Block - nLoop);
        for (int64_t cLoop = 0; cLoop < cLength4Block; cLoop += tilingData_->ubCFactor) {
            cOffset_ = cOffset4Block + cLoop;
            cLength_ = this->Min(tilingData_->ubCFactor, cLength4Block - cLoop);
            cLengthAlign_ = Ops::Base::CeilAlign<int64_t>(cLength_, ONE_BLOCK_BYTE / sizeof(T_DATA));

            SetMaskCount();
            SetVectorMask<int16_t, MaskMode::COUNTER>(cLength_ * sizeof(T_DATA) / sizeof(int16_t));

            CopyIn();

            LocalTensor<T_DATA> xTensor = xQue_.DeQue<T_DATA>();
            for (int64_t hwLoop = 0; hwLoop < hwLength4Block; hwLoop += tilingData_->ubHWFactor) {
                hwOffset_ = hwOffset4Block + hwLoop;
                hwLength_ = this->Min(tilingData_->ubHWFactor, hwLength4Block - hwLoop);

                Compute(xTensor);
                CopyOut();
            }
            xQue_.FreeTensor(xTensor);

            SetMaskNorm();
            ResetMask();
        }
    }
}

} // namespace ResizeBilinearV2

#endif // RESIZE_BILINEAR_V2_BROADCAST_NHWC_H
