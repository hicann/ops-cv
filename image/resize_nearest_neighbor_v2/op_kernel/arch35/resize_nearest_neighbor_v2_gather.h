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
 * \file resize_nearest_neighbor_v2_gather.h
 * \brief resize_nearest_neighbor_v2_gather
 */
#ifndef RESIZE_NEAREAST_NEIGHBOR_V2_GATHER_H
#define RESIZE_NEAREAST_NEIGHBOR_V2_GATHER_H

#include "resize_nearest_neighbor_v2_base.h"

namespace ResizeNearestNeighborV2 {
using namespace AscendC;

template <typename T>
class ResizeNearestNeighborV2Gather : public ResizeNearestNeighborV2Base {
public:
    __aicore__ inline ResizeNearestNeighborV2Gather(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, const ResizeNearestNeighborV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerBlock(int64_t ncLoopNum);
    __aicore__ inline void CopyInWithPad(int64_t inOffset);
    __aicore__ inline void Compute(int64_t hIdx, int64_t wIdx, int64_t hNum, int64_t wNum);
    __aicore__ inline void Compute32(
        __ubuf__ T *dstAddr, __ubuf__ T *srcAddr, const LocalTensor<int32_t> &hIdexUb, float startIdx, uint32_t num);
    __aicore__ inline void ComputeMain(
        __ubuf__ T *dstAddr, __ubuf__ T *srcAddr, const LocalTensor<int32_t> &hIdexUb, float startIdx, uint32_t num);
    __aicore__ inline void ComputeTail(
        __ubuf__ T *dstAddr, __ubuf__ T *srcAddr, const LocalTensor<int32_t> &hIdexUb, float startIdx);
    __aicore__ inline void CopyOutWithPad(int64_t outOffset, int64_t hNum, int64_t wNum);
    __aicore__ inline int64_t ComputeSrcOffset(int64_t hIdx, int64_t wIdx, int64_t hNum, int64_t wNum);
    __aicore__ inline int64_t ComputeBlockOffset(int64_t ncIdx, int64_t hwSize);
    __aicore__ inline int64_t CalcSrcIndex(int64_t dstIndex, float scale, int64_t srcIdxMax);

    constexpr static int32_t bufferNum = 2;
    constexpr static int64_t NUM_FOUR = 4;
    constexpr static int32_t blockSize = 32;
    constexpr static AscendC::MicroAPI::CastTrait castTraitRound = {AscendC::MicroAPI::RegLayout::UNKNOWN,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::MicroAPI::CastTrait castTraitFloor = {AscendC::MicroAPI::RegLayout::UNKNOWN,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_FLOOR};

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX;
    TQue<QuePosition::VECOUT, bufferNum> outQueueY;
    TBuf<QuePosition::VECCALC> coordinateQueueH;
    GlobalTensor<T> xGm, yGm;

    int32_t blockIdx_ = 0;
    int64_t gmOutOffset_ = 0;
    int64_t blockInOffset_ = 0;
    int64_t blockOutOffset_ = 0;

    uint32_t vlSize_ = VECTOR_REG_WIDTH / sizeof(T);  // vl size: 256 / dtypeSize
    uint32_t elementBlock_ = blockSize / sizeof(T);
    uint32_t elementBlock32_ = blockSize / sizeof(float);
    uint32_t repeatElm_ = vlSize_ / 2;
    uint32_t hNum_ = 0;
    uint32_t wNum_ = 0;
    uint32_t wTail_ = 0;
    uint32_t srcWAlign_ = 1;
    int32_t startIndexW_ = 0;
    int32_t startIndexH_ = 0;

    // tiling params
    const ResizeNearestNeighborV2TilingData *tiling_;
};

template <typename T>
__aicore__ inline void ResizeNearestNeighborV2Gather<T>::Init(
    GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, const ResizeNearestNeighborV2TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    tiling_ = tilingData;

    // 接收tilingdata信息
    hScale_ = tiling_->scaleH;
    wScale_ = tiling_->scaleW;
    srcHSize_ = tiling_->lenSrcH;
    srcWSize_ = tiling_->lenSrcW;
    dstHSize_ = tiling_->lenDesH;
    dstWSize_ = tiling_->lenDesW;
    hFactor_ = tiling_->splitFactorDesH;
    wFactor_ = tiling_->splitFactorDesW;
    hTailFactor_ = tiling_->splitFactorTailDesH;
    wTailFactor_ = tiling_->splitFactorTailDesW;
    hwCnt_ = tiling_->splitCountDesH * tiling_->splitCountDesW;
    bias_ = tiling_->halfPixelCenters == 1 ? 0.5f : 0.0f;

    // 计算预估srcfactor最大所需空间
    int32_t srcHEstimate = this->Ceil(static_cast<float>((hFactor_ + 0.5f) * hScale_));
    int32_t srcWEstimate = this->Ceil(static_cast<float>((wFactor_ + 0.5f) * wScale_));

    int32_t srcSize = srcHEstimate * CeilDivision(srcWEstimate, elementBlock_) * elementBlock_;
    int32_t dstSize = hFactor_ * CeilDivision(wFactor_, elementBlock_) * elementBlock_;
    int32_t coordinateHSize = CeilDivision(hFactor_, elementBlock32_) * elementBlock32_;

    pipe.InitBuffer(inQueueX, bufferNum, srcSize * sizeof(T));
    pipe.InitBuffer(outQueueY, bufferNum, dstSize * sizeof(T));
    pipe.InitBuffer(coordinateQueueH, coordinateHSize * sizeof(float));  // 坐标为float类型
}

/*
 * The entrance of Distribution from cpp-file
 */
template <typename T>
__aicore__ inline void ResizeNearestNeighborV2Gather<T>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    int64_t ncLoopNum = 0;
    // 计算offset
    int64_t totalNum = 0;
    if (blockIdx_ < tiling_->splitBlockFullCount) {
        ncLoopNum = tiling_->splitBlockFactor;
        totalNum = blockIdx_ * tiling_->splitBlockFactor;
    } else {
        ncLoopNum = tiling_->splitBlockTailFactor;
        totalNum = tiling_->splitBlockFullCount * tiling_->splitBlockFactor +
                   (blockIdx_ - tiling_->splitBlockFullCount) * tiling_->splitBlockTailFactor;
    }

    blockInOffset_ = totalNum * srcHSize_ * srcWSize_;
    blockOutOffset_ = totalNum * dstHSize_ * dstWSize_;

    ProcessPerBlock(ncLoopNum);
}

template <typename T>
__aicore__ inline int64_t ResizeNearestNeighborV2Gather<T>::CalcSrcIndex(
    int64_t dstIndex, float scale, int64_t srcIdxMax)
{
    int64_t srcIndex = 0;
    if (tiling_->halfPixelCenters == 1) {
        if (tiling_->alignCorners == 1) {
            srcIndex = this->Round((static_cast<float>(dstIndex) + 0.5f) * scale);
        } else {
            srcIndex = this->Floor((static_cast<float>(dstIndex) + 0.5f) * scale);
        }
    } else {
        if (tiling_->alignCorners) {
            srcIndex = this->Round((static_cast<float>(dstIndex) * scale));
        } else {
            srcIndex = this->Floor((static_cast<float>(dstIndex) * scale));
        }
    }
    return this->Min(srcIndex, srcIdxMax - 1);
}

/*
 * ProcessPerBlock
 */
template <typename T>
__aicore__ inline void ResizeNearestNeighborV2Gather<T>::ProcessPerBlock(int64_t ncLoopNum)
{
    int64_t totalSrcHw = srcHSize_ * srcWSize_;
    int64_t totalDstHw = dstHSize_ * dstWSize_;
    // todo 封装四个函数
    // process main loop
    int64_t loopH = tiling_->splitCountDesH - 1;
    int64_t loopW = tiling_->splitCountDesW - 1;
    if (tiling_->splitCountDesH == 1 && tiling_->splitCountDesW > 1) {
        loopH = 1;
    }
    if (tiling_->splitCountDesW == 1 && tiling_->splitCountDesH > 1) {
        loopW = 1;
    }
    for (int64_t hIndex = 0; hIndex < loopH; hIndex++) {
        int64_t hOutOffset = hIndex * hFactor_ * dstWSize_;
        int64_t hInOffset = CalcSrcIndex(hIndex * hFactor_, hScale_, srcHSize_);
        srcHFactor_ = CalcSrcIndex(hIndex * hFactor_ + hFactor_ - 1, hScale_, srcHSize_) - hInOffset + 1;
        for (int64_t wIndex = 0; wIndex < loopW; wIndex++) {
            int64_t wInOffset = CalcSrcIndex(wIndex * wFactor_, wScale_, srcWSize_);
            srcWFactor_ = CalcSrcIndex(wIndex * wFactor_ + wFactor_ - 1, wScale_, srcWSize_) - wInOffset + 1;
            for (int64_t ncIdx = 0; ncIdx < ncLoopNum; ncIdx++) {
                int64_t inOffset = ncIdx * totalSrcHw + hInOffset * srcWSize_ + wInOffset;
                int64_t outOffset = ncIdx * totalDstHw + hOutOffset + wIndex * wFactor_;
                CopyInWithPad(inOffset);
                Compute(hIndex, wIndex, hFactor_, wFactor_);
                CopyOutWithPad(outOffset, hFactor_, wFactor_);
            }
        }
    }

    // process w tail
    if (tiling_->splitCountDesW > 1) {
        for (int64_t hIndex = 0; hIndex < tiling_->splitCountDesH - 1; hIndex++) {
            int64_t wIndex = tiling_->splitCountDesW - 1;
            int64_t hInOffset = CalcSrcIndex(hIndex * hFactor_, hScale_, srcHSize_);
            int64_t wInOffset = CalcSrcIndex(dstWSize_ - wTailFactor_, wScale_, srcWSize_);
            srcHFactor_ = CalcSrcIndex(hIndex * hFactor_ + hFactor_ - 1, hScale_, srcHSize_) - hInOffset + 1;
            srcWFactor_ = srcWSize_ - wInOffset;
            int64_t hOutOffset = hIndex * hFactor_ * dstWSize_;
            for (int64_t ncIdx = 0; ncIdx < ncLoopNum; ncIdx++) {
                int64_t inOffset = ncIdx * totalSrcHw + hInOffset * srcWSize_ + wInOffset;
                int64_t outOffset = ncIdx * totalDstHw + hOutOffset + dstWSize_ - wTailFactor_;
                CopyInWithPad(inOffset);
                Compute(hIndex, wIndex, hFactor_, wTailFactor_);
                CopyOutWithPad(outOffset, hFactor_, wTailFactor_);
            }
        }
    }
    // process h tail
    if (tiling_->splitCountDesH > 1) {
        int64_t hIndex = tiling_->splitCountDesH - 1;
        int64_t hOutOffset = (dstHSize_ - hTailFactor_) * dstWSize_;
        int64_t hInOffset = CalcSrcIndex(dstHSize_ - hTailFactor_, hScale_, srcHSize_);
        srcHFactor_ = srcHSize_ - hInOffset;
        for (int64_t wIndex = 0; wIndex < tiling_->splitCountDesW - 1; wIndex++) {
            int64_t wInOffset = CalcSrcIndex(wIndex * wFactor_, wScale_, srcWSize_);
            srcWFactor_ = CalcSrcIndex(wIndex * wFactor_ + wFactor_ - 1, wScale_, srcWSize_) - wInOffset + 1;
            for (int64_t ncIdx = 0; ncIdx < ncLoopNum; ncIdx++) {
                int64_t inOffset = ncIdx * totalSrcHw + hInOffset * srcWSize_ + wInOffset;
                int64_t outOffset = ncIdx * totalDstHw + hOutOffset + wIndex * wFactor_;
                CopyInWithPad(inOffset);
                Compute(hIndex, wIndex, hTailFactor_, wFactor_);
                CopyOutWithPad(outOffset, hTailFactor_, wFactor_);
            }
        }
    }
    // process w and h tail
    int64_t hIndex = tiling_->splitCountDesH - 1;
    int64_t wIndex = tiling_->splitCountDesW - 1;
    int64_t hOutOffset = (dstHSize_ - hTailFactor_) * dstWSize_;
    int64_t hInOffset = CalcSrcIndex(dstHSize_ - hTailFactor_, hScale_, srcHSize_);
    int64_t wInOffset = CalcSrcIndex(dstWSize_ - wTailFactor_, wScale_, srcWSize_);
    srcHFactor_ = srcHSize_ - hInOffset;
    srcWFactor_ = srcWSize_ - wInOffset;
    for (int64_t ncIdx = 0; ncIdx < ncLoopNum; ncIdx++) {
        int64_t inOffset = ncIdx * totalSrcHw + hInOffset * srcWSize_ + wInOffset;
        int64_t outOffset = ncIdx * totalDstHw + hOutOffset + dstWSize_ - wTailFactor_;
        CopyInWithPad(inOffset);
        Compute(hIndex, wIndex, hTailFactor_, wTailFactor_);
        CopyOutWithPad(outOffset, hTailFactor_, wTailFactor_);
    }
}

template <typename T>
__aicore__ inline void ResizeNearestNeighborV2Gather<T>::ComputeMain(
    __ubuf__ T *dstAddr, __ubuf__ T *srcAddr, const LocalTensor<int32_t> &hIdexUb, float startIdx, uint32_t num)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> idxInitLower;
        AscendC::MicroAPI::RegTensor<int32_t> idxInitHeigher;
        AscendC::MicroAPI::RegTensor<float> idxLowerF;
        AscendC::MicroAPI::RegTensor<float> idxHigherF;
        AscendC::MicroAPI::RegTensor<int32_t> idxLowerI;
        AscendC::MicroAPI::RegTensor<int32_t> idxHigherI;
        AscendC::MicroAPI::RegTensor<uint16_t> idxLower;
        AscendC::MicroAPI::RegTensor<uint16_t> idxHigher;
        AscendC::MicroAPI::RegTensor<T> vDstReg;
        AscendC::MicroAPI::UnalignReg u0;

        AscendC::MicroAPI::MaskReg preg32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t sregLast = num;
        AscendC::MicroAPI::MaskReg preg16 = AscendC::MicroAPI::UpdateMask<uint16_t>(sregLast);
        float sregLow = startIdx;
        Arange(idxLowerF, sregLow);
        Muls(idxLowerF, idxLowerF, wScale_, preg32);
        float sregHeigh = startIdx + static_cast<float>(repeatElm_);
        Arange(idxHigherF, sregHeigh);
        Muls(idxHigherF, idxHigherF, wScale_, preg32);

        if (tiling_->alignCorners == 1) {
            Cast<int32_t, float, castTraitRound>(idxInitLower, idxLowerF, preg32);
            Cast<int32_t, float, castTraitRound>(idxInitHeigher, idxHigherF, preg32);
        } else {
            Cast<int32_t, float, castTraitFloor>(idxInitLower, idxLowerF, preg32);
            Cast<int32_t, float, castTraitFloor>(idxInitHeigher, idxHigherF, preg32);
        }
        Adds(idxInitLower, idxInitLower, startIndexW_, preg32);
        Adds(idxInitHeigher, idxInitHeigher, startIndexW_, preg32);

        __ubuf__ T *dstUbT = dstAddr;
        for (uint16_t j = 0; j < static_cast<uint16_t>(hNum_); ++j) {
            Adds(idxLowerI, idxInitLower, static_cast<int32_t>(srcWAlign_) * hIdexUb.GetValue(j), preg32);
            MicroAPI::Pack<uint16_t, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(idxLower, idxLowerI);
            Adds(idxHigherI, idxInitHeigher, static_cast<int32_t>(srcWAlign_) * hIdexUb.GetValue(j), preg32);
            MicroAPI::Pack<uint16_t, int32_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(idxHigher, idxHigherI);
            Or(idxLower, idxLower, idxHigher, preg16);
            DataCopyGather(vDstReg, srcAddr, idxLower, preg16);
            dstUbT = dstAddr + j * wNum_;
            DataCopyUnAlign(dstUbT, vDstReg, u0, num);
            AscendC::MicroAPI::DataCopyUnAlignPost(dstUbT, u0, 0);
        }
    }
}

template <typename T>
__aicore__ inline void ResizeNearestNeighborV2Gather<T>::ComputeTail(
    __ubuf__ T *dstAddr, __ubuf__ T *srcAddr, const LocalTensor<int32_t> &hIdexUb, float startIdx)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> idxLowerF;
        AscendC::MicroAPI::RegTensor<int32_t> idxLowerI;
        AscendC::MicroAPI::RegTensor<int32_t> idxInit;
        AscendC::MicroAPI::RegTensor<uint16_t> idxLower;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::RegTensor<T> vDstReg;

        AscendC::MicroAPI::MaskReg mask0 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t sregTail = wTail_;
        float sregLow = startIdx;
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<uint16_t>(sregTail);
        // init w direction index
        Arange(idxLowerF, sregLow);
        Muls(idxLowerF, idxLowerF, wScale_, mask0);
        if (tiling_->alignCorners == 1) {
            Cast<int32_t, float, castTraitRound>(idxInit, idxLowerF, mask0);
        } else {
            Cast<int32_t, float, castTraitFloor>(idxInit, idxLowerF, mask0);
        }
        Adds(idxInit, idxInit, startIndexW_, mask0);

        __ubuf__ T *dstUbT = dstAddr;
        for (uint16_t i = 0; i < static_cast<uint16_t>(hNum_); ++i) {
            Adds(idxLowerI, idxInit, srcWAlign_ * hIdexUb.GetValue(i), mask0);
            MicroAPI::Pack<uint16_t, int32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(idxLower, idxLowerI);
            DataCopyGather(vDstReg, srcAddr, idxLower, mask1);
            dstUbT = dstAddr + i * wNum_;
            DataCopyUnAlign(dstUbT, vDstReg, u0, wTail_);
            AscendC::MicroAPI::DataCopyUnAlignPost(dstUbT, u0, 0);
        }
    }
}

template <typename T>
__aicore__ inline void ResizeNearestNeighborV2Gather<T>::Compute32(
    __ubuf__ T *dstAddr, __ubuf__ T *srcAddr, const LocalTensor<int32_t> &hIdexUb, float startIdx, uint32_t num)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> idxFloat;
        AscendC::MicroAPI::RegTensor<int32_t> idxInt32;
        AscendC::MicroAPI::RegTensor<int32_t> idxInit;
        AscendC::MicroAPI::RegTensor<T> vDstReg;
        AscendC::MicroAPI::UnalignReg u0;

        uint32_t sregTail = num;
        AscendC::MicroAPI::MaskReg mask0 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<uint32_t>(sregTail);

        float sregLow = startIdx;
        Arange(idxFloat, sregLow);
        Muls(idxFloat, idxFloat, wScale_, mask0);
        if (tiling_->alignCorners == 1) {
            Cast<int32_t, float, castTraitRound>(idxInit, idxFloat, mask0);
        } else {
            Cast<int32_t, float, castTraitFloor>(idxInit, idxFloat, mask0);
        }
        Adds(idxInit, idxInit, startIndexW_, mask0);

        __ubuf__ T *dstUbT = dstAddr;
        for (uint16_t i = 0; i < static_cast<uint16_t>(hNum_); ++i) {
            Adds(idxInt32, idxInit, static_cast<int32_t>(srcWAlign_) * hIdexUb.GetValue(i), mask0);
            DataCopyGather(vDstReg, srcAddr, (AscendC::MicroAPI::RegTensor<uint32_t> &)idxInt32, mask1);
            dstUbT = dstAddr + i * wNum_;
            DataCopyUnAlign(dstUbT, vDstReg, u0, num);
            AscendC::MicroAPI::DataCopyUnAlignPost(dstUbT, u0, 0);
        }
    }
}

template <typename T>
__aicore__ inline void ResizeNearestNeighborV2Gather<T>::Compute(int64_t hIdx, int64_t wIdx, int64_t hNum, int64_t wNum)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

    LocalTensor<float> hCooLocal = coordinateQueueH.AllocTensor<float>();
    LocalTensor<int32_t> hIdexUb = hCooLocal.template ReinterpretCast<int32_t>();
    // Create dst index
    float startW = static_cast<float>(wIdx * wFactor_) + bias_;
    float startH = static_cast<float>(hIdx * hFactor_) + bias_;
    CreateVecIndex(hCooLocal, startH, hNum);
    Muls(hCooLocal, hCooLocal, hScale_, hNum);

    if (tiling_->alignCorners == 1) {
        Cast(hIdexUb, hCooLocal, RoundMode::CAST_ROUND, hNum);
        startIndexW_ = -(this->Round(startW * wScale_));
        startIndexH_ = -(this->Round(startH * hScale_));
    } else {
        Cast(hIdexUb, hCooLocal, RoundMode::CAST_FLOOR, hNum);
        startIndexW_ = -(this->Floor(startW * wScale_));
        startIndexH_ = -(this->Floor(startH * hScale_));
    }

    Adds(hIdexUb, hIdexUb, startIndexH_, hNum);

    // 基于wNum的大小分3个场景分开写，提升性能;1、每次处理128; 2、处理64<n<128 3、处理<=64部分
    hNum_ = hNum;
    if (tiling_->splitCountDesW == 1) {
        srcWAlign_ = tiling_->lenSrcW;
        wNum_ = wNum;
    } else {
        srcWAlign_ = CeilDivision(static_cast<int32_t>(srcWFactor_), elementBlock_) * elementBlock_;
        wNum_ = CeilDivision(wNum, elementBlock_) * elementBlock_;
    }

    uint32_t loopW = wNum / vlSize_;
    wTail_ = wNum % vlSize_;

    if constexpr (IsSameType<T, float>::value) {
        // process main loop, 64 elements once
        for (uint32_t i = 0; i < loopW; ++i) {
            Compute32((__ubuf__ T *)yLocal[i * vlSize_].GetPhyAddr(),
                (__ubuf__ T *)xLocal.GetPhyAddr(),
                hIdexUb,
                static_cast<float>(i * vlSize_) + startW,
                vlSize_);
        }
        // process tail
        if (wTail_ > 0) {
            Compute32((__ubuf__ T *)yLocal[loopW * vlSize_].GetPhyAddr(),
                (__ubuf__ T *)xLocal.GetPhyAddr(),
                hIdexUb,
                static_cast<float>(loopW * vlSize_) + startW,
                wTail_);
        }
    } else {
        // process main loop, 128 elements once
        for (uint32_t i = 0; i < loopW; ++i) {
            ComputeMain((__ubuf__ T *)yLocal[i * vlSize_].GetPhyAddr(),
                (__ubuf__ T *)xLocal.GetPhyAddr(),
                hIdexUb,
                static_cast<float>(i * vlSize_) + startW,
                vlSize_);
        }
        // process tail data
        if (wTail_ > repeatElm_) {
            ComputeMain((__ubuf__ T *)yLocal[loopW * vlSize_].GetPhyAddr(),
                (__ubuf__ T *)xLocal.GetPhyAddr(),
                hIdexUb,
                static_cast<float>(loopW * vlSize_) + startW,
                wTail_);
        } else if (wTail_ > 0) {
            ComputeTail((__ubuf__ T *)yLocal[loopW * vlSize_].GetPhyAddr(),
                (__ubuf__ T *)xLocal.GetPhyAddr(),
                hIdexUb,
                static_cast<float>(loopW * vlSize_) + startW);
        }
    }

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void ResizeNearestNeighborV2Gather<T>::CopyOutWithPad(int64_t outOffset, int64_t hNum, int64_t wNum)
{
    LocalTensor<T> outLocal = outQueueY.DeQue<T>();

    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    if (tiling_->splitCountDesW == 1) {
        copyParams.blockLen = hNum * wNum * sizeof(T);
    } else {
        copyParams.blockCount = static_cast<uint16_t>(hNum);
        copyParams.blockLen = wNum * sizeof(T);
        copyParams.dstStride = (dstWSize_ - wNum) * sizeof(T);
    }
    DataCopyPad(yGm[blockOutOffset_ + outOffset], outLocal, copyParams);  // 一次搬出

    outQueueY.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void ResizeNearestNeighborV2Gather<T>::CopyInWithPad(int64_t inOffset)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();

    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    if (tiling_->splitCountDesW == 1) {
        copyParams.blockCount = 1;
        copyParams.blockLen = srcWSize_ * srcHFactor_ * sizeof(T);
    } else {
        copyParams.blockCount = static_cast<uint16_t>(srcHFactor_);
        copyParams.blockLen = srcWFactor_ * sizeof(T);
        copyParams.srcStride = (srcWSize_ - srcWFactor_) * sizeof(T);
    }
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[blockInOffset_ + inOffset], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}

}  // namespace ResizeNearestNeighborV2

#endif  // RESIZE_NEAREAST_NEIGHBOR_V2_GATHER_H
