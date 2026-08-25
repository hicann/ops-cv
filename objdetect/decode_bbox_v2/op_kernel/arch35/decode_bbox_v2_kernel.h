/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DECODE_BBOX_V2_KERNEL_H
#define DECODE_BBOX_V2_KERNEL_H

#include "kernel_operator.h"
#include "decode_bbox_v2_struct.h"
#include "decode_bbox_v2_tiling_struct.h"

constexpr int64_t kMaxInputSlots = 2;
constexpr int64_t kMaxOutputSlots = 1;
constexpr int64_t kElemsPerBox = 4;

constexpr uint32_t VL_F32 = 256U / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait kCastB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr AscendC::MicroAPI::CastTrait kCastB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__simd_vf__ inline void CastInVF(__ubuf__ float* dst, __ubuf__ half* src, uint32_t count, uint16_t repeatTimes);

__simd_vf__ inline void CastOutVF(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count, uint16_t repeatTimes);

template <typename T>
__simd_vf__ inline void SubVF(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint32_t count,
                              uint16_t repeatTimes);

template <typename T>
__simd_vf__ inline void DivsVF(__ubuf__ T* dst, __ubuf__ T* src, T scalar, uint32_t count, uint16_t repeatTimes);

template <typename T>
__simd_vf__ inline void ExpClipMulVF(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* ah, T clipVal, uint32_t count,
                                     uint16_t repeatTimes);

template <typename T>
__simd_vf__ inline void ExpMulVF(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* ah, uint32_t count,
                                 uint16_t repeatTimes);

template <typename T>
__simd_vf__ inline void CyVF(__ubuf__ T* dst, __ubuf__ T* tys, __ubuf__ T* ah, __ubuf__ T* aymin, T halfVal,
                             uint32_t count, uint16_t repeatTimes);

template <typename T>
__simd_vf__ inline void CornerVF(__ubuf__ T* dst0, __ubuf__ T* dst1, __ubuf__ T* cy, __ubuf__ T* h, T halfVal,
                                 uint32_t count, uint16_t repeatTimes);

template <typename T, int LAYOUT>
class DecodeBboxV2Kernel {
    static constexpr bool NEED_CAST = !std::is_same<T, float>::value;
    static constexpr int64_t kNumIoBufs = 3;
    static constexpr int64_t kNumCalcBufs = NEED_CAST ? 3 : 0;
    static constexpr int64_t kPhysNodes = kNumIoBufs + kNumCalcBufs;

    AscendC::TPipe pipe_;
    const DecodeBboxV2TilingData* td_;
    AscendC::GlobalTensor<T> gmIn_[kMaxInputSlots];
    AscendC::GlobalTensor<T> gmOut_[kMaxOutputSlots];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];

    int32_t evMte2ToV_ = 0;
    int32_t evVToMte3_ = 0;
    int32_t evMte3ToMte2_ = 0;
    int32_t evMte3ToS_ = 0;

public:
    __aicore__ inline DecodeBboxV2Kernel() {}

    __aicore__ inline void Init(GM_ADDR inputs[kMaxInputSlots], GM_ADDR outputs[kMaxOutputSlots],
                                const DecodeBboxV2TilingData* td)
    {
        td_ = td;
        if (td_->ubFormer <= 0) {
            return;
        }
        for (int i = 0; i < kMaxInputSlots; i++) {
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        }
        for (int i = 0; i < kMaxOutputSlots; i++) {
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        }
        int64_t ioBufBytes = td_->ubFormer * kElemsPerBox * sizeof(T);
        int64_t calcBufBytes = td_->ubFormer * kElemsPerBox * sizeof(float);
        if (ioBufBytes > 0) {
            for (int64_t i = 0; i < kNumIoBufs; i++) {
                pipe_.InitBuffer(buf_[i], ioBufBytes);
            }
        }
        if constexpr (kNumCalcBufs > 0) {
            if (calcBufBytes > 0) {
                for (int64_t i = kNumIoBufs; i < kNumIoBufs + kNumCalcBufs; i++) {
                    pipe_.InitBuffer(buf_[i], calcBufBytes);
                }
            }
        }
        evMte2ToV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        evVToMte3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        evMte3ToMte2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        evMte3ToS_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    }

    __aicore__ inline void Process()
    {
        if (td_->blockNum <= 0) {
            return;
        }
        int64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= td_->blockNum) {
            return;
        }
        bool isLastBlock = (blockIdx == td_->blockNum - 1);
        int64_t loopNum = isLastBlock ? td_->ubLoopOfTailBlock : td_->ubLoopOfFormerBlock;
        int64_t tailNum = isLastBlock ? td_->ubTailOfTailBlock : td_->ubTailOfFormerBlock;
        int64_t boxOffset = td_->blockFormer * blockIdx;

        int64_t fullTiles = (tailNum > 0) ? (loopNum - 1) : loopNum;
        int64_t totalTiles = loopNum;

        int64_t tileIdx = 0;
        for (int64_t i = 0; i < fullTiles; i++) {
            bool notFirst = (tileIdx != 0);
            bool notLast = (tileIdx != totalTiles - 1);
            ProcessTile(boxOffset, td_->ubFormer, notFirst, notLast);
            boxOffset += td_->ubFormer;
            tileIdx++;
        }
        if (tailNum > 0) {
            bool notFirst = (tileIdx != 0);
            bool notLast = (tileIdx != totalTiles - 1);
            ProcessTile(boxOffset, tailNum, notFirst, notLast);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evMte3ToS_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evMte3ToS_);
    }

private:
    __aicore__ inline void ProcessTile(int64_t boxOffset, int64_t boxCount, bool notFirst, bool notLast)
    {
        if (notFirst)
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3ToMte2_);
        CopyIn(boxOffset, boxCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2ToV_);
        Compute(boxCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVToMte3_);
        CopyOut(boxOffset, boxCount);
        if (notLast)
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3ToMte2_);
    }

    __aicore__ inline int64_t ChanStride() const { return td_->ubFormer; }

    static __aicore__ inline uint16_t CalcRep(uint32_t count)
    {
        return static_cast<uint16_t>((count + VL_F32 - 1) / VL_F32);
    }

    __aicore__ inline void CopyIn(int64_t boxOffset, int64_t boxCount)
    {
        AscendC::DataCopyPadExtParams<T> pad(false, 0, 0, 0);

        int64_t N = boxCount;
        int64_t stride = ChanStride();
        int64_t elemBase = boxOffset * kElemsPerBox;
        const uint32_t cap = static_cast<uint32_t>(stride);

        if constexpr (LAYOUT == DECODE_BBOX_V2_LAYOUT_N4) {
            AscendC::DataCopyExtParams params(static_cast<uint16_t>(N), static_cast<uint32_t>(sizeof(T)),
                                              static_cast<int64_t>((kElemsPerBox - 1) * sizeof(T)), 0, 0);
            auto boxesUb = buf_[0].template Get<T>();
            for (int c = 0; c < kElemsPerBox; c++) {
                AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(boxesUb[static_cast<uint32_t>(c) * cap],
                                                                       gmIn_[0][elemBase + c], params, pad);
            }
            auto ancUb = buf_[1].template Get<T>();
            for (int c = 0; c < kElemsPerBox; c++) {
                AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(ancUb[static_cast<uint32_t>(c) * cap],
                                                                       gmIn_[1][elemBase + c], params, pad);
            }
        } else {
            AscendC::DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(N * sizeof(T));
            params.srcStride = 0;
            params.dstStride = 0;
            auto boxesUb = buf_[0].template Get<T>();
            for (int c = 0; c < kElemsPerBox; c++) {
                AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
                    boxesUb[static_cast<uint32_t>(c) * cap], gmIn_[0][c * td_->dim0 + boxOffset], params, pad);
            }
            auto ancUb = buf_[1].template Get<T>();
            for (int c = 0; c < kElemsPerBox; c++) {
                AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
                    ancUb[static_cast<uint32_t>(c) * cap], gmIn_[1][c * td_->dim0 + boxOffset], params, pad);
            }
        }
    }

    __aicore__ inline void Compute(int64_t boxCount)
    {
        if constexpr (NEED_CAST) {
            ComputeFp16(boxCount);
        } else {
            ComputeFp32(boxCount);
        }
    }

    __aicore__ inline void ComputeFp32(int64_t boxCount)
    {
        int64_t s = ChanStride();

        __ubuf__ float* boxes = (__ubuf__ float*)buf_[0].template Get<float>().GetPhyAddr();
        __ubuf__ float* anchors = (__ubuf__ float*)buf_[1].template Get<float>().GetPhyAddr();
        __ubuf__ float* out = (__ubuf__ float*)buf_[2].template Get<float>().GetPhyAddr();

        __ubuf__ float* ty = boxes + 0 * s;
        __ubuf__ float* tx = boxes + 1 * s;
        __ubuf__ float* th = boxes + 2 * s;
        __ubuf__ float* tw = boxes + 3 * s;
        __ubuf__ float* aymin = anchors + 0 * s;
        __ubuf__ float* axmin = anchors + 1 * s;
        __ubuf__ float* aymax = anchors + 2 * s;
        __ubuf__ float* axmax = anchors + 3 * s;
        __ubuf__ float* ymin = out + 0 * s;
        __ubuf__ float* xmin = out + 1 * s;
        __ubuf__ float* ymax = out + 2 * s;
        __ubuf__ float* xmax = out + 3 * s;

        uint32_t total = static_cast<uint32_t>(boxCount);
        uint16_t rep = CalcRep(total);

        asc_vf_call<SubVF<float>>(aymax, aymax, aymin, total, rep);
        asc_vf_call<SubVF<float>>(axmax, axmax, axmin, total, rep);
        asc_vf_call<DivsVF<float>>(ty, ty, td_->scales[0], total, rep);
        asc_vf_call<DivsVF<float>>(tx, tx, td_->scales[1], total, rep);
        asc_vf_call<DivsVF<float>>(th, th, td_->scales[2], total, rep);
        asc_vf_call<DivsVF<float>>(tw, tw, td_->scales[3], total, rep);
        if (td_->decodeClip > 0.0f) {
            asc_vf_call<ExpClipMulVF<float>>(th, th, aymax, td_->decodeClip, total, rep);
        } else {
            asc_vf_call<ExpMulVF<float>>(th, th, aymax, total, rep);
        }
        if (td_->decodeClip > 0.0f) {
            asc_vf_call<ExpClipMulVF<float>>(tw, tw, axmax, td_->decodeClip, total, rep);
        } else {
            asc_vf_call<ExpMulVF<float>>(tw, tw, axmax, total, rep);
        }
        asc_vf_call<CyVF<float>>(ty, ty, aymax, aymin, td_->halfVal, total, rep);
        asc_vf_call<CyVF<float>>(tx, tx, axmax, axmin, td_->halfVal, total, rep);
        asc_vf_call<CornerVF<float>>(ymin, ymax, ty, th, td_->halfVal, total, rep);
        asc_vf_call<CornerVF<float>>(xmin, xmax, tx, tw, td_->halfVal, total, rep);
    }

    __aicore__ inline void ComputeFp16(int64_t boxCount)
    {
        int64_t N = boxCount;
        int64_t s = ChanStride();
        int64_t castTotal = s * kElemsPerBox;

        __ubuf__ half* boxesFp16 = (__ubuf__ half*)buf_[0].template Get<T>().GetPhyAddr();
        __ubuf__ half* anchorsFp16 = (__ubuf__ half*)buf_[1].template Get<T>().GetPhyAddr();
        __ubuf__ float* C0 = (__ubuf__ float*)buf_[3].template Get<float>().GetPhyAddr();
        __ubuf__ float* C1 = (__ubuf__ float*)buf_[4].template Get<float>().GetPhyAddr();
        __ubuf__ float* C2 = (__ubuf__ float*)buf_[5].template Get<float>().GetPhyAddr();

        uint32_t castCount = static_cast<uint32_t>(castTotal);
        uint16_t castRep = CalcRep(castCount);
        asc_vf_call<CastInVF>(C0, boxesFp16, castCount, castRep);
        asc_vf_call<CastInVF>(C1, anchorsFp16, castCount, castRep);

        __ubuf__ float* c0ty = C0 + 0 * s;
        __ubuf__ float* c0tx = C0 + 1 * s;
        __ubuf__ float* c0th = C0 + 2 * s;
        __ubuf__ float* c0tw = C0 + 3 * s;
        __ubuf__ float* c1aymin = C1 + 0 * s;
        __ubuf__ float* c1axmin = C1 + 1 * s;
        __ubuf__ float* c1aymax = C1 + 2 * s;
        __ubuf__ float* c1axmax = C1 + 3 * s;

        uint32_t total = static_cast<uint32_t>(N);
        uint16_t rep = CalcRep(total);

        asc_vf_call<SubVF<float>>(C2 + 0 * s, c1aymax, c1aymin, total, rep);
        asc_vf_call<DivsVF<float>>(C2 + 1 * s, c0ty, td_->scales[0], total, rep);
        asc_vf_call<DivsVF<float>>(C2 + 2 * s, c0th, td_->scales[2], total, rep);
        if (td_->decodeClip > 0.0f) {
            asc_vf_call<ExpClipMulVF<float>>(C2 + 2 * s, C2 + 2 * s, C2 + 0 * s, td_->decodeClip, total, rep);
        } else {
            asc_vf_call<ExpMulVF<float>>(C2 + 2 * s, C2 + 2 * s, C2 + 0 * s, total, rep);
        }
        asc_vf_call<CyVF<float>>(C2 + 3 * s, C2 + 1 * s, C2 + 0 * s, c1aymin, td_->halfVal, total, rep);
        asc_vf_call<CornerVF<float>>(c0ty, c0th, C2 + 3 * s, C2 + 2 * s, td_->halfVal, total, rep);

        asc_vf_call<SubVF<float>>(C2 + 0 * s, c1axmax, c1axmin, total, rep);
        asc_vf_call<DivsVF<float>>(C2 + 1 * s, c0tx, td_->scales[1], total, rep);
        asc_vf_call<DivsVF<float>>(C2 + 2 * s, c0tw, td_->scales[3], total, rep);
        if (td_->decodeClip > 0.0f) {
            asc_vf_call<ExpClipMulVF<float>>(C2 + 2 * s, C2 + 2 * s, C2 + 0 * s, td_->decodeClip, total, rep);
        } else {
            asc_vf_call<ExpMulVF<float>>(C2 + 2 * s, C2 + 2 * s, C2 + 0 * s, total, rep);
        }
        asc_vf_call<CyVF<float>>(C2 + 3 * s, C2 + 1 * s, C2 + 0 * s, c1axmin, td_->halfVal, total, rep);
        asc_vf_call<CornerVF<float>>(c0tx, c0tw, C2 + 3 * s, C2 + 2 * s, td_->halfVal, total, rep);

        __ubuf__ half* outFp16 = (__ubuf__ half*)buf_[2].template Get<T>().GetPhyAddr();
        asc_vf_call<CastOutVF>(outFp16, C0, castCount, castRep);
    }

    __aicore__ inline void CopyOut(int64_t boxOffset, int64_t boxCount)
    {
        int64_t N = boxCount;
        int64_t s = ChanStride();
        int64_t elemBase = boxOffset * kElemsPerBox;
        auto outUb = buf_[2].template Get<T>();
        const uint32_t cap = static_cast<uint32_t>(s);

        if constexpr (LAYOUT == DECODE_BBOX_V2_LAYOUT_N4) {
            AscendC::DataCopyExtParams params(static_cast<uint16_t>(N), static_cast<uint32_t>(sizeof(T)), 0,
                                              static_cast<int64_t>((kElemsPerBox - 1) * sizeof(T)), 0);
            for (int c = 0; c < kElemsPerBox; c++) {
                AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(gmOut_[0][elemBase + c],
                                                                       outUb[static_cast<uint32_t>(c) * cap], params);
            }
        } else {
            AscendC::DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(N * sizeof(T));
            params.srcStride = 0;
            params.dstStride = 0;
            for (int c = 0; c < kElemsPerBox; c++) {
                AscendC::DataCopyPad(gmOut_[0][c * td_->dim0 + boxOffset], outUb[c * s], params);
            }
        }
    }
};

__simd_vf__ inline void CastInVF(__ubuf__ float* dst, __ubuf__ half* src, uint32_t count, uint16_t repeatTimes)
{
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::RegTensor<half> hReg;
        AscendC::Reg::RegTensor<float> fReg;
        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(hReg, src + off);
        AscendC::Reg::Cast<float, half, kCastB162B32>(fReg, hReg, mask);
        AscendC::Reg::StoreAlign(dst + off, fReg, mask);
    }
}

__simd_vf__ inline void CastOutVF(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count, uint16_t repeatTimes)
{
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::RegTensor<float> fReg;
        AscendC::Reg::RegTensor<half> hReg;
        AscendC::Reg::LoadAlign(fReg, src + off);
        AscendC::Reg::Cast<half, float, kCastB322B16>(hReg, fReg, mask);
        AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + off, hReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void DivsVF(__ubuf__ T* dst, __ubuf__ T* src, T scalar, uint32_t count, uint16_t repeatTimes)
{
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::RegTensor<T> srcReg, scalarReg, dstReg;
        AscendC::Reg::LoadAlign(srcReg, src + off);
        vdup(scalarReg, scalar, mask, MODE_ZEROING);
        vdiv(dstReg, srcReg, scalarReg, mask, MODE_ZEROING);
        AscendC::Reg::StoreAlign(dst + off, dstReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void SubVF(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint32_t count, uint16_t repeatTimes)
{
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::RegTensor<T> src0Reg, src1Reg, dstReg;
        AscendC::Reg::LoadAlign(src0Reg, src0 + off);
        AscendC::Reg::LoadAlign(src1Reg, src1 + off);
        AscendC::Reg::Sub(dstReg, src0Reg, src1Reg, mask);
        AscendC::Reg::StoreAlign(dst + off, dstReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void ExpClipMulVF(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* ah, T clipVal, uint32_t count,
                                     uint16_t repeatTimes)
{
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::RegTensor<T> srcReg, ahReg, clipReg, expReg, dstReg;
        AscendC::Reg::LoadAlign(srcReg, src + off);
        AscendC::Reg::LoadAlign(ahReg, ah + off);
        AscendC::Reg::Mins(clipReg, srcReg, clipVal, mask);
        AscendC::Reg::Exp(expReg, clipReg, mask);
        AscendC::Reg::Mul(dstReg, expReg, ahReg, mask);
        AscendC::Reg::StoreAlign(dst + off, dstReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void ExpMulVF(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* ah, uint32_t count, uint16_t repeatTimes)
{
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::RegTensor<T> srcReg, ahReg, expReg, dstReg;
        AscendC::Reg::LoadAlign(srcReg, src + off);
        AscendC::Reg::LoadAlign(ahReg, ah + off);
        AscendC::Reg::Exp(expReg, srcReg, mask);
        AscendC::Reg::Mul(dstReg, expReg, ahReg, mask);
        AscendC::Reg::StoreAlign(dst + off, dstReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void CyVF(__ubuf__ T* dst, __ubuf__ T* tys, __ubuf__ T* ah, __ubuf__ T* aymin, T halfVal,
                             uint32_t count, uint16_t repeatTimes)
{
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::RegTensor<T> tysReg, ahReg, ayminReg, t1Reg, ahHalfReg, dstReg;
        AscendC::Reg::LoadAlign(tysReg, tys + off);
        AscendC::Reg::LoadAlign(ahReg, ah + off);
        AscendC::Reg::LoadAlign(ayminReg, aymin + off);
        AscendC::Reg::Mul(t1Reg, tysReg, ahReg, mask);
        AscendC::Reg::Add(t1Reg, t1Reg, ayminReg, mask);
        AscendC::Reg::Muls(ahHalfReg, ahReg, halfVal, mask);
        AscendC::Reg::Add(dstReg, t1Reg, ahHalfReg, mask);
        AscendC::Reg::StoreAlign(dst + off, dstReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void CornerVF(__ubuf__ T* dst0, __ubuf__ T* dst1, __ubuf__ T* cy, __ubuf__ T* h, T halfVal,
                                 uint32_t count, uint16_t repeatTimes)
{
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t off = static_cast<uint32_t>(i) * VL_F32;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::RegTensor<T> cyReg, hReg, hHalfReg, y0Reg, y1Reg;
        AscendC::Reg::LoadAlign(cyReg, cy + off);
        AscendC::Reg::LoadAlign(hReg, h + off);
        AscendC::Reg::Muls(hHalfReg, hReg, halfVal, mask);
        AscendC::Reg::Sub(y0Reg, cyReg, hHalfReg, mask);
        AscendC::Reg::Add(y1Reg, cyReg, hHalfReg, mask);
        AscendC::Reg::StoreAlign(dst0 + off, y0Reg, mask);
        AscendC::Reg::StoreAlign(dst1 + off, y1Reg, mask);
    }
}

#endif
