/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file c_io_u.h
 * \brief CIoU 2-output forward kernel.
 *
 * Inputs:  bboxes, gtboxes  (4, N), dtype in {fp32, fp16}
 * Outputs: overlap          (1, N), same dtype
 *          atan_sub         (1, N), same dtype
 * Attrs:   trans (xyxy/cxcywh), is_cross (false only), mode (iou/iof), atan_sub_flag
 *
 * Compute is fp32 throughout; cast at I/O for fp16.
 * Per-core partition is cache-line aligned (32B) on the (1,N) outputs to
 * eliminate write-conflict between cores; the last core absorbs the tail.
 */

#ifndef __C_IO_U_H__
#define __C_IO_U_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "c_io_u_tiling_data.h"
#include "c_io_u_tiling_key.h"
#include <type_traits>

namespace NsCIoU {

using namespace AscendC;

template <typename T>
class CIoU {
private:
    AscendC::TPipe* pipe;

    // Reuse one input queue per source tensor.  Each coordinate is copied and
    // cast into TBuf before the next coordinate reuses the queue.
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inAQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inBQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outOverlapQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outAtanSubQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> a0Buf, a1Buf, a2Buf, a3Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0Buf, b1Buf, b2Buf, b3Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1aBuf, y1aBuf, x2aBuf, y2aBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1bBuf, y1bBuf, x2bBuf, y2bBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> w1Buf, h1Buf, w2Buf, h2Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp1Buf, tmp2Buf, tmp3Buf, tmp4Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> overlapBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> atanSubBuf;
    // Atan workspace: hardware Atan (fp32) needs 5x src bytes (ATAN_FLOAT_CALC_PROCEDURE).
    AscendC::TBuf<AscendC::TPosition::VECCALC> atanWsBuf;

    AscendC::GlobalTensor<T> bboxesGm;
    AscendC::GlobalTensor<T> gtboxesGm;
    AscendC::GlobalTensor<T> overlapGm;
    AscendC::GlobalTensor<T> atanSubGm;

    uint32_t totalN, basePerCore, pivot, tileN, usedCoreNum, alignElem, tailN;
    int32_t trans, modeId, atanSubFlag;
    float eps;

    uint32_t programId, myN, myStart, innerLoops;

public:
    __aicore__ inline CIoU() {}

    __aicore__ inline void Init(GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR overlap, GM_ADDR atanSub,
                                const CIoUTilingData* td, AscendC::TPipe* pipePtr)
    {
        this->pipe = pipePtr;
        InitTilingFields(td);
        InitCoreRange();
        InitGlobalBuffers(bboxes, gtboxes, overlap, atanSub);
        uint32_t qBytes = ((tileN * sizeof(T) + 31u) / 32u) * 32u;
        uint32_t fBytes = CalcFloatBufferBytes();
        InitQueues(qBytes);
        InitComputeBuffers(fBytes);
    }

    __aicore__ inline void Process()
    {
        if (myN == 0)
            return;
        for (uint32_t i = 0; i < innerLoops; i++) {
            uint32_t tileOffset = i * tileN;
            if (tileOffset >= myN)
                break;
            uint32_t cnt = (tileOffset + tileN <= myN) ? tileN : (myN - tileOffset);
            Compute(myStart + tileOffset, cnt);
            CopyOut(i, cnt);
        }
    }

private:
    struct ComputeLocal {
        AscendC::LocalTensor<float> a0, a1, a2, a3;
        AscendC::LocalTensor<float> b0, b1, b2, b3;
        AscendC::LocalTensor<float> x1a, y1a, x2a, y2a;
        AscendC::LocalTensor<float> x1b, y1b, x2b, y2b;
        AscendC::LocalTensor<float> w1, h1, w2, h2;
        AscendC::LocalTensor<float> tmp1, tmp2, tmp3, tmp4;
        AscendC::LocalTensor<float> overlap, atanSub;
    };

    __aicore__ inline void InitTilingFields(const CIoUTilingData* td)
    {
        this->totalN = td->totalN;
        this->basePerCore = td->basePerCore;
        this->pivot = td->pivot;
        this->tileN = td->tileN;
        this->usedCoreNum = td->usedCoreNum;
        this->alignElem = td->alignElem;
        this->tailN = td->tailN;
        this->trans = td->trans;
        this->modeId = td->modeId;
        this->atanSubFlag = td->atanSubFlag;
        this->eps = td->eps;
    }

    __aicore__ inline void InitCoreRange()
    {
        this->programId = AscendC::GetBlockIdx();
        uint32_t extra = (programId < pivot) ? alignElem : 0u;
        this->myN = basePerCore + extra;
        this->myStart = programId * basePerCore + ((programId < pivot) ? programId : pivot) * alignElem;
        if (programId == usedCoreNum - 1) {
            this->myN += tailN;
        }
        this->innerLoops = (myN == 0) ? 0u : (myN + tileN - 1) / tileN;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR overlap, GM_ADDR atanSub)
    {
        bboxesGm.SetGlobalBuffer((__gm__ T*)bboxes, totalN * 4);
        gtboxesGm.SetGlobalBuffer((__gm__ T*)gtboxes, totalN * 4);
        overlapGm.SetGlobalBuffer((__gm__ T*)overlap, totalN);
        atanSubGm.SetGlobalBuffer((__gm__ T*)atanSub, totalN);
    }

    __aicore__ inline uint32_t CalcFloatBufferBytes() const
    {
        uint32_t castAligned = ((tileN + 15u) / 16u) * 16u * sizeof(float);
        uint32_t padAligned = ((tileN * sizeof(float) + 31u) / 32u) * 32u;
        return (castAligned > padAligned) ? castAligned : padAligned;
    }

    __aicore__ inline void InitQueues(uint32_t qBytes)
    {
        pipe->InitBuffer(inAQueue, 1, qBytes);
        pipe->InitBuffer(inBQueue, 1, qBytes);
        pipe->InitBuffer(outOverlapQueue, 1, qBytes);
        pipe->InitBuffer(outAtanSubQueue, 1, qBytes);
    }

    __aicore__ inline void InitComputeBuffers(uint32_t fBytes)
    {
        pipe->InitBuffer(a0Buf, fBytes);
        pipe->InitBuffer(a1Buf, fBytes);
        pipe->InitBuffer(a2Buf, fBytes);
        pipe->InitBuffer(a3Buf, fBytes);
        pipe->InitBuffer(b0Buf, fBytes);
        pipe->InitBuffer(b1Buf, fBytes);
        pipe->InitBuffer(b2Buf, fBytes);
        pipe->InitBuffer(b3Buf, fBytes);
        pipe->InitBuffer(x1aBuf, fBytes);
        pipe->InitBuffer(y1aBuf, fBytes);
        pipe->InitBuffer(x2aBuf, fBytes);
        pipe->InitBuffer(y2aBuf, fBytes);
        pipe->InitBuffer(x1bBuf, fBytes);
        pipe->InitBuffer(y1bBuf, fBytes);
        pipe->InitBuffer(x2bBuf, fBytes);
        pipe->InitBuffer(y2bBuf, fBytes);
        pipe->InitBuffer(w1Buf, fBytes);
        pipe->InitBuffer(h1Buf, fBytes);
        pipe->InitBuffer(w2Buf, fBytes);
        pipe->InitBuffer(h2Buf, fBytes);
        pipe->InitBuffer(tmp1Buf, fBytes);
        pipe->InitBuffer(tmp2Buf, fBytes);
        pipe->InitBuffer(tmp3Buf, fBytes);
        pipe->InitBuffer(tmp4Buf, fBytes);
        pipe->InitBuffer(overlapBuf, fBytes);
        pipe->InitBuffer(atanSubBuf, fBytes);
        if (atanSubFlag) {
            pipe->InitBuffer(atanWsBuf, 5u * fBytes);
        }
    }

    template <int BUF_NUM>
    __aicore__ inline void CopyInOne(AscendC::TQue<AscendC::TPosition::VECIN, BUF_NUM>& q, AscendC::GlobalTensor<T>& gm,
                                     uint32_t gmOffset, uint32_t cnt)
    {
        AscendC::LocalTensor<T> local = q.template AllocTensor<T>();
        uint32_t blockLen = cnt * sizeof(T);
        uint32_t paddedLen = ((blockLen + 31u) / 32u) * 32u;
        uint8_t rightPad = static_cast<uint8_t>((paddedLen - blockLen) / sizeof(T));
        AscendC::DataCopyPad(local, gm[gmOffset], {1, static_cast<uint16_t>(blockLen), 0, 0},
                             {true, static_cast<uint8_t>(0), rightPad, static_cast<uint64_t>(0)});
        q.EnQue(local);
    }

    template <int BUF_NUM>
    __aicore__ inline void DequeueCast(AscendC::TQue<AscendC::TPosition::VECIN, BUF_NUM>& q,
                                       AscendC::LocalTensor<float>& dstF, uint32_t cnt)
    {
        AscendC::LocalTensor<T> local = q.template DeQue<T>();
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Adds(dstF, local, 0.0f, cnt);
        } else {
            AscendC::Cast(dstF, local, AscendC::RoundMode::CAST_NONE, cnt);
        }
        q.FreeTensor(local);
    }

    template <int BUF_NUM>
    __aicore__ inline void EnqueueCast(AscendC::TQue<AscendC::TPosition::VECOUT, BUF_NUM>& outQ,
                                       AscendC::LocalTensor<float>& src, uint32_t cnt)
    {
        AscendC::LocalTensor<T> outLocal = outQ.template AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Adds(outLocal, src, 0.0f, cnt);
        } else {
            AscendC::Cast(outLocal, src, AscendC::RoundMode::CAST_NONE, cnt);
        }
        outQ.EnQue(outLocal);
    }

    __aicore__ inline void CopyInPair(uint32_t tileStart, uint32_t coord, AscendC::LocalTensor<float>& dstA,
                                      AscendC::LocalTensor<float>& dstB, uint32_t cnt)
    {
        CopyInOne(inAQueue, bboxesGm, coord * totalN + tileStart, cnt);
        CopyInOne(inBQueue, gtboxesGm, coord * totalN + tileStart, cnt);
        DequeueCast(inAQueue, dstA, cnt);
        DequeueCast(inBQueue, dstB, cnt);
    }

    __aicore__ inline void CopyInAndCast(ComputeLocal& t, uint32_t tileStart, uint32_t cnt)
    {
        CopyInPair(tileStart, 0u, t.a0, t.b0, cnt);
        CopyInPair(tileStart, 1u, t.a1, t.b1, cnt);
        CopyInPair(tileStart, 2u, t.a2, t.b2, cnt);
        CopyInPair(tileStart, 3u, t.a3, t.b3, cnt);
    }

    __aicore__ inline void Compute(uint32_t tileStart, uint32_t cnt)
    {
        ComputeLocal tensors = GetComputeLocal();
        CopyInAndCast(tensors, tileStart, cnt);
        PipeBarrier<PIPE_V>();
        ResolveCorners(tensors, cnt);
        PipeBarrier<PIPE_V>();
        ComputeOverlap(tensors, cnt);
        PipeBarrier<PIPE_V>();
        ComputeAtanSub(tensors, cnt);
        PipeBarrier<PIPE_V>();
        ComputeCenterPenalty(tensors, cnt);
        PipeBarrier<PIPE_V>();
        EnqueueCast<1>(outOverlapQueue, tensors.overlap, cnt);
        EnqueueCast<1>(outAtanSubQueue, tensors.atanSub, cnt);
    }

    __aicore__ inline ComputeLocal GetComputeLocal()
    {
        return {a0Buf.template Get<float>(),      a1Buf.template Get<float>(),     a2Buf.template Get<float>(),
                a3Buf.template Get<float>(),      b0Buf.template Get<float>(),     b1Buf.template Get<float>(),
                b2Buf.template Get<float>(),      b3Buf.template Get<float>(),     x1aBuf.template Get<float>(),
                y1aBuf.template Get<float>(),     x2aBuf.template Get<float>(),    y2aBuf.template Get<float>(),
                x1bBuf.template Get<float>(),     y1bBuf.template Get<float>(),    x2bBuf.template Get<float>(),
                y2bBuf.template Get<float>(),     w1Buf.template Get<float>(),     h1Buf.template Get<float>(),
                w2Buf.template Get<float>(),      h2Buf.template Get<float>(),     tmp1Buf.template Get<float>(),
                tmp2Buf.template Get<float>(),    tmp3Buf.template Get<float>(),   tmp4Buf.template Get<float>(),
                overlapBuf.template Get<float>(), atanSubBuf.template Get<float>()};
    }

    __aicore__ inline void ResolveCorners(ComputeLocal& t, uint32_t cnt)
    {
        if (trans == 1) {
            ResolveTransCorners(t, cnt);
        } else {
            ResolveXyxyCorners(t, cnt);
        }
    }

    __aicore__ inline void ResolveTransCorners(ComputeLocal& t, uint32_t cnt)
    {
        AscendC::Maxs(t.w1, t.a2, 0.0f, cnt);
        AscendC::Maxs(t.h1, t.a3, 0.0f, cnt);
        AscendC::Maxs(t.w2, t.b2, 0.0f, cnt);
        AscendC::Maxs(t.h2, t.b3, 0.0f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Muls(t.tmp1, t.w1, 0.5f, cnt);
        AscendC::Muls(t.tmp2, t.h1, 0.5f, cnt);
        AscendC::Muls(t.tmp3, t.w2, 0.5f, cnt);
        AscendC::Muls(t.tmp4, t.h2, 0.5f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(t.x1a, t.a0, t.tmp1, cnt);
        AscendC::Add(t.x2a, t.a0, t.tmp1, cnt);
        AscendC::Sub(t.y1a, t.a1, t.tmp2, cnt);
        AscendC::Add(t.y2a, t.a1, t.tmp2, cnt);
        AscendC::Sub(t.x1b, t.b0, t.tmp3, cnt);
        AscendC::Add(t.x2b, t.b0, t.tmp3, cnt);
        AscendC::Sub(t.y1b, t.b1, t.tmp4, cnt);
        AscendC::Add(t.y2b, t.b1, t.tmp4, cnt);
    }

    __aicore__ inline void ResolveXyxyCorners(ComputeLocal& t, uint32_t cnt)
    {
        AscendC::Adds(t.x1a, t.a0, 0.0f, cnt);
        AscendC::Adds(t.y1a, t.a1, 0.0f, cnt);
        AscendC::Adds(t.x2a, t.a2, 0.0f, cnt);
        AscendC::Adds(t.y2a, t.a3, 0.0f, cnt);
        AscendC::Adds(t.x1b, t.b0, 0.0f, cnt);
        AscendC::Adds(t.y1b, t.b1, 0.0f, cnt);
        AscendC::Adds(t.x2b, t.b2, 0.0f, cnt);
        AscendC::Adds(t.y2b, t.b3, 0.0f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(t.tmp1, t.x2a, t.x1a, cnt);
        AscendC::Sub(t.tmp2, t.y2a, t.y1a, cnt);
        AscendC::Sub(t.tmp3, t.x2b, t.x1b, cnt);
        AscendC::Sub(t.tmp4, t.y2b, t.y1b, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Maxs(t.w1, t.tmp1, 0.0f, cnt);
        AscendC::Maxs(t.h1, t.tmp2, 0.0f, cnt);
        AscendC::Maxs(t.w2, t.tmp3, 0.0f, cnt);
        AscendC::Maxs(t.h2, t.tmp4, 0.0f, cnt);
    }

    __aicore__ inline void ComputeOverlap(ComputeLocal& t, uint32_t cnt)
    {
        AscendC::Min(t.tmp1, t.x2a, t.x2b, cnt);
        AscendC::Max(t.tmp2, t.x1a, t.x1b, cnt);
        AscendC::Min(t.tmp3, t.y2a, t.y2b, cnt);
        AscendC::Max(t.tmp4, t.y1a, t.y1b, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(t.tmp1, t.tmp1, t.tmp2, cnt);
        AscendC::Sub(t.tmp2, t.tmp3, t.tmp4, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Maxs(t.tmp1, t.tmp1, 0.0f, cnt);
        AscendC::Maxs(t.tmp2, t.tmp2, 0.0f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(t.tmp1, t.tmp1, t.tmp2, cnt);
        AscendC::Mul(t.tmp3, t.w2, t.h2, cnt);
        PipeBarrier<PIPE_V>();
        if (modeId == 1) {
            AscendC::Adds(t.tmp4, t.tmp3, eps, cnt);
        } else {
            AscendC::Mul(t.tmp2, t.w1, t.h1, cnt);
            PipeBarrier<PIPE_V>();
            AscendC::Add(t.tmp4, t.tmp2, t.tmp3, cnt);
            PipeBarrier<PIPE_V>();
            AscendC::Sub(t.tmp4, t.tmp4, t.tmp1, cnt);
            PipeBarrier<PIPE_V>();
            AscendC::Adds(t.tmp4, t.tmp4, eps, cnt);
        }
        PipeBarrier<PIPE_V>();
        AscendC::Maxs(t.tmp4, t.tmp4, 1e-4f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Div(t.overlap, t.tmp1, t.tmp4, cnt);
    }

    __aicore__ inline void ApplyAtanPenalty(ComputeLocal& t, uint32_t cnt)
    {
        if (modeId != 0) {
            return;
        }
        AscendC::Mul(t.tmp1, t.atanSub, t.atanSub, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Muls(t.tmp1, t.tmp1, 0.4052847345693511f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Adds(t.tmp2, t.tmp1, 1.0f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(t.tmp2, t.tmp2, t.overlap, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Maxs(t.tmp2, t.tmp2, 1e-4f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Div(t.tmp3, t.tmp1, t.tmp2, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(t.tmp3, t.tmp3, t.tmp1, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(t.overlap, t.overlap, t.tmp3, cnt);
    }

    __aicore__ inline void ComputeAtanSub(ComputeLocal& t, uint32_t cnt)
    {
        if (atanSubFlag) {
            AscendC::Adds(t.tmp2, t.h1, eps, cnt);
            AscendC::Adds(t.tmp3, t.h2, eps, cnt);
            PipeBarrier<PIPE_V>();
            AscendC::Maxs(t.tmp2, t.tmp2, 1e-4f, cnt);
            AscendC::Maxs(t.tmp3, t.tmp3, 1e-4f, cnt);
            PipeBarrier<PIPE_V>();
            AscendC::Div(t.tmp2, t.w1, t.tmp2, cnt);
            AscendC::Div(t.tmp3, t.w2, t.tmp3, cnt);
            AscendC::LocalTensor<uint8_t> atanWs = atanWsBuf.template Get<uint8_t>();
            PipeBarrier<PIPE_V>();
            AscendC::Adds(t.tmp1, t.tmp2, 0.0f, cnt);
            PipeBarrier<PIPE_V>();
            AscendC::Atan(t.tmp4, t.tmp1, atanWs, cnt);
            PipeBarrier<PIPE_V>();
            AscendC::Adds(t.tmp1, t.tmp3, 0.0f, cnt);
            PipeBarrier<PIPE_V>();
            AscendC::Atan(t.tmp3, t.tmp1, atanWs, cnt);
            PipeBarrier<PIPE_V>();
            AscendC::Sub(t.atanSub, t.tmp4, t.tmp3, cnt);
            PipeBarrier<PIPE_V>();
            ApplyAtanPenalty(t, cnt);
        } else {
            AscendC::Duplicate(t.atanSub, 0.0f, cnt);
        }
    }

    __aicore__ inline void ComputeCenterPenalty(ComputeLocal& t, uint32_t cnt)
    {
        AscendC::Add(t.tmp1, t.x1a, t.x2a, cnt);
        AscendC::Add(t.tmp2, t.x1b, t.x2b, cnt);
        AscendC::Add(t.tmp3, t.y1a, t.y2a, cnt);
        AscendC::Add(t.tmp4, t.y1b, t.y2b, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(t.tmp1, t.tmp1, t.tmp2, cnt);
        AscendC::Sub(t.tmp3, t.tmp3, t.tmp4, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Muls(t.tmp1, t.tmp1, 0.5f, cnt);
        AscendC::Muls(t.tmp3, t.tmp3, 0.5f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(t.tmp1, t.tmp1, t.tmp1, cnt);
        AscendC::Mul(t.tmp3, t.tmp3, t.tmp3, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Add(t.tmp1, t.tmp1, t.tmp3, cnt);
        PipeBarrier<PIPE_V>();

        AscendC::Max(t.tmp2, t.x2a, t.x2b, cnt);
        AscendC::Min(t.tmp3, t.x1a, t.x1b, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(t.tmp2, t.tmp2, t.tmp3, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(t.tmp2, t.tmp2, t.tmp2, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Max(t.tmp3, t.y2a, t.y2b, cnt);
        AscendC::Min(t.tmp4, t.y1a, t.y1b, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(t.tmp3, t.tmp3, t.tmp4, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(t.tmp3, t.tmp3, t.tmp3, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Add(t.tmp2, t.tmp2, t.tmp3, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Adds(t.tmp2, t.tmp2, eps, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Maxs(t.tmp2, t.tmp2, 1e-4f, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Div(t.tmp1, t.tmp1, t.tmp2, cnt);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(t.overlap, t.overlap, t.tmp1, cnt);
    }

    __aicore__ inline void CopyOut(uint32_t i, uint32_t cnt)
    {
        AscendC::LocalTensor<T> ov = outOverlapQueue.template DeQue<T>();
        AscendC::LocalTensor<T> at = outAtanSubQueue.template DeQue<T>();
        uint32_t tileStart = myStart + i * tileN;
        AscendC::DataCopyPad(overlapGm[tileStart], ov, {1, static_cast<uint16_t>(cnt * sizeof(T)), 0, 0});
        AscendC::DataCopyPad(atanSubGm[tileStart], at, {1, static_cast<uint16_t>(cnt * sizeof(T)), 0, 0});
        outOverlapQueue.FreeTensor(ov);
        outAtanSubQueue.FreeTensor(at);
    }
};

} // namespace NsCIoU

#endif // __C_IO_U_H__
