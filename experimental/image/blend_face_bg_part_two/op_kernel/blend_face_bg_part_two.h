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
 * \file blend_face_bg_part_two.h
 * \brief
 */
#ifndef __BLEND_FACE_BG_PART_TWO_H__
#define __BLEND_FACE_BG_PART_TWO_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "blend_face_bg_part_two_tiling_data.h"
#include "blend_face_bg_part_two_tiling_key.h"

namespace NsBlendFaceBgPartTwo {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // double buffer
constexpr uint32_t DMA_ALIGN_BYTES = 32;

template <typename T, typename BgT>
class KernelBlendFaceBgPartTwo {
public:
    __aicore__ inline KernelBlendFaceBgPartTwo() {}

    __aicore__ inline void Init(GM_ADDR acc_face, GM_ADDR acc_mask, GM_ADDR max_mask, GM_ADDR bg_img, GM_ADDR fused_img,
                                const BlendFaceBgPartTwoTilingData* tilingData, TPipe* pipePtr)
    {
        this->pipe = pipePtr;
        this->tileSize = tilingData->tileSize;
        this->epsilon = tilingData->epsilon;

        uint32_t baseElems = tilingData->baseElems;
        uint32_t pivot = tilingData->pivot;

        uint32_t blockId = GetBlockIdx();
        this->myElems = baseElems + (blockId < pivot ? 1U : 0U);
        uint64_t myStart = static_cast<uint64_t>(blockId) * baseElems + (blockId < pivot ? blockId : pivot);

        this->innerLoops = (this->myElems + this->tileSize - 1) / this->tileSize;

        accFaceGm.SetGlobalBuffer((__gm__ T*)acc_face + myStart, this->myElems);
        accMaskGm.SetGlobalBuffer((__gm__ T*)acc_mask + myStart, this->myElems);
        maxMaskGm.SetGlobalBuffer((__gm__ T*)max_mask + myStart, this->myElems);
        bgImgGm.SetGlobalBuffer((__gm__ BgT*)bg_img + myStart, this->myElems);
        fusedImgGm.SetGlobalBuffer((__gm__ T*)fused_img + myStart, this->myElems);

        pipe->InitBuffer(inQueueAccFace, BUFFER_NUM, this->tileSize * sizeof(T));
        pipe->InitBuffer(inQueueAccMask, BUFFER_NUM, this->tileSize * sizeof(T));
        pipe->InitBuffer(inQueueMaxMask, BUFFER_NUM, this->tileSize * sizeof(T));
        pipe->InitBuffer(inQueueBgImg, BUFFER_NUM, this->tileSize * sizeof(BgT));
        pipe->InitBuffer(outQueueFusedImg, BUFFER_NUM, this->tileSize * sizeof(T));

        pipe->InitBuffer(fusionFaceBuf, this->tileSize * sizeof(T));
        pipe->InitBuffer(invMaxBuf, this->tileSize * sizeof(T));
        pipe->InitBuffer(bgPartBuf, this->tileSize * sizeof(T));
        pipe->InitBuffer(facePartBuf, this->tileSize * sizeof(T));
        if constexpr (sizeof(BgT) != sizeof(T)) {
            pipe->InitBuffer(bgImgHalfBuf, this->tileSize * sizeof(half));
            pipe->InitBuffer(bgImgCastBuf, this->tileSize * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->innerLoops; i++) {
            uint32_t curTile = this->tileSize;
            if (i == this->innerLoops - 1) {
                curTile = this->myElems - i * this->tileSize;
            }
            CopyIn(i, curTile);
            Compute(curTile);
            CopyOut(i, curTile);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t curTile)
    {
        uint32_t offset = idx * this->tileSize;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTile * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = MakeCopyInPadParams<T>(curTile);
        DataCopyExtParams bgCopyParams{1, static_cast<uint32_t>(curTile * sizeof(BgT)), 0, 0, 0};
        DataCopyPadExtParams<BgT> bgPadParams = MakeCopyInPadParams<BgT>(curTile);

        LocalTensor<T> accFaceLocal = inQueueAccFace.template AllocTensor<T>();
        LocalTensor<T> accMaskLocal = inQueueAccMask.template AllocTensor<T>();
        LocalTensor<T> maxMaskLocal = inQueueMaxMask.template AllocTensor<T>();
        LocalTensor<BgT> bgImgLocal = inQueueBgImg.template AllocTensor<BgT>();

        DataCopyPad(accFaceLocal, accFaceGm[offset], copyParams, padParams);
        DataCopyPad(accMaskLocal, accMaskGm[offset], copyParams, padParams);
        DataCopyPad(maxMaskLocal, maxMaskGm[offset], copyParams, padParams);
        DataCopyPad(bgImgLocal, bgImgGm[offset], bgCopyParams, bgPadParams);

        inQueueAccFace.EnQue(accFaceLocal);
        inQueueAccMask.EnQue(accMaskLocal);
        inQueueMaxMask.EnQue(maxMaskLocal);
        inQueueBgImg.EnQue(bgImgLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> accFaceLocal = inQueueAccFace.template DeQue<T>();
        LocalTensor<T> accMaskLocal = inQueueAccMask.template DeQue<T>();
        LocalTensor<T> maxMaskLocal = inQueueMaxMask.template DeQue<T>();
        LocalTensor<BgT> bgImgLocal = inQueueBgImg.template DeQue<BgT>();

        LocalTensor<T> fusionFaceLocal = fusionFaceBuf.template Get<T>();
        LocalTensor<T> invMaxLocal = invMaxBuf.template Get<T>();
        LocalTensor<T> bgPartLocal = bgPartBuf.template Get<T>();
        LocalTensor<T> facePartLocal = facePartBuf.template Get<T>();
        LocalTensor<T> bgImgCalcLocal;

        LocalTensor<T> fusedImgLocal = outQueueFusedImg.template AllocTensor<T>();

        if constexpr (sizeof(BgT) == sizeof(T)) {
            bgImgCalcLocal = bgImgLocal;
        } else {
            LocalTensor<half> bgImgHalfLocal = bgImgHalfBuf.template Get<half>();
            bgImgCalcLocal = bgImgCastBuf.template Get<T>();
            Cast(bgImgHalfLocal, bgImgLocal, RoundMode::CAST_NONE, count);
            Cast(bgImgCalcLocal, bgImgHalfLocal, RoundMode::CAST_NONE, count);
        }

        // Step 1: tmp_acc_mask = acc_mask + epsilon (reuse accMaskLocal)
        Adds(accMaskLocal, accMaskLocal, this->epsilon, count);

        // Step 2: fusion_face = acc_face / tmp_acc_mask
        Div(fusionFaceLocal, accFaceLocal, accMaskLocal, count);

        // Step 3: inv_max = 1.0 - max_mask
        Muls(invMaxLocal, maxMaskLocal, static_cast<T>(-1.0f), count);
        Adds(invMaxLocal, invMaxLocal, static_cast<T>(1.0f), count);

        // Step 4: bg_part = bg_img * inv_max
        Mul(bgPartLocal, bgImgCalcLocal, invMaxLocal, count);

        // Step 5: face_part = fusion_face * max_mask (original mask, not inv_max)
        Mul(facePartLocal, fusionFaceLocal, maxMaskLocal, count);

        // Step 6: fused_img = face_part + bg_part
        Add(fusedImgLocal, facePartLocal, bgPartLocal, count);

        outQueueFusedImg.template EnQue<T>(fusedImgLocal);
        inQueueAccFace.FreeTensor(accFaceLocal);
        inQueueAccMask.FreeTensor(accMaskLocal);
        inQueueMaxMask.FreeTensor(maxMaskLocal);
        inQueueBgImg.FreeTensor(bgImgLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t curTile)
    {
        uint32_t offset = idx * this->tileSize;
        LocalTensor<T> fusedImgLocal = outQueueFusedImg.template DeQue<T>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTile * sizeof(T)), 0, 0, 0};
        DataCopyPad(fusedImgGm[offset], fusedImgLocal, copyParams);

        outQueueFusedImg.FreeTensor(fusedImgLocal);
    }

    template <typename CopyT>
    __aicore__ inline DataCopyPadExtParams<CopyT> MakeCopyInPadParams(uint32_t count)
    {
        uint32_t bytes = count * sizeof(CopyT);
        uint32_t tailBytes = bytes % DMA_ALIGN_BYTES;
        uint8_t rightPadding = 0;
        if (tailBytes != 0) {
            rightPadding = static_cast<uint8_t>((DMA_ALIGN_BYTES - tailBytes) / sizeof(CopyT));
        }
        return DataCopyPadExtParams<CopyT>{rightPadding != 0, 0, rightPadding, static_cast<CopyT>(0)};
    }

private:
    TPipe* pipe;

    TQue<TPosition::VECIN, BUFFER_NUM> inQueueAccFace;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueAccMask;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueMaxMask;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueBgImg;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueFusedImg;

    TBuf<TPosition::VECCALC> fusionFaceBuf;
    TBuf<TPosition::VECCALC> invMaxBuf;
    TBuf<TPosition::VECCALC> bgPartBuf;
    TBuf<TPosition::VECCALC> facePartBuf;
    TBuf<TPosition::VECCALC> bgImgHalfBuf;
    TBuf<TPosition::VECCALC> bgImgCastBuf;

    GlobalTensor<T> accFaceGm;
    GlobalTensor<T> accMaskGm;
    GlobalTensor<T> maxMaskGm;
    GlobalTensor<BgT> bgImgGm;
    GlobalTensor<T> fusedImgGm;

    uint32_t myElems = 0;
    uint32_t innerLoops = 0;
    uint32_t tileSize = 0;
    float epsilon = 0.0f;
};

} // namespace NsBlendFaceBgPartTwo
#endif
