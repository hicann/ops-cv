/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file nms_with_mask_regbase_base.h
 * \brief nms_with_mask regbase base
 */

#ifndef NMS_WITH_MASK_REGBASE_BASE_H_
#define NMS_WITH_MASK_REGBASE_BASE_H_

#include "kernel_operator.h"
#include "kernel_utils.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "nms_with_mask_tiling_data.h"

using namespace AscendC;

namespace NMSWithMaskOp {
static constexpr int64_t BUFFER_NUM = 1;
static constexpr int64_t INPUT_DIM_NUM = 2;
static constexpr int64_t ELEMENT_NUM = 5;
static constexpr int64_t BIT_PER_BYTE = 8;
static constexpr int64_t ALIGNED_NUM_B16 = 16;
static constexpr int64_t ALIGNED_NUM_B32 = 8;
static constexpr int64_t ALIGNED_UB_BYTES = 32;
static constexpr int32_t INDEX_X1 = 0;
static constexpr int32_t INDEX_Y1 = 1;
static constexpr int32_t INDEX_X2 = 2;
static constexpr int32_t INDEX_Y2 = 3;
static constexpr int32_t PACK_BYTES = 16;
static constexpr int32_t VL_SIZE = 256; 
static constexpr int32_t VL_SIZE_FLOAT = 64; 
static constexpr MultiCopyConfig copyConfig = {false, 0, 0, false};

static constexpr MicroAPI::CastTrait castTraitB16ToB32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T, bool isBroadcast>
__aicore__ inline void CopyInReg(MicroAPI::RegTensor<float>& vregIn, __ubuf__ T* inAddr, MicroAPI::MaskReg& mask)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        if constexpr (isBroadcast) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregIn, inAddr);
        } else {
            MicroAPI::DataCopy<T>(vregIn, inAddr);
        }
    } else {
        MicroAPI::RegTensor<T> vregInB16;
        if constexpr (isBroadcast) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(vregInB16, inAddr);
        } else {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInB16, inAddr);
        }
        MicroAPI::Cast<float, T, castTraitB16ToB32>(vregIn, vregInB16, mask);
    }
}

template <typename T, bool isBroadcast>
__aicore__ inline void CopyInReg(
    MicroAPI::RegTensor<float>& vregIn, MicroAPI::RegTensor<T>& vregInB16, __ubuf__ T* inAddr, MicroAPI::MaskReg& mask)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        if constexpr (isBroadcast) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregIn, inAddr);
        } else {
            MicroAPI::DataCopy<T>(vregIn, inAddr);
        }
    } else {
        if constexpr (isBroadcast) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(vregInB16, inAddr);
        } else {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInB16, inAddr);
        }
        MicroAPI::Cast<float, T, castTraitB16ToB32>(vregIn, vregInB16, mask);
    }
}

template <typename T, bool isBroadcast>
__aicore__ inline void CopyInRegToFP32(MicroAPI::RegTensor<float>& vregIn, __ubuf__ T* inAddr, MicroAPI::MaskReg& mask)
{
    if constexpr (isBroadcast) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregIn, inAddr);
    } else {
        MicroAPI::DataCopy<T>(vregIn, inAddr);
    }
}

template <typename T>
class NMSWithMaskRegbaseMultiProcess {
public:
    __aicore__ inline NMSWithMaskRegbaseMultiProcess(TPipe* pipeIn)
    {
        pipe_ = pipeIn;
    }
    __aicore__ inline void Init(
        GM_ADDR boxScores, GM_ADDR selectedBoxes, GM_ADDR selectedIdx, GM_ADDR selectedMask, GM_ADDR workspace,
        const NMSWithMaskTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PreProcess();
    __aicore__ inline void PostProcess();
    __aicore__ inline void ReInit();
    __aicore__ inline void CopyIn(int64_t refGroupIdx, int64_t dstGroupIdx, int32_t refCount, int32_t dstCount);
    __aicore__ inline void ComputeMask(int64_t refGroupIdx, int64_t dstGroupIdx, int32_t refCount, int32_t dstCount);
    __aicore__ inline void ComputeRefArea(__ubuf__ T* refLocalAddr, __ubuf__ float* refAreaAddr, int32_t refCount);
    __aicore__ inline void CalcIntersection(
        MicroAPI::MaskReg& pregIou, MicroAPI::RegTensor<float>& sumArea, MicroAPI::RegTensor<float>& vregZeros,
        MicroAPI::RegTensor<float>& refX1, MicroAPI::RegTensor<float>& refY1, MicroAPI::RegTensor<float>& refX2,
        MicroAPI::RegTensor<float>& refY2, MicroAPI::RegTensor<float>& dstX1, MicroAPI::RegTensor<float>& dstY1,
        MicroAPI::RegTensor<float>& dstX2, MicroAPI::RegTensor<float>& dstY2, MicroAPI::MaskReg& preg);
    template <bool dstIsOddBlock>
    __aicore__ inline void ComputeMaskVf(
        __ubuf__ T* refLocalAddr, __ubuf__ T* dstLocalAddr, __ubuf__ float* refAreaAddr, __ubuf__ int32_t* maskUbAddr,
        int32_t refCount, int32_t dstCount);
    __aicore__ inline void CopyOut(int64_t dstGroupIdx, int32_t dstCount);
    __aicore__ inline void CopyOutMask(int64_t refGroupIdx, int64_t dstGroupIdx, int64_t blockIdx, int32_t refCount);
    template <bool isDiagonal>
    __aicore__ inline void CopyInMask(
        int64_t blockIdx, int64_t refGroupIdx, int64_t dstGroupIdx, int32_t refCount, int32_t dstCount);
    template <bool isDiagonal>
    __aicore__ inline void ComputeNMS(int64_t refGroupIdx, int64_t dstGroupIdx, int32_t refCount, int32_t dstCount);
    __aicore__ inline void ComputeNMSForDiagonal(
        __ubuf__ uint8_t* dstMaskAddr, __ubuf__ int32_t* maskUbAddr, int32_t dstCount);
    __aicore__ inline void ComputeNMSForNormal(
        __ubuf__ uint8_t* refMaskAddr, __ubuf__ uint8_t* dstMaskAddr, __ubuf__ int32_t* maskUbAddr, int32_t refCount,
        int32_t dstCount);

    __aicore__ inline void ParseTilingData(const NMSWithMaskTilingData& tilingData)
    {
        boxesNum_ = tilingData.boxesNum;
        usedCoreNum_ = tilingData.usedCoreNum;
        groupSize_ = tilingData.groupSize;
        groupNum_ = tilingData.groupNum;
        blockNum_ = tilingData.blockNum;
        headCoreNum_ = tilingData.headCoreNum;
        blockPerHead_ = tilingData.blockPerHead;
        iouThreshold_ = tilingData.iouThreshold;
        tailGroupSize_ = boxesNum_ - (groupNum_ - 1) * groupSize_;
        bytesPerBlock_ = groupSize_ * groupSize_ / BIT_PER_BYTE;
        alignedElementsPerRow_ = Ops::Base::CeilAlign(ELEMENT_NUM, static_cast<int64_t>(ALIGNED_UB_BYTES / sizeof(T)));
    };

private:
    TPipe* pipe_;
    // 暂时只计算iou < iou_threshold的掩码矩阵
    TQue<QuePosition::VECIN, BUFFER_NUM> refBoxesQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dstBoxesQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> refAreaQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> selectedBoxesQueIn_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> maskQueOut_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> selectedBoxesQueOut_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> selectedIndicesOut_;
    // phase two queues
    TQue<QuePosition::VECIN, BUFFER_NUM> maskQueIn_;
    TQue<QuePosition::VECIN, BUFFER_NUM> refSelMaskQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dstSelMaskQueIn_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dstSelMaskQueOut_;
    GlobalTensor<T> boxScoresGm_;
    GlobalTensor<T> selectedBoxesGm_;
    GlobalTensor<int32_t> selectedIdxGm_;
    GlobalTensor<uint8_t> selectedMaskGm_;
    GlobalTensor<int32_t> tempMaskGm_;

    GM_ADDR boxScoresGmAddr_;
    GM_ADDR selectedBoxesGmAddr_;
    GM_ADDR selectedIdxGmAddr_;
    GM_ADDR selectedMaskGmAddr_;
    GM_ADDR workspaceGmAddr_;

    int32_t alignNum_ = 0;
    int64_t alignedElementsPerRow_ = 0;
    int64_t blockIdx_ = 0;
    int64_t boxesNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t groupSize_ = 0;
    int64_t tailGroupSize_ = 0;
    int64_t groupNum_ = 0;
    int64_t blockNum_ = 0;
    int64_t headCoreNum_ = 0;
    int64_t blockPerHead_ = 0;
    float iouThreshold_ = 0.0f;
    int64_t curCoreProcessNum_ = 0;
    int64_t blockStart_ = 0;
    int64_t bytesPerBlock_ = 0;
};
} // namespace NMSWithMaskOp
#endif // NMS_WITH_MASK_REGBASE_BASE_H_
