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
 * \file nms_with_mask_regbase_multiprocess.h
 * \brief nms_with_mask regbase multiprocess
 */

#ifndef NMS_WITH_MASK_REGBASE_MULTIPROCESS_H_
#define NMS_WITH_MASK_REGBASE_MULTIPROCESS_H_

#include "nms_with_mask_regbase_base.h"

namespace NMSWithMaskOp {
template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::Init(
    GM_ADDR boxScores, GM_ADDR selectedBoxes, GM_ADDR selectedIdx, GM_ADDR selectedMask, GM_ADDR workspace,
    const NMSWithMaskTilingData& tilingData)
{
    ParseTilingData(tilingData);
    boxScoresGmAddr_ = boxScores;
    selectedBoxesGmAddr_ = selectedBoxes;
    selectedIdxGmAddr_ = selectedIdx;
    selectedMaskGmAddr_ = selectedMask;
    workspaceGmAddr_ = workspace;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    curCoreProcessNum_ = blockIdx_ < headCoreNum_ ? blockPerHead_ : blockPerHead_ - 1;
    blockStart_ = blockIdx_ < headCoreNum_ ?
                      (blockPerHead_ * blockIdx_) :
                      (blockPerHead_ * headCoreNum_ + (blockIdx_ - headCoreNum_) * (blockPerHead_ - 1));
    if constexpr (sizeof(T) == sizeof(float)) {
        alignNum_ = ALIGNED_NUM_B32;
    } else {
        alignNum_ = ALIGNED_NUM_B16;
    }
    boxScoresGm_.SetGlobalBuffer((__gm__ T*)boxScores, boxesNum_ * ELEMENT_NUM);
    selectedBoxesGm_.SetGlobalBuffer((__gm__ T*)selectedBoxes, boxesNum_ * ELEMENT_NUM);
    selectedIdxGm_.SetGlobalBuffer((__gm__ int32_t*)selectedIdx, boxesNum_);
    selectedMaskGm_.SetGlobalBuffer((__gm__ uint8_t*)selectedMask, boxesNum_);
    tempMaskGm_.SetGlobalBuffer((__gm__ int32_t*)workspace, blockNum_ * (bytesPerBlock_ / sizeof(int32_t)));

    pipe_->InitBuffer(refBoxesQue_, BUFFER_NUM, groupSize_ * sizeof(T) * ELEMENT_NUM);
    pipe_->InitBuffer(dstBoxesQue_, BUFFER_NUM, groupSize_ * sizeof(T) * ELEMENT_NUM);
    pipe_->InitBuffer(selectedBoxesQueIn_, BUFFER_NUM, groupSize_ * sizeof(T) * alignedElementsPerRow_);
    pipe_->InitBuffer(selectedBoxesQueOut_, BUFFER_NUM, groupSize_ * sizeof(T) * alignedElementsPerRow_);
    pipe_->InitBuffer(selectedIndicesOut_, BUFFER_NUM, groupSize_ * sizeof(int32_t));
    pipe_->InitBuffer(refAreaQue_, BUFFER_NUM, groupSize_ * sizeof(float));
    pipe_->InitBuffer(maskQueOut_, BUFFER_NUM, bytesPerBlock_);
};

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::ReInit()
{
    pipe_->Reset();
    pipe_->InitBuffer(maskQueIn_, BUFFER_NUM, bytesPerBlock_);
    pipe_->InitBuffer(refSelMaskQue_, BUFFER_NUM, groupSize_ * sizeof(uint8_t));
    pipe_->InitBuffer(dstSelMaskQueIn_, BUFFER_NUM, groupSize_ * sizeof(uint8_t));
    pipe_->InitBuffer(dstSelMaskQueOut_, BUFFER_NUM, groupSize_ * sizeof(uint8_t));
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    PreProcess();
    // reset tque
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    ReInit();
    PostProcess();
};

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::PreProcess()
{
    // calculate iou masks and copyout, only upper triangle part of the iou matrix will be calculated
    // block idx will be calculated in row-major order
    int64_t rowBlockIdx = 0; // row-wise block index
    int64_t colBlockIdx = 0; // col-wise block index
    int64_t preBlock = 0;
    for (int64_t i = 0; i < groupNum_; i++) {
        if (preBlock + groupNum_ - i > blockStart_) {
            colBlockIdx = i;
            rowBlockIdx = blockStart_ - preBlock + i;
            break;
        }
        preBlock += (groupNum_ - i);
    }
    for (int64_t idx = blockStart_; idx < blockStart_ + curCoreProcessNum_; idx++) { // 逐个处理当前核心上的数据块
        // reference blocks are in col-wise direction, destination blocks are in row-wise direction
        bool isLastRowBlock = (rowBlockIdx + 1 == groupNum_);
        int32_t refCount = colBlockIdx + 1 == groupNum_ ? tailGroupSize_ : groupSize_;
        int32_t dstCount = rowBlockIdx + 1 == groupNum_ ? tailGroupSize_ : groupSize_;
        CopyIn(colBlockIdx, rowBlockIdx, refCount, dstCount); // 拷贝ref和dst bounding box，然后分别存在各自的UB空间
        ComputeMask(colBlockIdx, rowBlockIdx, refCount, dstCount);
        CopyOutMask(colBlockIdx, rowBlockIdx, idx, refCount);
        if (isLastRowBlock) {
            colBlockIdx += 1;
            rowBlockIdx = colBlockIdx;
        } else {
            rowBlockIdx += 1;
        }
    }
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::PostProcess()
{
    for (int64_t rowIdx = 0; rowIdx < groupNum_; rowIdx++) {
        int64_t remainGroups = groupNum_ - 1 - rowIdx;
        int64_t headCount = Ops::Base::CeilDiv(remainGroups, usedCoreNum_);
        int64_t tailCount = headCount - 1;
        int64_t headCoreNum = remainGroups - tailCount * usedCoreNum_;
        // 对角线块计算
        // 对角线编号是一个等差数列倒序相加的结果，所以计算公式等于：（首项 + 末项）* 项数 / 2
        // 这里首项是groupNum_，末项是groupNum_ - rowIdx + 1，项数是rowIdx
        int64_t maskIdx = rowIdx * (2 * groupNum_ + 1 - rowIdx) / 2;
        if (blockIdx_ == 0) {
            int32_t count = rowIdx + 1 == groupNum_ ? tailGroupSize_ : groupSize_;
            CopyInMask<true>(maskIdx, rowIdx, rowIdx, count, count);
            ComputeNMS<true>(rowIdx, rowIdx, count, count);
            // 对于sorted_nms来说，需要额外补充gather操作得到selected_indices，对角块算完直接输出
            CopyOut(rowIdx, count);
        }
        SyncAll();

        // 非对角线块计算，每一行重新分核
        int64_t colLoopNum = blockIdx_ < headCoreNum ? headCount : tailCount;
        int64_t colBlockStart = blockIdx_ < headCoreNum ?
                                    blockIdx_ * headCount + rowIdx + 1 :
                                    headCoreNum * headCount + (blockIdx_ - headCoreNum) * tailCount + rowIdx + 1;
        for (int64_t k = 0; k < colLoopNum; k++) {
            int64_t curColIdx = colBlockStart + k;
            int64_t curMaskIdx = maskIdx + curColIdx - rowIdx;
            int32_t refCount = rowIdx + 1 == groupNum_ ? tailGroupSize_ : groupSize_;
            int32_t dstCount = curColIdx + 1 == groupNum_ ? tailGroupSize_ : groupSize_;
            CopyInMask<false>(curMaskIdx, rowIdx, curColIdx, refCount, dstCount);
            ComputeNMS<false>(rowIdx, curColIdx, refCount, dstCount);
            CopyOut(curColIdx, dstCount);
        }
        SyncAll();
    }
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::CopyIn(
    int64_t refGroupIdx, int64_t dstGroupIdx, int32_t refCount, int32_t dstCount)
{
    LocalTensor<T> refBoxesUb = refBoxesQue_.AllocTensor<T>();
    LocalTensor<T> dstBoxesUb = dstBoxesQue_.AllocTensor<T>();
    uint64_t refOffset = groupSize_ * refGroupIdx * ELEMENT_NUM;
    uint64_t dstOffset = groupSize_ * dstGroupIdx * ELEMENT_NUM;
    uint8_t refPadNum = static_cast<uint8_t>(Ops::Base::CeilAlign(refCount, alignNum_) - refCount);
    uint8_t dstPadNum = static_cast<uint8_t>(Ops::Base::CeilAlign(dstCount, alignNum_) - dstCount);

    MultiCopyLoopInfo<INPUT_DIM_NUM> refLoopInfo;
    refLoopInfo.loopSrcStride[0] = 1;
    refLoopInfo.loopSrcStride[1] = ELEMENT_NUM;
    refLoopInfo.loopDstStride[0] = refCount + refPadNum;
    refLoopInfo.loopDstStride[1] = 1;
    refLoopInfo.loopSize[0] = ELEMENT_NUM;
    refLoopInfo.loopSize[1] = refCount;
    refLoopInfo.loopRpSize[0] = refPadNum;
    refLoopInfo.loopRpSize[1] = 0;
    MultiCopyParams<T, INPUT_DIM_NUM> refParams{refLoopInfo, 0};

    MultiCopyLoopInfo<INPUT_DIM_NUM> dstLoopInfo;
    dstLoopInfo.loopSrcStride[0] = 1;
    dstLoopInfo.loopSrcStride[1] = ELEMENT_NUM;
    dstLoopInfo.loopDstStride[0] = dstCount + dstPadNum;
    dstLoopInfo.loopDstStride[1] = 1;
    dstLoopInfo.loopSize[0] = ELEMENT_NUM;
    dstLoopInfo.loopSize[1] = dstCount;
    dstLoopInfo.loopRpSize[0] = dstPadNum;
    dstLoopInfo.loopRpSize[1] = 0;
    MultiCopyParams<T, INPUT_DIM_NUM> dstParams{dstLoopInfo, 0};

    DataCopy<T, INPUT_DIM_NUM, copyConfig>(refBoxesUb, boxScoresGm_[refOffset], refParams);
    DataCopy<T, INPUT_DIM_NUM, copyConfig>(dstBoxesUb, boxScoresGm_[dstOffset], dstParams);
    // 添加selectedBoxes的搬入
    if (refGroupIdx == dstGroupIdx) {
        LocalTensor<T> selectedBoxesUb = selectedBoxesQueIn_.AllocTensor<T>();
        DataCopyExtParams copyParams = {
            static_cast<uint16_t>(refCount), static_cast<uint32_t>(ELEMENT_NUM * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(selectedBoxesUb, boxScoresGm_[refOffset], copyParams, padParams);
        selectedBoxesQueIn_.EnQue(selectedBoxesUb);
    }
    refBoxesQue_.EnQue(refBoxesUb);
    dstBoxesQue_.EnQue(dstBoxesUb);
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::ComputeMask(
    int64_t refGroupIdx, int64_t dstGroupIdx, int32_t refCount, int32_t dstCount)
{
    LocalTensor<T> refBoxesUb = refBoxesQue_.DeQue<T>();
    LocalTensor<T> dstBoxesUb = dstBoxesQue_.DeQue<T>();
    LocalTensor<float> refAreaUb = refAreaQue_.AllocTensor<float>();
    LocalTensor<int32_t> maskUb = maskQueOut_.AllocTensor<int32_t>();
    if (refGroupIdx == dstGroupIdx) {
        LocalTensor<T> selectedBoxesInUb = selectedBoxesQueIn_.DeQue<T>();
        LocalTensor<T> selectedBoxesOutUb = selectedBoxesQueOut_.AllocTensor<T>();
        LocalTensor<int32_t> selectedIndicesUb = selectedIndicesOut_.AllocTensor<int32_t>();
        Copy(selectedBoxesOutUb, selectedBoxesInUb, refCount * alignedElementsPerRow_);
        int32_t firstValue = refGroupIdx * groupSize_;
        CreateVecIndex(selectedIndicesUb, firstValue, refCount);
        selectedBoxesQueOut_.EnQue(selectedBoxesOutUb);
        selectedIndicesOut_.EnQue(selectedIndicesUb);
        selectedBoxesQueIn_.FreeTensor(selectedBoxesInUb);
    }
    __ubuf__ T* refLocalAddr = (__ubuf__ T*)refBoxesUb.GetPhyAddr();
    __ubuf__ T* dstLocalAddr = (__ubuf__ T*)dstBoxesUb.GetPhyAddr();
    __ubuf__ float* refAreaAddr = (__ubuf__ float*)refAreaUb.GetPhyAddr();
    __ubuf__ int32_t* maskUbAddr = (__ubuf__ int32_t*)maskUb.GetPhyAddr();
    ComputeRefArea(refLocalAddr, refAreaAddr, refCount); // 计算当前group内的ref bounding box面积，存到refAreaAddr
    refAreaQue_.EnQue(refAreaUb);
    refAreaUb = refAreaQue_.DeQue<float>();
    refAreaAddr = (__ubuf__ float*)refAreaUb.GetPhyAddr();
    int32_t vlLoopNum = Ops::Base::CeilDiv(dstCount, VL_SIZE_FLOAT);
    if (vlLoopNum % 2 == 0) {
        ComputeMaskVf<false>(refLocalAddr, dstLocalAddr, refAreaAddr, maskUbAddr, refCount, dstCount);
    } else {
        ComputeMaskVf<true>(refLocalAddr, dstLocalAddr, refAreaAddr, maskUbAddr, refCount, dstCount);
    }
    maskQueOut_.EnQue(maskUb);
    refBoxesQue_.FreeTensor(refBoxesUb);
    dstBoxesQue_.FreeTensor(dstBoxesUb);
    refAreaQue_.FreeTensor(refAreaUb);
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::CopyOutMask(
    int64_t refGroupIdx, int64_t dstGroupIdx, int64_t blockIdx, int32_t refCount)
{
    // 搬出iou_mask
    LocalTensor<int32_t> maskUb = maskQueOut_.DeQue<int32_t>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(groupSize_), static_cast<uint32_t>(groupSize_ / BIT_PER_BYTE), 0, 0, 0};
    DataCopyPad<int32_t>(tempMaskGm_[blockIdx * bytesPerBlock_ / sizeof(int32_t)], maskUb, copyParams);
    maskQueOut_.FreeTensor(maskUb);
    // 搬出selectedBoxes和selectedIndices
    if (refGroupIdx == dstGroupIdx) {
        LocalTensor<T> selectedBoxesUb = selectedBoxesQueOut_.DeQue<T>();
        LocalTensor<int32_t> selectedIndicesUb = selectedIndicesOut_.DeQue<int32_t>();
        DataCopyExtParams copyParamsForSelectedBoxes{
            static_cast<uint16_t>(refCount), static_cast<uint32_t>(ELEMENT_NUM * sizeof(T)), 0, 0, 0};
        uint64_t offsetForSelectedBoxes = refGroupIdx * groupSize_ * ELEMENT_NUM;
        DataCopyPad(selectedBoxesGm_[offsetForSelectedBoxes], selectedBoxesUb, copyParamsForSelectedBoxes);
        DataCopyExtParams copyParamsForSelectedIndices{
            static_cast<uint16_t>(1), static_cast<uint32_t>(refCount * sizeof(int32_t)), 0, 0, 0};
        uint64_t offsetForSelectedIndices = refGroupIdx * groupSize_;
        DataCopyPad(selectedIdxGm_[offsetForSelectedIndices], selectedIndicesUb, copyParamsForSelectedIndices);
        selectedBoxesQueOut_.FreeTensor(selectedBoxesUb);
        selectedIndicesOut_.FreeTensor(selectedIndicesUb);
    }
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::CopyOut(int64_t dstGroupIdx, int32_t dstCount)
{
    LocalTensor<uint8_t> dstSelMask = dstSelMaskQueOut_.DeQue<uint8_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(dstCount * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(selectedMaskGm_[dstGroupIdx * groupSize_], dstSelMask, copyParams);
    dstSelMaskQueOut_.FreeTensor(dstSelMask);
}

template <typename T>
template <bool isDiagonal>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::CopyInMask(
    int64_t blockIdx, int64_t refGroupIdx, int64_t dstGroupIdx, int32_t refCount, int32_t dstCount)
{
    // 拷贝PreProcess获得的局部mask结果（bytesPerBlock个元素）
    LocalTensor<int32_t> maskUb = maskQueIn_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(groupSize_), static_cast<uint32_t>(groupSize_ / BIT_PER_BYTE), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad<int32_t>(maskUb, tempMaskGm_[blockIdx * bytesPerBlock_ / sizeof(int32_t)], copyParams, padParams);
    maskQueIn_.EnQue(maskUb);
    // 拷贝（初始化）待更新box的局部mask结果
    DataCopyExtParams dstMaskParams{1, static_cast<uint32_t>(dstCount), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> maskPadParams{false, 0, 0, 0};
    LocalTensor<uint8_t> dstSelMaskUbIn = dstSelMaskQueIn_.AllocTensor<uint8_t>();
    if (refGroupIdx == 0) {
        Duplicate<uint8_t>(dstSelMaskUbIn, 1, dstCount);
    } else {
        DataCopyPad<uint8_t>(dstSelMaskUbIn, selectedMaskGm_[dstGroupIdx * groupSize_], dstMaskParams, maskPadParams);
    }
    dstSelMaskQueIn_.EnQue(dstSelMaskUbIn);
    if constexpr (!isDiagonal) {
        // 拷贝之前计算的局部结果
        DataCopyExtParams refMaskParams{1, static_cast<uint32_t>(refCount), 0, 0, 0};
        LocalTensor<uint8_t> refSelMaskUb = refSelMaskQue_.AllocTensor<uint8_t>();
        DataCopyPad<uint8_t>(refSelMaskUb, selectedMaskGm_[refGroupIdx * groupSize_], refMaskParams, maskPadParams);
        refSelMaskQue_.EnQue(refSelMaskUb);
    }
}

template <typename T>
template <bool isDiagonal>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::ComputeNMS(
    int64_t refGroupIdx, int64_t dstGroupIdx, int32_t refCount, int32_t dstCount)
{
    LocalTensor<int32_t> maskUb = maskQueIn_.DeQue<int32_t>();
    LocalTensor<uint8_t> dstSelMaskUbIn = dstSelMaskQueIn_.DeQue<uint8_t>();
    __ubuf__ int32_t* maskUbAddr = (__ubuf__ int32_t*)maskUb.GetPhyAddr();
    __ubuf__ uint8_t* dstMaskAddr = (__ubuf__ uint8_t*)dstSelMaskUbIn.GetPhyAddr();
    LocalTensor<uint8_t> refSelMaskUb;
    __ubuf__ uint8_t* refMaskAddr;
    if constexpr (isDiagonal) {
        ComputeNMSForDiagonal(dstMaskAddr, maskUbAddr, dstCount);
    } else {
        refSelMaskUb = refSelMaskQue_.DeQue<uint8_t>();
        refMaskAddr = (__ubuf__ uint8_t*)refSelMaskUb.GetPhyAddr();
        ComputeNMSForNormal(refMaskAddr, dstMaskAddr, maskUbAddr, refCount, dstCount);
    }
    LocalTensor<uint8_t> dstSelMaskUbOut = dstSelMaskQueOut_.AllocTensor<uint8_t>();
    Copy(dstSelMaskUbOut, dstSelMaskUbIn, dstCount);
    dstSelMaskQueOut_.EnQue(dstSelMaskUbOut);
    maskQueIn_.FreeTensor(maskUb);
    dstSelMaskQueIn_.FreeTensor(dstSelMaskUbIn);
    if constexpr (!isDiagonal) {
        refSelMaskQue_.FreeTensor(refSelMaskUb);
    }
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::ComputeNMSForDiagonal(
    __ubuf__ uint8_t* dstMaskAddr, __ubuf__ int32_t* maskUbAddr, int32_t dstCount)
{
    int32_t vlSize = VL_SIZE / sizeof(uint8_t);
    uint16_t rowNum = static_cast<uint16_t>(dstCount);          // how many cols to iterate
    uint16_t loopPerRow = Ops::Base::CeilDiv(dstCount, vlSize); // how many loops to iterate per row
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> refTensor;
        MicroAPI::RegTensor<uint8_t> dstTensor;
        MicroAPI::RegTensor<uint8_t> vregZeros;
        MicroAPI::RegTensor<uint8_t> outTensor;
        MicroAPI::MaskReg preg;
        MicroAPI::MaskReg iouMask;
        MicroAPI::MaskReg removeMask;
        MicroAPI::MaskReg refMask;
        MicroAPI::MaskReg trilMask; // preg for lower triangular
        MicroAPI::MaskReg triuMask; // preg for upper triangular
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<uint8_t>(vregZeros, 0, pregAll);
        for (uint16_t rowIdx = 0; rowIdx < rowNum; rowIdx++) {
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            uint32_t rowEleNum = static_cast<uint32_t>(dstCount);
            uint32_t trilEleNum = rowIdx + 1;
            MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_BRC_B8>(refTensor, dstMaskAddr + rowIdx);
            MicroAPI::CompareScalar<uint8_t, CMPMODE::EQ>(
                refMask, refTensor, 1, pregAll); // refMask表示要么全选要么全不选，基于当前refTensor是否全为1来判断
            for (uint16_t loopIndex = 0; loopIndex < loopPerRow; loopIndex++) {
                preg = MicroAPI::UpdateMask<uint8_t>(rowEleNum);
                trilMask = MicroAPI::UpdateMask<uint8_t>(trilEleNum);
                MicroAPI::MaskNot(triuMask, trilMask, pregAll);
                MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<int32_t>(
                    rowIdx, groupSize_ / BIT_PER_BYTE / sizeof(int32_t), loopIndex,
                    vlSize / BIT_PER_BYTE / sizeof(int32_t));
                // 搬入待比较mask的reg，每一bit表示一个有效值
                MicroAPI::DataCopy<int32_t, MicroAPI::MaskDist::DIST_NORM>(iouMask, maskUbAddr, offset);
                MicroAPI::DataCopy<uint8_t>(dstTensor, dstMaskAddr + loopIndex * vlSize);
                MicroAPI::MaskAnd(removeMask, iouMask, refMask, pregAll);
                MicroAPI::MaskAnd(removeMask, removeMask, triuMask, pregAll);
                MicroAPI::Select<uint8_t>(outTensor, vregZeros, dstTensor, removeMask);
                MicroAPI::DataCopy<uint8_t>(dstMaskAddr + loopIndex * vlSize, outTensor, preg);
            }
        }
    }
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::ComputeNMSForNormal(
    __ubuf__ uint8_t* refMaskAddr, __ubuf__ uint8_t* dstMaskAddr, __ubuf__ int32_t* maskUbAddr, int32_t refCount,
    int32_t dstCount)
{
    int32_t vlSize = VL_SIZE / sizeof(uint8_t);
    uint16_t rowNum = static_cast<uint16_t>(refCount);          // how many rows to iterate
    uint16_t loopPerRow = Ops::Base::CeilDiv(dstCount, vlSize); // how many loops to iterate per row
    int32_t dstCountAligned = Ops::Base::CeilAlign(dstCount, alignNum_);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> refTensor;
        MicroAPI::RegTensor<uint8_t> dstTensor;
        MicroAPI::RegTensor<uint8_t> vregZeros;
        MicroAPI::RegTensor<uint8_t> outTensor;
        MicroAPI::MaskReg preg;
        MicroAPI::MaskReg iouMask;
        MicroAPI::MaskReg removeMask;
        MicroAPI::MaskReg refMask;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<uint8_t>(vregZeros, 0, pregAll);
        for (uint16_t rowIdx = 0; rowIdx < rowNum; rowIdx++) {
            uint32_t rowEleNum = static_cast<uint32_t>(dstCount);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_BRC_B8>(refTensor, refMaskAddr + rowIdx);
            MicroAPI::CompareScalar<uint8_t, CMPMODE::EQ>(refMask, refTensor, 1, pregAll);
            for (uint16_t loopIndex = 0; loopIndex < loopPerRow; loopIndex++) {
                preg = MicroAPI::UpdateMask<uint8_t>(rowEleNum);
                MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<int32_t>(
                    rowIdx, groupSize_ / BIT_PER_BYTE / sizeof(int32_t), loopIndex,
                    vlSize / BIT_PER_BYTE / sizeof(int32_t));
                // 搬入待比较mask的reg，每一bit表示一个有效值
                MicroAPI::DataCopy<int32_t, MicroAPI::MaskDist::DIST_NORM>(iouMask, maskUbAddr, offset);
                MicroAPI::DataCopy<uint8_t>(dstTensor, dstMaskAddr + loopIndex * vlSize);
                MicroAPI::MaskAnd(removeMask, iouMask, refMask, pregAll);
                MicroAPI::Select<uint8_t>(outTensor, vregZeros, dstTensor, removeMask);
                MicroAPI::DataCopy<uint8_t>(dstMaskAddr + loopIndex * vlSize, outTensor, preg);
            }
        }
    }
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::ComputeRefArea(
    __ubuf__ T* refLocalAddr, __ubuf__ float* refAreaAddr, int32_t refCount)
{
    int32_t vlSize = VL_SIZE / sizeof(float);
    uint16_t loopNum = Ops::Base::CeilDiv(refCount, vlSize);
    int32_t refCountAligned = Ops::Base::CeilAlign(refCount, alignNum_);
    uint32_t count = static_cast<uint32_t>(refCount);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> x1;
        MicroAPI::RegTensor<float> y1;
        MicroAPI::RegTensor<float> x2;
        MicroAPI::RegTensor<float> y2;
        MicroAPI::RegTensor<float> width;
        MicroAPI::RegTensor<float> height;
        MicroAPI::RegTensor<float> area;
        MicroAPI::MaskReg preg;
        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            preg = MicroAPI::UpdateMask<float>(count);
            CopyInReg<T, false>(y1, refLocalAddr + loopIdx * vlSize, preg);
            CopyInReg<T, false>(x1, refLocalAddr + loopIdx * vlSize + INDEX_Y1 * refCountAligned, preg);
            CopyInReg<T, false>(y2, refLocalAddr + loopIdx * vlSize + INDEX_X2 * refCountAligned, preg);
            CopyInReg<T, false>(x2, refLocalAddr + loopIdx * vlSize + INDEX_Y2 * refCountAligned, preg);
            MicroAPI::Sub(width, x2, x1, preg);
            MicroAPI::Sub(height, y2, y1, preg);
            MicroAPI::Mul(area, width, height, preg);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(refAreaAddr, area, vlSize, preg);
        }
    }
}

template <typename T>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::CalcIntersection(
    MicroAPI::MaskReg& pregIou, MicroAPI::RegTensor<float>& sumArea, MicroAPI::RegTensor<float>& vregZeros,
    MicroAPI::RegTensor<float>& refX1, MicroAPI::RegTensor<float>& refY1, MicroAPI::RegTensor<float>& refX2,
    MicroAPI::RegTensor<float>& refY2, MicroAPI::RegTensor<float>& dstX1, MicroAPI::RegTensor<float>& dstY1,
    MicroAPI::RegTensor<float>& dstX2, MicroAPI::RegTensor<float>& dstY2, MicroAPI::MaskReg& preg)
{
    MicroAPI::RegTensor<float> minX2;
    MicroAPI::RegTensor<float> maxX1;
    MicroAPI::RegTensor<float> minY2;
    MicroAPI::RegTensor<float> maxY1;
    MicroAPI::RegTensor<float> intersection;
    MicroAPI::Min(minX2, refX2, dstX2, preg);
    MicroAPI::Max(maxX1, refX1, dstX1, preg);
    MicroAPI::Min(minY2, refY2, dstY2, preg);
    MicroAPI::Max(maxY1, refY1, dstY1, preg);
    MicroAPI::Sub(minX2, minX2, maxX1, preg);
    MicroAPI::Sub(minY2, minY2, maxY1, preg);
    MicroAPI::Max(minX2, minX2, vregZeros, preg);
    MicroAPI::Max(minY2, minY2, vregZeros, preg);
    MicroAPI::Mul(intersection, minX2, minY2, preg);
    MicroAPI::Sub(sumArea, sumArea, intersection, preg); // 视sumArea为并集大小
    MicroAPI::Muls(sumArea, sumArea, iouThreshold_, preg);
    MicroAPI::Compare<float, CMPMODE::GT>(pregIou, intersection, sumArea, preg);
}

template <typename T>
template <bool dstIsOddBlock>
__aicore__ inline void NMSWithMaskRegbaseMultiProcess<T>::ComputeMaskVf(
    __ubuf__ T* refLocalAddr, __ubuf__ T* dstLocalAddr, __ubuf__ float* refAreaAddr, __ubuf__ int32_t* maskUbAddr,
    int32_t refCount, int32_t dstCount)
{
    int32_t vlSize = VL_SIZE / sizeof(float);
    uint16_t rowNum = static_cast<uint16_t>(refCount);            // how many rows to iterate
    uint16_t rowLoopCount = Ops::Base::CeilDiv(dstCount, vlSize); // how many vl(blocks) per row
    uint16_t rowLoopNum = rowLoopCount / 2;                       // loop num per row (each time iterate 2 vls)
    int32_t refCountAligned = Ops::Base::CeilAlign(refCount, alignNum_);
    int32_t dstCountAligned = Ops::Base::CeilAlign(dstCount, alignNum_);
    uint32_t dstCountU32 = static_cast<uint32_t>(dstCount);
    uint32_t dstStride = PACK_BYTES / sizeof(int32_t);
    uint32_t srcStride = groupSize_ / BIT_PER_BYTE / sizeof(int32_t);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> refX1;
        MicroAPI::RegTensor<float> refY1;
        MicroAPI::RegTensor<float> refX2;
        MicroAPI::RegTensor<float> refY2;
        MicroAPI::RegTensor<float> dstX1;
        MicroAPI::RegTensor<float> dstY1;
        MicroAPI::RegTensor<float> dstX2;
        MicroAPI::RegTensor<float> dstY2;
        MicroAPI::RegTensor<float> dstX3;
        MicroAPI::RegTensor<float> dstY3;
        MicroAPI::RegTensor<float> dstX4;
        MicroAPI::RegTensor<float> dstY4;
        MicroAPI::RegTensor<float> refArea;
        MicroAPI::RegTensor<float> dstHeight;
        MicroAPI::RegTensor<float> dstWidth;
        MicroAPI::RegTensor<float> dstArea0;
        MicroAPI::RegTensor<float> dstArea1;
        MicroAPI::RegTensor<float> sumArea;
        MicroAPI::RegTensor<float> vregZeros;
        MicroAPI::MaskReg preg0;
        MicroAPI::MaskReg preg1;
        MicroAPI::MaskReg pregIou0;
        MicroAPI::MaskReg pregIou1;
        MicroAPI::MaskReg pregRes0;
        MicroAPI::MaskReg pregRes1;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<float>(vregZeros, 0.0f, pregAll);
        for (uint16_t dstBlockIdx = 0; dstBlockIdx < rowLoopNum; dstBlockIdx++) {
            preg0 = MicroAPI::UpdateMask<float>(dstCountU32);
            preg1 = MicroAPI::UpdateMask<float>(dstCountU32);
            CopyInReg<T, false>(dstY1, dstLocalAddr + dstBlockIdx * vlSize * 2, preg0);
            CopyInReg<T, false>(dstX1, dstLocalAddr + dstBlockIdx * vlSize * 2 + INDEX_Y1 * dstCountAligned, preg0);
            CopyInReg<T, false>(dstY2, dstLocalAddr + dstBlockIdx * vlSize * 2 + INDEX_X2 * dstCountAligned, preg0);
            CopyInReg<T, false>(dstX2, dstLocalAddr + dstBlockIdx * vlSize * 2 + INDEX_Y2 * dstCountAligned, preg0);
            CopyInReg<T, false>(dstY3, dstLocalAddr + dstBlockIdx * vlSize * 2 + vlSize, preg1);
            CopyInReg<T, false>(
                dstX3, dstLocalAddr + dstBlockIdx * vlSize * 2 + vlSize + INDEX_Y1 * dstCountAligned, preg1);
            CopyInReg<T, false>(
                dstY4, dstLocalAddr + dstBlockIdx * vlSize * 2 + vlSize + INDEX_X2 * dstCountAligned, preg1);
            CopyInReg<T, false>(
                dstX4, dstLocalAddr + dstBlockIdx * vlSize * 2 + vlSize + INDEX_Y2 * dstCountAligned, preg1);
            MicroAPI::Sub(dstWidth, dstX2, dstX1, preg0);
            MicroAPI::Sub(dstHeight, dstY2, dstY1, preg0);
            MicroAPI::Mul(dstArea0, dstWidth, dstHeight, preg0);
            MicroAPI::Sub(dstWidth, dstX4, dstX3, preg1);
            MicroAPI::Sub(dstHeight, dstY4, dstY3, preg1);
            MicroAPI::Mul(dstArea1, dstWidth, dstHeight, preg1);
            for (uint16_t refIdx = 0; refIdx < rowNum; refIdx++) {
                // 要满足16 byte搬出，最终mask存为int32类型；pre过程计算的中间mask大小为groupSize_ *
                // groupSize_，每行对应groupSize_个bit，因此源操作数的偏移量是groupSize_ / BIT_PER_BYTE /
                // sizeof(int32_t)
                MicroAPI::AddrReg offsetReg = MicroAPI::CreateAddrReg<int32_t>(
                    dstBlockIdx, dstStride, refIdx, srcStride); // 4：16 / sizeof(int32)
                CopyInReg<T, true>(refY1, refLocalAddr + refIdx, pregAll);
                CopyInReg<T, true>(refX1, refLocalAddr + refIdx + INDEX_Y1 * refCountAligned, pregAll);
                CopyInReg<T, true>(refY2, refLocalAddr + refIdx + INDEX_X2 * refCountAligned, pregAll);
                CopyInReg<T, true>(refX2, refLocalAddr + refIdx + INDEX_Y2 * refCountAligned, pregAll);
                CopyInReg<float, true>(refArea, refAreaAddr + refIdx, pregAll);
                MicroAPI::Add(sumArea, dstArea0, refArea, preg0);
                CalcIntersection(
                    pregIou0, sumArea, vregZeros, refX1, refY1, refX2, refY2, dstX1, dstY1, dstX2, dstY2, preg0);
                MicroAPI::Add(sumArea, dstArea1, refArea, preg1);
                CalcIntersection(
                    pregIou1, sumArea, vregZeros, refX1, refY1, refX2, refY2, dstX3, dstY3, dstX4, dstY4, preg1);
                // interleave from b32 maskreg to b16 maskreg
                MicroAPI::MaskDeInterleave<half>(pregRes0, pregRes1, pregIou0, pregIou1); // 16B对齐
                // maskUbAddr + offset
                MicroAPI::DataCopy<int32_t, MicroAPI::MaskDist::DIST_PACK>(maskUbAddr, pregRes0, offsetReg);
            }
        }
        if constexpr (dstIsOddBlock) {
            preg0 = MicroAPI::UpdateMask<float>(dstCountU32);
            CopyInReg<T, false>(dstY1, dstLocalAddr + rowLoopNum * vlSize * 2, preg0);
            CopyInReg<T, false>(dstX1, dstLocalAddr + rowLoopNum * vlSize * 2 + INDEX_Y1 * dstCountAligned, preg0);
            CopyInReg<T, false>(dstY2, dstLocalAddr + rowLoopNum * vlSize * 2 + INDEX_X2 * dstCountAligned, preg0);
            CopyInReg<T, false>(dstX2, dstLocalAddr + rowLoopNum * vlSize * 2 + INDEX_Y2 * dstCountAligned, preg0);
            MicroAPI::Sub(dstWidth, dstX2, dstX1, preg0);
            MicroAPI::Sub(dstHeight, dstY2, dstY1, preg0);
            MicroAPI::Mul(dstArea0, dstWidth, dstHeight, preg0);
            for (uint16_t refIdx = 0; refIdx < rowNum; refIdx++) {
                CopyInReg<T, true>(refY1, refLocalAddr + refIdx, pregAll);
                CopyInReg<T, true>(refX1, refLocalAddr + refIdx + INDEX_Y1 * refCountAligned, pregAll);
                CopyInReg<T, true>(refY2, refLocalAddr + refIdx + INDEX_X2 * refCountAligned, pregAll);
                CopyInReg<T, true>(refX2, refLocalAddr + refIdx + INDEX_Y2 * refCountAligned, pregAll);
                CopyInReg<float, true>(refArea, refAreaAddr + refIdx, pregAll);
                MicroAPI::Add(sumArea, dstArea0, refArea, preg0);
                CalcIntersection(
                    pregIou0, sumArea, vregZeros, refX1, refY1, refX2, refY2, dstX1, dstY1, dstX2, dstY2, preg0);
                // interleave from b32 maskreg to b16 maskreg
                MicroAPI::MaskDeInterleave<half>(pregRes0, pregRes1, pregIou0, pregIou0);
                MicroAPI::DataCopy<int32_t, MicroAPI::MaskDist::DIST_PACK>(
                    maskUbAddr + rowLoopNum * dstStride + refIdx * srcStride, pregRes0);
            }
        }
    }
}
} // namespace NMSWithMaskOp

#endif // NMS_WITH_MASK_REGBASE_MULTIPROCESS_H_