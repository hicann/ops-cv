/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file col2im_simt.h
 * \brief col2im_simt
 */

#ifndef COL2IM_SIMT_H
#define COL2IM_SIMT_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "col2im_tiling_data.h"
namespace Col2imOps {

using namespace AscendC;

const uint32_t VF_MAX_THREAD_NUM = 1024;

template <typename ACC_T, typename D_T>
class Col2imSimt {
public:
    __aicore__ inline Col2imSimt()
    {}
    __aicore__ inline void Init(
        GM_ADDR gradIn, GM_ADDR gradOut, GM_ADDR workspace,
        const Col2imRegBaseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<D_T> inputXGm_;
    GlobalTensor<D_T> yGm_;
    uint32_t blockId_ = GetBlockIdx();
    const Col2imRegBaseTilingData* tiling_;
};

template <typename ACC_T, typename D_T>
__aicore__ inline void Col2imSimt<ACC_T, D_T>::Init(
    GM_ADDR gradIn, GM_ADDR gradOut, GM_ADDR workspace, const Col2imRegBaseTilingData* __restrict tilingData)
{
    inputXGm_.SetGlobalBuffer((__gm__ D_T*)(gradOut));
    yGm_.SetGlobalBuffer((__gm__ D_T*)(gradIn));

    tiling_ = tilingData;
}

template <typename ACC_T, typename D_T>
__simt_vf__ LAUNCH_BOUND(VF_MAX_THREAD_NUM) inline void Col2imSimtCompute(
    __gm__ D_T* inputXGmAddr, __gm__ D_T* yGmAddr, const uint32_t cnt, const uint32_t gradInH, const uint32_t gradInW, 
    const uint32_t kernelSizeH, const uint32_t kernelSizeW, const uint32_t padHeight, const uint32_t padWidth, 
    const uint32_t strideHeight, const uint32_t strideWidth, const uint32_t dilationHeight, const uint32_t dilationWidth, 
    const uint32_t heightGradOut, const uint32_t widthGradOut, uint32_t shiftGW_, uint32_t mGW_, uint32_t shiftGWH_, 
    uint32_t mGWH_, uint32_t shiftSW_, uint32_t mSW_, uint32_t shiftSH_, uint32_t mSH_, uint32_t shiftDH_, 
    uint32_t mDH_, uint32_t shiftDW_, uint32_t mDW_)
{
    for(uint32_t idx = AscendC::Simt::GetThreadIdx<0>() + AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum<0>(); idx < cnt; idx += AscendC::Simt::GetBlockNum() * AscendC::Simt::GetThreadNum<0>()) {
        ACC_T valv = static_cast<ACC_T>(0);
        const uint32_t wIm = idx % gradInW + padWidth;
        const uint32_t hIm = Simt::UintDiv(idx, mGW_, shiftGW_) % gradInH + padHeight;
        const uint32_t cIm = Simt::UintDiv(idx, mGWH_, shiftGWH_);
        uint32_t kernelExtentW = (kernelSizeW - 1) * dilationWidth + 1;
        uint32_t kernelExtentH = (kernelSizeH - 1) * dilationHeight + 1;
        // 计算所需的起始与结束位置
        uint32_t wGradOutStart = 0;
        if (wIm >= kernelExtentW) {
            wGradOutStart = Simt::UintDiv((wIm - kernelExtentW), mSW_, shiftSW_) + 1;
        }
        uint32_t wGradOutEnd = Simt::UintDiv(wIm, mSW_, shiftSW_) + 1;
        if (wGradOutEnd > widthGradOut) {
            wGradOutEnd = widthGradOut;
        }
        uint32_t hGradOutStart = 0;
        if (hIm >= kernelExtentH) {
            hGradOutStart = Simt::UintDiv((hIm - kernelExtentH), mSH_, shiftSH_) + 1;
        }
        uint32_t hGradOutEnd = Simt::UintDiv(hIm, mSH_, shiftSH_) + 1;
        if (hGradOutEnd > heightGradOut) {
            hGradOutEnd = heightGradOut;
        }

        // 计算结果
        for (uint32_t hGradOut = hGradOutStart; hGradOut < hGradOutEnd; hGradOut += 1) {
            for (uint32_t wGradOut = wGradOutStart; wGradOut < wGradOutEnd; wGradOut += 1) {
                uint32_t hK = (hIm - hGradOut * strideHeight);
                uint32_t wK = (wIm - wGradOut * strideWidth);
                if (hK - Simt::UintDiv(hK, mDH_, shiftDH_) * dilationHeight == 0 && 
                        wK - Simt::UintDiv(wK, mDW_, shiftDW_) * dilationWidth == 0) {
                    hK = Simt::UintDiv(hK, mDH_, shiftDH_);
                    wK = Simt::UintDiv(wK, mDW_, shiftDW_);
                    uint32_t gradOutIdx =
                        (((cIm * kernelSizeH + hK) * kernelSizeW + wK) * heightGradOut +
                        hGradOut) *
                            widthGradOut +
                        wGradOut;
                    valv += inputXGmAddr[gradOutIdx];
                }
            }
        }
        yGmAddr[idx] = static_cast<D_T>(valv);
    }
}

template <typename ACC_T, typename D_T>
__aicore__ inline void Col2imSimt<ACC_T, D_T>::Process()
{
    uint32_t shiftGW_, mGW_, shiftGWH_, mGWH_, shiftSW_, mSW_, shiftSH_, mSH_, shiftDH_, mDH_, shiftDW_, mDW_;
    uint32_t gradInW = static_cast<uint32_t>(tiling_->outputSizeW);
    uint32_t gradInWH = static_cast<uint32_t>(tiling_->outputSizeW * tiling_->outputSizeH);
    uint32_t strideWidth = static_cast<uint32_t>(tiling_->strideW);
    uint32_t strideHeight = static_cast<uint32_t>(tiling_->strideH);
    uint32_t dilationHeight = static_cast<uint32_t>(tiling_->dilationH);
    uint32_t dilationWidth = static_cast<uint32_t>(tiling_->dilationW);

    GetUintDivMagicAndShift(mGW_, shiftGW_, gradInW);
    GetUintDivMagicAndShift(mGWH_, shiftGWH_, gradInWH);
    GetUintDivMagicAndShift(mSW_, shiftSW_, strideWidth);
    GetUintDivMagicAndShift(mSH_, shiftSH_, strideHeight);
    GetUintDivMagicAndShift(mDH_, shiftDH_, dilationHeight);
    GetUintDivMagicAndShift(mDW_, shiftDW_, dilationWidth);
    Simt::VF_CALL<Col2imSimtCompute<ACC_T, D_T>>(
        Simt::Dim3{VF_MAX_THREAD_NUM, 1, 1}, (__gm__ D_T*)(inputXGm_.GetPhyAddr()), (__gm__ D_T*)(yGm_.GetPhyAddr()),
        static_cast<uint32_t>(tiling_->totalLength), static_cast<uint32_t>(tiling_->outputSizeH), 
        static_cast<uint32_t>(tiling_->outputSizeW), static_cast<uint32_t>(tiling_->kernelSizeH), 
        static_cast<uint32_t>(tiling_->kernelSizeW), static_cast<uint32_t>(tiling_->paddingH), 
        static_cast<uint32_t>(tiling_->paddingW), static_cast<uint32_t>(tiling_->strideH), 
        static_cast<uint32_t>(tiling_->strideW), static_cast<uint32_t>(tiling_->dilationH), 
        static_cast<uint32_t>(tiling_->dilationW), static_cast<uint32_t>(tiling_->colH), 
        static_cast<uint32_t>(tiling_->colW), shiftGW_, mGW_, shiftGWH_, mGWH_, 
        shiftSW_, mSW_, shiftSH_, mSH_, shiftDH_, mDH_, shiftDW_, mDW_);
}
}
#endif