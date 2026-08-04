/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grid_unnormal.h
 * \brief GridUnnormal Kernel — GridUnnormalKernel<T>（arch35 / RegBase 寄存器编程）
 *
 * 逐元素:
 *   t        = (grid + 1) * 0.5
 *   pos_base = align_corners ? t * (assist - 1) : t * assist - 0.5
 *   position = floor(pos_base)            -> int32 输出
 *   diff     = pos_base - floor(pos_base) -> 与 grid 同 dtype 输出
 *
 * RegBase 范式：GM↔UB 仍用 TQue + DataCopyPad 搬运；向量计算在 __VEC_SCOPE__ 内用
 * MicroAPI 寄存器算子（RegTensor / MaskReg / Adds / Muls / Mul / Sub / Cast）完成，
 * 统一 fp32 中间计算（fp16 由 LoadOneTensorForDtypeT 载入即升 fp32、StoreOneTensorForDtypeT
 * 存回即降 fp16）。position 使用 CAST_FLOOR trait 保证负数坐标按 floor 语义下取整。
 */
#ifndef GRID_UNNORMAL_H
#define GRID_UNNORMAL_H

#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"
#include "grid_unnormal_tiling_data.h"

namespace NsGridUnnormal {
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;

// float -> int32 下取整（floor，负数向 -inf）
constexpr AscendC::MicroAPI::CastTrait kCastF32ToI32Floor = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_FLOOR};

// int32 -> float（整数值精确表示，round mode 不敏感）
constexpr AscendC::MicroAPI::CastTrait kCastI32ToF32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// RegBase-native（DAV_3510）：向量寄存器 256B；fp32 通道每拍 64 元素。
constexpr uint32_t kVlBytes = 256;
constexpr uint32_t kFp32PerLoop = kVlBytes / sizeof(float);

template <typename T>
class GridUnnormalKernel {
public:
    static constexpr int32_t kBufNum = 2;

    __aicore__ inline GridUnnormalKernel() {}

    __aicore__ inline void Init(GM_ADDR grid, GM_ADDR assist, GM_ADDR diff, GM_ADDR position,
                                const GridUnnormalTilingData* td)
    {
        totalNum_ = td->totalNum;
        perCoreNum_ = td->perCoreNum;
        ubFactor_ = td->ubFactor;
        alignCorners_ = (td->alignCorners != 0);

        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        coreStart_ = blockIdx * perCoreNum_;
        if (coreStart_ >= totalNum_) {
            coreLen_ = 0;
        } else {
            coreLen_ = perCoreNum_;
            if (coreStart_ + coreLen_ > totalNum_) {
                coreLen_ = totalNum_ - coreStart_;
            }
        }

        gridGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grid));
        assistGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(assist));
        diffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(diff));
        posGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(position));

        pipe_.InitBuffer(inQueGrid_, kBufNum, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(inQueAssist_, kBufNum, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(outQueDiff_, kBufNum, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(outQuePos_, kBufNum, ubFactor_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (coreLen_ <= 0) {
            return;
        }
        for (int64_t off = 0; off < coreLen_; off += ubFactor_) {
            const int64_t len = (off + ubFactor_ > coreLen_) ? (coreLen_ - off) : ubFactor_;
            const int64_t gmOff = coreStart_ + off;
            CopyIn(gmOff, len);
            Compute(len);
            CopyOut(gmOff, len);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t gmOff, int64_t len)
    {
        LocalTensor<T> gridL = inQueGrid_.template AllocTensor<T>();
        LocalTensor<T> assistL = inQueAssist_.template AllocTensor<T>();

        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(T)));
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPadExtParams<T> pad; // isPad=false, 尾段非对齐由 DataCopyPad 处理

        DataCopyPad(gridL, gridGm_[gmOff], params, pad);
        DataCopyPad(assistL, assistGm_[gmOff], params, pad);
        inQueGrid_.EnQue(gridL);
        inQueAssist_.EnQue(assistL);
    }

    __aicore__ inline void Compute(int64_t len)
    {
        if (alignCorners_) {
            ComputeImpl<true>(len);
        } else {
            ComputeImpl<false>(len);
        }
    }

    template <bool AlignCorners>
    __aicore__ inline void ComputeImpl(int64_t len)
    {
        LocalTensor<T> gridL = inQueGrid_.template DeQue<T>();
        LocalTensor<T> assistL = inQueAssist_.template DeQue<T>();
        LocalTensor<T> diffL = outQueDiff_.template AllocTensor<T>();
        LocalTensor<int32_t> posL = outQuePos_.template AllocTensor<int32_t>();

        __local_mem__ T* gridUb = reinterpret_cast<__local_mem__ T*>(gridL.GetPhyAddr());
        __local_mem__ T* assistUb = reinterpret_cast<__local_mem__ T*>(assistL.GetPhyAddr());
        __local_mem__ T* diffUb = reinterpret_cast<__local_mem__ T*>(diffL.GetPhyAddr());
        __local_mem__ int32_t* posUb = reinterpret_cast<__local_mem__ int32_t*>(posL.GetPhyAddr());

        const uint16_t repeatTimes = static_cast<uint16_t>((len + kFp32PerLoop - 1) / kFp32PerLoop);
        uint32_t sreg = static_cast<uint32_t>(len);

        __VEC_SCOPE__
        {
            MaskReg mask;
            RegTensor<float> gReg;
            RegTensor<float> aReg;
            RegTensor<float> posReg;
            RegTensor<float> floorReg;
            RegTensor<int32_t> posIReg;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                const uint32_t off = i * kFp32PerLoop;
                mask = UpdateMask<float>(sreg);
                ops::LoadOneTensorForDtypeT<T>(gridUb, gReg, mask, off);
                ops::LoadOneTensorForDtypeT<T>(assistUb, aReg, mask, off);

                // t = (grid + 1) * 0.5
                Adds(gReg, gReg, 1.0f, mask);
                Muls(gReg, gReg, 0.5f, mask);
                if constexpr (AlignCorners) {
                    // pos_base = t * (assist - 1)
                    Adds(aReg, aReg, -1.0f, mask);
                    Mul(posReg, gReg, aReg, mask);
                } else {
                    // pos_base = t * assist - 0.5
                    Mul(posReg, gReg, aReg, mask);
                    Adds(posReg, posReg, -0.5f, mask);
                }

                // position = floor(pos_base) (int32)；diff = pos_base - floor
                MicroAPI::Cast<int32_t, float, kCastF32ToI32Floor>(posIReg, posReg, mask);
                MicroAPI::Cast<float, int32_t, kCastI32ToF32>(floorReg, posIReg, mask);
                Sub(posReg, posReg, floorReg, mask);

                ops::StoreOneTensorForDtypeT<T>(diffUb, posReg, mask, off); // diff (fp32->T)
                MicroAPI::DataCopy<int32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(posUb + off, posIReg, mask);
            }
        }

        inQueGrid_.FreeTensor(gridL);
        inQueAssist_.FreeTensor(assistL);
        outQueDiff_.EnQue(diffL);
        outQuePos_.EnQue(posL);
    }

    __aicore__ inline void CopyOut(int64_t gmOff, int64_t len)
    {
        LocalTensor<T> diffL = outQueDiff_.template DeQue<T>();
        LocalTensor<int32_t> posL = outQuePos_.template DeQue<int32_t>();

        DataCopyExtParams dParams;
        dParams.blockCount = 1;
        dParams.blockLen = static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(T)));
        dParams.srcStride = 0;
        dParams.dstStride = 0;
        DataCopyPad(diffGm_[gmOff], diffL, dParams);

        DataCopyExtParams pParams;
        pParams.blockCount = 1;
        pParams.blockLen = static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(int32_t)));
        pParams.srcStride = 0;
        pParams.dstStride = 0;
        DataCopyPad(posGm_[gmOff], posL, pParams);

        outQueDiff_.FreeTensor(diffL);
        outQuePos_.FreeTensor(posL);
    }

    TPipe pipe_;
    GlobalTensor<T> gridGm_;
    GlobalTensor<T> assistGm_;
    GlobalTensor<T> diffGm_;
    GlobalTensor<int32_t> posGm_;
    TQue<QuePosition::VECIN, kBufNum> inQueGrid_;
    TQue<QuePosition::VECIN, kBufNum> inQueAssist_;
    TQue<QuePosition::VECOUT, kBufNum> outQueDiff_;
    TQue<QuePosition::VECOUT, kBufNum> outQuePos_;

    int64_t totalNum_ = 0;
    int64_t perCoreNum_ = 0;
    int64_t ubFactor_ = 0;
    int64_t coreStart_ = 0;
    int64_t coreLen_ = 0;
    bool alignCorners_ = false;
};

} // namespace NsGridUnnormal

#endif // GRID_UNNORMAL_H
