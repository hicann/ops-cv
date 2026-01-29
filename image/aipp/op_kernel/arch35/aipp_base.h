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
 * \file aipp_base.h
 * \brief aipp kernel base
 */
#ifndef AIPP_OP_KERNEL_ARCH35_BASE_H
#define AIPP_OP_KERNEL_ARCH35_BASE_H

#include "kernel_operator.h"
#include "aipp_struct.h"

namespace Aipp_Kernel {
using namespace AscendC;
constexpr uint32_t MAX_TGHREAD_NUM = 256;
constexpr uint8_t DIGIT_TWO = 2;
constexpr uint8_t NCHW_FORMAT_INDEX = 1;
constexpr uint8_t NHWC_FORMAT_INDEX = 2;

template <typename T, typename DataType>
class AippBase {
public:
    __aicore__ inline AippBase(){};
    __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR y, const AippTilingData& tilingData);
    __aicore__ inline void BaseParseTilingData(const AippTilingData& tilingData);

public:
    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<T> yGm_;

    DataType channelNum_ = 0;
    DataType batchNum_ = 0;
    DataType cropSizeH_ = 0;
    DataType cropSizeW_ = 0;

    DataType totalNum_ = 0;
    uint8_t inputFormat_ = NCHW_FORMAT_INDEX;
};

template <typename T, typename DataType>
__aicore__ inline void AippBase<T, DataType>::BaseParseTilingData(const AippTilingData& tilingData)
{
    channelNum_ = tilingData.channelNum;
    batchNum_ = tilingData.batchNum;
    cropSizeH_ = tilingData.cropSizeH;
    cropSizeW_ = tilingData.cropSizeW;
    inputFormat_ = tilingData.inputFormat;
}

template <typename T, typename DataType>
__aicore__ inline void AippBase<T, DataType>::BaseInit(GM_ADDR x, GM_ADDR y, const AippTilingData& tilingData)
{
    BaseParseTilingData(tilingData);
    totalNum_ = batchNum_ * cropSizeW_ * cropSizeH_;

    xGm_.SetGlobalBuffer((__gm__ uint8_t*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
}

}  // namespace Aipp

#endif  // AIPP_OP_KERNEL_ARCH35_BASE_H