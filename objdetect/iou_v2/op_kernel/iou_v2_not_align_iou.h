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
 * \file iou_v2_not_align_iou.h
 * \brief
 */
#ifndef _IOU_V2_NOT_ALIGN_IOU_H
#define _IOU_V2_NOT_ALIGN_IOU_H

#include "iou_v2_common.h"

namespace IouV2
{
    using namespace AscendC;

    template <typename inType>
    class KernelIouV2NotAlign : public KernelIouV2NotAlignBase<inType, KernelIouV2NotAlign>
    {
    public:
        __aicore__ inline void Compute(uint32_t bBoxLoop)
        {
            LocalTensor<float> boxLocal = this->inQue.template AllocTensor<float>();
            LocalTensor<float> tmpBuffer1 = this->tmpTensor1.template Get<float>();
            LocalTensor<float> tmpBuffer2 = this->tmpTensor2.template Get<float>();
            this->NotAlignComputePrepareBox(boxLocal, tmpBuffer2);

            Sub(tmpBuffer1, tmpBuffer1[this->doubleTotalLen], tmpBuffer1, this->doubleTotalLen);
            Adds(tmpBuffer1, tmpBuffer1, this->eps, this->doubleTotalLen);
            Mul(tmpBuffer1[this->doubleTotalLen], tmpBuffer1, tmpBuffer1[this->totalLen], this->loopTileLen);
            Mul(tmpBuffer1[this->totalLen1], tmpBuffer1[this->loopTileLen], tmpBuffer1[this->totalLen2], this->tileLen);

            Max(boxLocal, boxLocal, boxLocal[this->quadMulLen], this->mulLen);
            Max(boxLocal[this->mulLen], boxLocal[this->mulLen], boxLocal[this->pentaMulLen], this->mulLen);
            Min(boxLocal[this->doubleMulLen], boxLocal[this->doubleMulLen], boxLocal[this->hexaMulLen], this->mulLen);
            Min(boxLocal[this->tripMulLen], boxLocal[this->tripMulLen], boxLocal[this->septaMulLen], this->mulLen);
            Adds(boxLocal[this->doubleMulLen], boxLocal[this->doubleMulLen], this->eps, this->doubleMulLen);
            PipeBarrier<PIPE_V>();
            SubRelu(boxLocal, boxLocal[this->doubleMulLen], boxLocal, this->doubleMulLen);

            uint8_t dstRepStride = this->loopTileLen / 8; // 8个数组成一个block
            uint8_t bBoxRepTimes = this->tileLen / 8;     // 8个数组成一个block
            uint16_t gtBoxRepTimes = this->loopTileLen / 8; // 8个数组成一个block
            BrcbRepeatParams gtBoxBrcbRepParams{gtBoxRepTimes, static_cast<uint16_t>(this->loopTileLen)};
            Adds(boxLocal[this->quadMulLen], tmpBuffer1[this->doubleTotalLen], (float)0.0, this->loopTileLen, this->tileLen, {1, 1, dstRepStride, 0});
            for (uint16_t repTime = 0; repTime < gtBoxRepTimes; ++repTime)
            {
                Brcb(boxLocal[this->pentaMulLen + repTime * 8], tmpBuffer1[this->totalLen1], bBoxRepTimes, gtBoxBrcbRepParams); // 8个数组成一个block
            }
            PipeBarrier<PIPE_V>();
            Add(boxLocal[this->hexaMulLen], boxLocal[this->quadMulLen], boxLocal[this->pentaMulLen], this->mulLen);
            PipeBarrier<PIPE_V>();
            Mul(boxLocal[this->septaMulLen], boxLocal, boxLocal[this->mulLen], this->mulLen);
            PipeBarrier<PIPE_V>();
            Sub(boxLocal, boxLocal[this->hexaMulLen], boxLocal[this->septaMulLen], this->mulLen);

            LocalTensor<float> outLocal = this->outQue.template AllocTensor<float>();
            PipeBarrier<PIPE_V>();
            Div(outLocal, boxLocal[this->septaMulLen], boxLocal, this->mulLen);

            this->outQue.template EnQue<float>(outLocal);
            this->inQue.FreeTensor(boxLocal);
        }
    };

} // namespace IouV2
#endif // _IOU_V2_NOT_ALIGN_IOU_H
