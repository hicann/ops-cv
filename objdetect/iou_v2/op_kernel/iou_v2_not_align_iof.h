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
 * \file iou_v2_not_align_iof.h
 * \brief
 */
#ifndef _IOU_V2_NOT_ALIGN_IOF_H
#define _IOU_V2_NOT_ALIGN_IOF_H

#include "iou_v2_common.h"

namespace IouV2
{
    using namespace AscendC;

    template <typename inType>
    class KernelIofV2NotAlign : public KernelIouV2NotAlignBase<inType, KernelIofV2NotAlign>
    {
    public:
        __aicore__ inline void Compute(uint32_t bBoxLoop)
        {
            LocalTensor<float> boxLocal = this->inQue.template AllocTensor<float>();
            LocalTensor<float> tmpBuffer2 = this->tmpTensor2.template Get<float>();
            this->NotAlignComputePrepareBox(boxLocal, tmpBuffer2);
            PipeBarrier<PIPE_V>();
            Max(boxLocal, boxLocal, boxLocal[this->quadMulLen], this->mulLen);
            Max(boxLocal[this->mulLen], boxLocal[this->mulLen], boxLocal[this->pentaMulLen], this->mulLen);
            Min(boxLocal[this->doubleMulLen], boxLocal[this->doubleMulLen], boxLocal[this->hexaMulLen], this->mulLen);
            Min(boxLocal[this->tripMulLen], boxLocal[this->tripMulLen], boxLocal[this->septaMulLen], this->mulLen);
            Adds(boxLocal[this->doubleMulLen], boxLocal[this->doubleMulLen], this->eps, this->doubleMulLen);
            PipeBarrier<PIPE_V>();
            SubRelu(boxLocal, boxLocal[this->doubleMulLen], boxLocal, this->doubleMulLen);

            Sub(boxLocal[this->quadMulLen], boxLocal[this->hexaMulLen], boxLocal[this->quadMulLen], this->doubleMulLen);
            PipeBarrier<PIPE_V>();
            Adds(boxLocal[this->quadMulLen], boxLocal[this->quadMulLen], this->eps, this->doubleMulLen);
            PipeBarrier<PIPE_V>();
            Mul(boxLocal[this->quadMulLen], boxLocal[this->quadMulLen], boxLocal[this->pentaMulLen], this->mulLen);
            Mul(boxLocal[this->septaMulLen], boxLocal, boxLocal[this->mulLen], this->mulLen);

            LocalTensor<float> outLocal = this->outQue.template AllocTensor<float>();
            PipeBarrier<PIPE_V>();
            Div(outLocal, boxLocal[this->septaMulLen], boxLocal[this->quadMulLen], this->mulLen);

            this->outQue.template EnQue<float>(outLocal);
            this->inQue.FreeTensor(boxLocal);
        }
    };

} // namespace IouV2
#endif // _IOU_V2_NOT_ALIGN_IOF_H
