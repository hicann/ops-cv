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
 * \file iou_v2_align_iof.h
 * \brief
 */
#ifndef _IOU_V2_ALIGN_IOF_H
#define _IOU_V2_ALIGN_IOF_H

#include "iou_v2_common.h"

namespace IouV2
{
    using namespace AscendC;

    template <typename inType>
    class KernelIofV2Align : public KernelIouV2AlignBase<inType, KernelIofV2Align>
    {
    public:
        __aicore__ inline void Init(GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR overlap,
                                    const IouV2TilingData *tilingData)
        {
            uint64_t tileLen = tilingData->tileLength;
            KernelIouV2AlignBase<inType, KernelIofV2Align>::Init(bboxes, gtboxes, overlap, tilingData, (4 * tileLen) * 4);
        }

        __aicore__ inline void Compute()
        {
            LocalTensor<float> box1Local = this->box1Que.template DeQue<float>();
            LocalTensor<float> box2Local = this->box2Que.template DeQue<float>();
            LocalTensor<float> tmpBuffer = this->tmpTensor.template Get<float>();
            Sub(tmpBuffer, box2Local[this->doubleTileLen], box2Local, this->doubleTileLen);
            Adds(tmpBuffer, tmpBuffer, this->eps, this->doubleTileLen);

            LocalTensor<float> outLocal = this->outQue.template AllocTensor<float>();
            Mul(outLocal, tmpBuffer, tmpBuffer[this->tileLen], this->tileLen);

            Max(tmpBuffer, box1Local, box2Local, this->doubleTileLen);
            Min(tmpBuffer[this->doubleTileLen], box1Local[this->doubleTileLen], box2Local[this->doubleTileLen], this->doubleTileLen);
            Adds(tmpBuffer[this->doubleTileLen], tmpBuffer[this->doubleTileLen], this->eps, this->doubleTileLen);
            SubRelu(tmpBuffer, tmpBuffer[this->doubleTileLen], tmpBuffer, this->doubleTileLen);

            Mul(tmpBuffer, tmpBuffer, tmpBuffer[this->tileLen], this->tileLen);
            Div(outLocal, tmpBuffer, outLocal, this->tileLen);

            this->outQue.template EnQue<float>(outLocal);
            this->box1Que.FreeTensor(box1Local);
            this->box2Que.FreeTensor(box2Local);
        }
    };

} // namespace IouV2
#endif // _IOU_V2_ALIGN_IOF_H
