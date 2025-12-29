/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file nms_with_mask.h
 * \brief
 */
#ifndef NMS_WITH_MASK_H
#define NMS_WITH_MASK_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "nms_with_mask_tiling_data.h"
#include "nms_with_mask_tiling_key.h"

constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue
constexpr int32_t ALIGN32_FLOAT = 32 / sizeof(float);
constexpr int32_t BOX_NEED_IDX = 4;//每个box需要四个坐标
constexpr int32_t GATHER_MASK_MASK_NEED = 32 / sizeof(float);// gather每次操作32byte数据

namespace NsNMSWithMask {

using namespace AscendC;

template <typename T>
class NMSWithMask {
public:
    __aicore__ inline NMSWithMask(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const NMSWithMaskTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void IOU(LocalTensor<T>& xLocal, LocalTensor<T>& yLocal, LocalTensor<uint8_t>& zLocal);
    __aicore__ inline void GatherIdx(LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal);
    __aicore__ inline void CopyOut();
    __aicore__ inline void Compute();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;
    TBuf<QuePosition::VECCALC> baseBuffer;
    TBuf<QuePosition::VECCALC> areaBuffer;
    TBuf<QuePosition::VECCALC> mathBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<uint8_t> outputGMZ;

    int64_t rawLength;
    int64_t scoresLength;
    int64_t Align32BoxNum;
    int64_t AlignintLengh;
    
    float iou_threshold;
    float scores_threshold;

    AscendC::GatherMaskParams params;
    uint8_t gatherPattern;
};

template <typename T>
__aicore__ inline void NMSWithMask<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const NMSWithMaskTilingData* tilingData)
{   
    rawLength = tilingData->totalLength;
    scoresLength = tilingData->totalLength / BOX_NEED_IDX;
    Align32BoxNum = (scoresLength + ALIGN32_FLOAT - 1) / ALIGN32_FLOAT * ALIGN32_FLOAT;
    AlignintLengh = Align32BoxNum * BOX_NEED_IDX;
    iou_threshold = tilingData->iou_threshold;
    scores_threshold = tilingData->scores_threshold;

    inputGMX.SetGlobalBuffer((__gm__ T*)x , AlignintLengh);
    inputGMY.SetGlobalBuffer((__gm__ T*)y , Align32BoxNum);
    outputGMZ.SetGlobalBuffer((__gm__ uint8_t*)z , Align32BoxNum);//一个box对应四个输入

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, AlignintLengh * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, Align32BoxNum * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, Align32BoxNum * sizeof(uint8_t));

    pipe.InitBuffer(mathBuffer, AlignintLengh * sizeof(T));
    pipe.InitBuffer(areaBuffer, Align32BoxNum * sizeof(T));
    pipe.InitBuffer(baseBuffer,  AlignintLengh  * sizeof(T));
    this->params.src0BlockStride   = 1;
    this->params.src0RepeatStride  = 1;
    this->params.src1RepeatStride  = 1;
    this->gatherPattern = static_cast<uint8_t>(0);
}

template <typename T>
__aicore__ inline void NMSWithMask<T>::CopyIn()
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGMX[0], rawLength);
    AscendC::DataCopy(yLocal, inputGMY[0], scoresLength);
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void NMSWithMask<T>::CopyOut()
{
    AscendC::LocalTensor<uint8_t> zLocal = outputQueueZ.DeQue<uint8_t>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(scoresLength * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(outputGMZ[0], zLocal, copyParams);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void NMSWithMask<T>::GatherIdx(LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal)
{
    uint64_t rsvdCnt;// Gather接口保留参数
    int32_t mask = GATHER_MASK_MASK_NEED;// 每次repeat操作mask个元素
    for(int32_t idx = 0 ; idx < BOX_NEED_IDX ; idx++){//select四个坐标获得area存储areabuf
        gatherPattern = static_cast<uint8_t>(idx + 3);
        this->params.repeatTimes = scoresLength / 2;
        AscendC::GatherMask (dstLocal[idx * Align32BoxNum], srcLocal, gatherPattern, true, mask, params, rsvdCnt);
    }
}

template <typename T>
__aicore__ inline void NMSWithMask<T>::IOU(LocalTensor<T>& xLocal, LocalTensor<T>& yLocal, 
            LocalTensor<uint8_t>& zLocal){
    for(uint32_t i = 0 ; i < (this->rawLength / BOX_NEED_IDX) ; i++){// 初始化全部保留
        zLocal.SetValue(i, 1);
    }
    LocalTensor<T> areaLocal = areaBuffer.Get<T>();
    LocalTensor<T> mathLocal = mathBuffer.Get<T>();
    GatherIdx(mathLocal, xLocal);// mathLocal临时存储每个box的四个坐标

    Sub(mathLocal, mathLocal[Align32BoxNum * 2], mathLocal, scoresLength);
    Sub(mathLocal[Align32BoxNum], mathLocal[Align32BoxNum* 3],mathLocal[Align32BoxNum], scoresLength);
    Mul(areaLocal, mathLocal, mathLocal[Align32BoxNum], scoresLength);//mul得到area存入arealocal

    Duplicate(mathLocal, static_cast<T>(0), this->rawLength);//归零

    for(uint32_t i = 0 ; i < (scoresLength - 1) ; i++){//最后一个框处理不进行处理只最后进行判断
        if (zLocal.GetValue(i) == 0) {//该框已经被抑制
            continue;
        }
        if (yLocal.GetValue(i) < scores_threshold) {// 低于分数阈值不保留
            for (int32_t end = i; end < scoresLength; end++) {
                zLocal.SetValue(end, 0);
            }
            break;
        }

        GatherIdx(mathLocal, xLocal);

        LocalTensor<T> baseLocal = baseBuffer.Get<T>();
        for(int idx = 0 ; idx < BOX_NEED_IDX ; idx++){
            Duplicate(baseLocal[idx * Align32BoxNum], xLocal.GetValue(i * BOX_NEED_IDX + idx), Align32BoxNum);
        }
        //计算交集
        Max(baseLocal, baseLocal, mathLocal, scoresLength);
        Max(baseLocal[Align32BoxNum], baseLocal[Align32BoxNum], mathLocal[Align32BoxNum], scoresLength);
        Min(baseLocal[Align32BoxNum * 2], baseLocal[Align32BoxNum * 2], mathLocal[Align32BoxNum * 2], scoresLength);
        Min(baseLocal[Align32BoxNum * 3], baseLocal[Align32BoxNum * 3], mathLocal[Align32BoxNum * 3], scoresLength);
        Sub(baseLocal, baseLocal[Align32BoxNum * 2], baseLocal, scoresLength);
        Sub(baseLocal[Align32BoxNum], baseLocal[Align32BoxNum * 3], baseLocal[Align32BoxNum], scoresLength);
        //计算交面积
        Duplicate(mathLocal, static_cast<T>(0), this->rawLength);
        Max(baseLocal, baseLocal, mathLocal, scoresLength);
        Max(baseLocal[Align32BoxNum], baseLocal[Align32BoxNum], mathLocal[Align32BoxNum], scoresLength);
        Mul(baseLocal, baseLocal, baseLocal[Align32BoxNum], scoresLength);
        //IOU = area_inter / (area_box1 + area_box2 - area_inter)
        Duplicate(mathLocal, areaLocal.GetValue(i), scoresLength);//area_box1
        Add(mathLocal, mathLocal, areaLocal, scoresLength);//area_box1 + area_box2
        Sub(mathLocal, mathLocal, baseLocal, scoresLength);//area_box1 + area_box2 - area_inter
        Div(baseLocal, baseLocal, mathLocal, scoresLength);

        for(int t = i + 1 ; t < scoresLength ; t++){// And不支持int8和uint8_t类型，转int16，需要额外空间cast两次转换，故采用普通比较赋值
            if(baseLocal.GetValue(t) >= static_cast<T>(iou_threshold)){
                zLocal.SetValue(t, static_cast<uint8_t>(0));
            }
        }
    }
}

template <typename T>
__aicore__ inline void NMSWithMask<T>::Compute()
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<uint8_t> zLocal = outputQueueZ.AllocTensor<uint8_t>();

    IOU(xLocal, yLocal, zLocal);
    
    outputQueueZ.EnQue<uint8_t>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void NMSWithMask<T>::Process()
{
    //增加输入只有一个的判断
    if(rawLength <= BOX_NEED_IDX){
        outputGMZ.SetValue(0, static_cast<uint8_t>(1));
    }else{
        CopyIn();
        Compute();
        CopyOut();
    }
}

} // namespace NsNMSWithMask
#endif // NMS_WITH_MASK_H