/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 # Authors (accounts):
 # - Liu Jun <@kbryantttt>
 # - Tu Yuanhang <@TuYHAAAAAA>
 # - Zhou Jianhua<@LePenseur>
 # - Liang Yanglin <@liang-yanglin>
 # - Su Tonghua <@sutonghua>
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
 * \file roi_align.h
 * \brief
 */
#ifndef __ROI_ALIGN_V2_H__
#define __ROI_ALIGN_V2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "roi_align_v2_tiling_data.h"
#include "roi_align_v2_tiling_key.h"

namespace NsRoiAlignV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class RoiAlignV2 {
public:
    __aicore__ inline RoiAlignV2(){};

    __aicore__ inline void Init(GM_ADDR features, GM_ADDR rois, GM_ADDR output,
                                const RoiAlignV2TilingData* tiling_data);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneROI(uint32_t roiIdx);
    __aicore__ inline float BilinearInterpolate(AscendC::LocalTensor<T> &featureLocal,
                                                 float y, float x,
                                                 int32_t featW, int32_t featH);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> featureQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> roiQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;

    AscendC::GlobalTensor<T> featuresGm;
    AscendC::GlobalTensor<T> roisGm;
    AscendC::GlobalTensor<T> outputGm;

    uint32_t myRoiNum;
    uint32_t myRoiStart;

    uint32_t roiLength;
    uint32_t outRoiSize; 

    uint32_t channels;
    uint32_t height;
    uint32_t width;
    int32_t pooledHeight;
    int32_t pooledWidth;
    float spatialScale;
    int32_t samplingRatio;
};

template <typename T>
__aicore__ inline void RoiAlignV2<T>::Init(GM_ADDR features, GM_ADDR rois, GM_ADDR output,
                                const RoiAlignV2TilingData* tiling_data)
{
    uint32_t blockIdx = AscendC::GetBlockIdx();
    // Calculate how many ROIs this core processes
    if (blockIdx < tiling_data->tailRoiNum) {
        this->myRoiNum = tiling_data->bigTotalRois;
    } else {
        this->myRoiNum = tiling_data->baseRoisPerCore;
    }
    // Calculate starting ROI index for this core
    if (blockIdx < tiling_data->tailRoiNum) {
        this->myRoiStart = myRoiNum * blockIdx;
    } else {
        this->myRoiStart = myRoiNum * blockIdx + tiling_data->tailRoiNum;
    }
    this->roiLength = tiling_data->roiLength;
    this->outRoiSize = tiling_data->outRoiSize;
    this->channels = tiling_data->channels;
    this->height = tiling_data->height;
    this->width = tiling_data->width;
    this->pooledHeight = tiling_data->pooledHeight;
    this->pooledWidth = tiling_data->pooledWidth;
    this->spatialScale = tiling_data->spatialScale;
    this->samplingRatio = tiling_data->samplingRatio;
    // Set global buffers
    featuresGm.SetGlobalBuffer((__gm__ T*)features, tiling_data->featureTotalSize);
    roisGm.SetGlobalBuffer((__gm__ T*)rois + myRoiStart * this->roiLength,
                            myRoiNum * this->roiLength);
    outputGm.SetGlobalBuffer((__gm__ T*)output + myRoiStart * this->outRoiSize,
                            myRoiNum * this->outRoiSize);

    // KEY CHANGE: Initialize queues for ONE channel at a time (not entire C*H*W)
    uint32_t singleChannelSize = this->height * this->width;
    pipe.InitBuffer(featureQueue, BUFFER_NUM, singleChannelSize * sizeof(T));
    pipe.InitBuffer(roiQueue, BUFFER_NUM, this->roiLength * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->outRoiSize * sizeof(T));
}

template <typename T>
__aicore__ inline void RoiAlignV2<T>::Process()
{
    // Process each ROI assigned to this core
    for (uint32_t i = 0; i < myRoiNum; i++) {
        ProcessOneROI(i);
    }
}

template <typename T>
__aicore__ inline void RoiAlignV2<T>::ProcessOneROI(uint32_t roiIdx)
{
    // Step 1: Load ROI coordinates
    AscendC::LocalTensor<T> roiLocal = roiQueue.AllocTensor<T>();
    AscendC::DataCopyExtParams roiCopyParams{1, static_cast<uint32_t>(this->roiLength * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> roiPadParams{true, 0, 0, 0};
    AscendC::DataCopyPad(roiLocal, roisGm[roiIdx * this->roiLength], roiCopyParams, roiPadParams);

    // Extract ROI coordinates
    int32_t batchIdx = static_cast<int32_t>(roiLocal.GetValue(0));
    float roi_x1 = static_cast<float>(roiLocal.GetValue(1)) * this->spatialScale;
    float roi_y1 = static_cast<float>(roiLocal.GetValue(2)) * this->spatialScale;
    float roi_x2 = static_cast<float>(roiLocal.GetValue(3)) * this->spatialScale;
    float roi_y2 = static_cast<float>(roiLocal.GetValue(4)) * this->spatialScale;

    roiQueue.FreeTensor(roiLocal);

    // Convert to (x, y, w, h) format
    float roi_x = roi_x1;
    float roi_y = roi_y1;
    float roi_w = roi_x2 - roi_x1;
    float roi_h = roi_y2 - roi_y1;

    // Clamp to minimum size
    if (roi_w < 1.0f) roi_w = 1.0f;
    if (roi_h < 1.0f) roi_h = 1.0f;

    int32_t outW = this->pooledWidth;
    int32_t outH = this->pooledHeight;
    int32_t featW = this->width;
    int32_t featH = this->height;
    int32_t outHW = outH * outW;

    float bin_w = roi_w / static_cast<float>(outW);
    float bin_h = roi_h / static_cast<float>(outH);

    // Determine sampling grid size
    int32_t samplingRatio = this->samplingRatio;
    int32_t grid_h = samplingRatio > 0 ? samplingRatio :
                    static_cast<int32_t>((roi_h / static_cast<float>(outH)) + 0.99f);
    int32_t grid_w = samplingRatio > 0 ? samplingRatio :
                    static_cast<int32_t>((roi_w / static_cast<float>(outW)) + 0.99f);

    if (grid_h < 1) grid_h = 1;
    if (grid_w < 1) grid_w = 1;

    float count = static_cast<float>(grid_h * grid_w);

    // Allocate output tensor
    AscendC::LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();

    // KEY OPTIMIZATION: Process one channel at a time
    uint32_t featMapOffset = batchIdx * this->channels * featH * featW;

    for (int32_t c = 0; c < this->channels; ++c) {
        // Copy only ONE channel to local memory
        AscendC::LocalTensor<T> featureLocal = featureQueue.AllocTensor<T>();
        uint32_t channelOffset = featMapOffset + c * featH * featW;
        uint32_t singleChannelSize = featH * featW;

        AscendC::DataCopyExtParams featureCopyParams{1, static_cast<uint32_t>(singleChannelSize * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> featurePadParams{true, 0, 0, 0};
        AscendC::DataCopyPad(featureLocal, featuresGm[channelOffset], featureCopyParams, featurePadParams);

        // Process each output position for this channel
        int32_t outCOffset = c * outHW;

        for (int32_t ph = 0; ph < outH; ++ph) {
            for (int32_t pw = 0; pw < outW; ++pw) {
                // Calculate bin boundaries
                float bin_start_y = roi_y + static_cast<float>(ph) * bin_h;
                float bin_start_x = roi_x + static_cast<float>(pw) * bin_w;

                float acc = 0.0f;

                // Sample within the bin
                for (int32_t iy = 0; iy < grid_h; ++iy) {
                    float yy = bin_start_y + (static_cast<float>(iy) + 0.5f) * (bin_h / static_cast<float>(grid_h));

                    for (int32_t ix = 0; ix < grid_w; ++ix) {
                        float xx = bin_start_x + (static_cast<float>(ix) + 0.5f) * (bin_w / static_cast<float>(grid_w));

                        // Bilinear interpolation
                        float val = BilinearInterpolate(featureLocal, yy, xx, featW, featH);
                        acc += val;
                    }
                }

                // Average pooling
                int32_t outIdx = outCOffset + ph * outW + pw;
                outputLocal.SetValue(outIdx, static_cast<T>(acc / count));
            }
        }

        featureQueue.FreeTensor(featureLocal);
    }

    // Copy output back to GM
    AscendC::DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(this->outRoiSize * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(outputGm[roiIdx * this->outRoiSize], outputLocal, outCopyParams);

    outQueue.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline float RoiAlignV2<T>::BilinearInterpolate(AscendC::LocalTensor<T> &featureLocal,
                                                 float y, float x,
                                                 int32_t featW, int32_t featH)
{
    // Boundary check
    float featH_f = static_cast<float>(featH);
    float featW_f = static_cast<float>(featW);

    if (y < -1.0f || y > featH_f || x < -1.0f || x > featW_f) {
        return 0.0f;
    }

    // Clamp to valid range
    if (y < 0.0f) y = 0.0f;
    if (y > featH_f - 1.0f) y = featH_f - 1.0f;
    if (x < 0.0f) x = 0.0f;
    if (x > featW_f - 1.0f) x = featW_f - 1.0f;

    // Get integer coordinates
    int32_t y0 = static_cast<int32_t>(y);
    int32_t x0 = static_cast<int32_t>(x);
    int32_t y1 = (y0 + 1 < featH) ? (y0 + 1) : y0;
    int32_t x1 = (x0 + 1 < featW) ? (x0 + 1) : x0;

    // Interpolation weights
    float ly = y - static_cast<float>(y0);
    float lx = x - static_cast<float>(x0);
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    // Load 4 corner values from local memory
    T v00 = featureLocal.GetValue(y0 * featW + x0);
    T v01 = featureLocal.GetValue(y0 * featW + x1);
    T v10 = featureLocal.GetValue(y1 * featW + x0);
    T v11 = featureLocal.GetValue(y1 * featW + x1);

    // Convert to float for computation
    float v00_f = static_cast<float>(v00);
    float v01_f = static_cast<float>(v01);
    float v10_f = static_cast<float>(v10);
    float v11_f = static_cast<float>(v11);

    // Bilinear interpolation
    float result = (hy * hx) * v00_f + (hy * lx) * v01_f +
                    (ly * hx) * v10_f + (ly * lx) * v11_f;

    return result;
}

} // namespace NsRoiAlignV2
#endif // ROI_ALIGN_H
