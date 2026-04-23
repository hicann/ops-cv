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
 * \file roi_align_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __ROI_ALIGN_V2_TILLING_DATA_H__
#define __ROI_ALIGN_V2_TILLING_DATA_H__

struct RoiAlignV2TilingData {
    uint32_t baseRoisPerCore;
    uint32_t bigTotalRois;
    uint32_t tailRoiNum;
    uint32_t featureTotalSize;
    uint32_t roiLength;
    uint32_t featureMapSize;
    uint32_t outRoiSize;
    uint32_t numRois;
    uint32_t batch;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    int32_t pooledHeight;
    int32_t pooledWidth;
    float spatialScale;
    int32_t samplingRatio;
    // 扩展其他tilling参数
};
#endif
