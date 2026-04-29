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
 * \file test_roi_pooling_with_arg_max.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kernel_tiling/kernel_tiling.h"
#include "objdetect/roi_pooling_with_arg_max/op_kernel/roi_pooling_with_arg_max.cpp"
#include "../../../op_kernel/arch35/roi_pooling_with_arg_max_tiling_data.h"

using namespace std;

template <uint64_t dType>
__global__ __aicore__ void roi_pooling_with_arg_max(GM_ADDR x, GM_ADDR rois, GM_ADDR roi_actual_num, GM_ADDR y,
                                                    GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling);

class roi_pooling_with_arg_max_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "roi_pooling_with_arg_max SetUp\n"
             << endl;
    }
    static void TearDownTestCase()
    {
        cout << "roi_pooling_with_arg_max_test TearDown\n"
             << endl;
    }
};

TEST_F(roi_pooling_with_arg_max_test, test_roi_pooling_with_arg_max_950_fp16)
{
    size_t xByteSize = 2 * 2 * 6 * 8 * sizeof(float);
    size_t roisByteSize = 2 * 5 * sizeof(float);
    size_t roiActualNumByteSize = 2 * 5 * sizeof(int32_t);
    size_t yByteSize = 2 * 2 * 2 * 2 * sizeof(float);
    size_t argmaxByteSize = 2 * 2 * 2 * 2 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(RoiPoolingWithArgMaxRegBaseTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xByteSize);
    uint8_t *rois = (uint8_t *)AscendC::GmAlloc(roisByteSize);
    uint8_t *roiActualNum = (uint8_t *)AscendC::GmAlloc(roiActualNumByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yByteSize);
    uint8_t *argmax = (uint8_t *)AscendC::GmAlloc(argmaxByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    RoiPoolingWithArgMaxRegBaseTilingData* tilingData = reinterpret_cast<RoiPoolingWithArgMaxRegBaseTilingData*>(tiling);
    tilingData->channels = 2;
    tilingData->fmHeight = 6;
    tilingData->fmWidth = 8;
    tilingData->roiNumber = 2;
    tilingData->poolH = 2;
    tilingData->poolW = 2;
    tilingData->spatialH = 1.0;
    tilingData->spatialW = 1.0;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(roi_pooling_with_arg_max<0>, numBlocks, x, rois, roiActualNum, argmax, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(rois);
    AscendC::GmFree(roiActualNum);
    AscendC::GmFree(argmax);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}