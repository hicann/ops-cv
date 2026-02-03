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
 * \file test_roi_align_rotated_grad.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include <cstdint>
#include "../../../op_host/roi_align_rotated_grad_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void roi_align_rotated_grad(GM_ADDR input, GM_ADDR rois, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class roi_align_rotated_grad_test : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    cout << "roi_align_rotated_grad_test SetUp\n"
         << endl;
    const string cmd = "cp -rf " + dataPath + " ./";
    system(cmd.c_str());
    system("chmod -R 755 ./roi_align_rotated_grad_data/");
  }
  static void TearDownTestCase()
  {
    cout << "roi_align_rotated_grad_test TearDown\n"
         << endl;
  }

private:
  const static std::string dataPath;
};

const std::string roi_align_rotated_grad_test::dataPath = "../../../../objdetect/roi_align_rotated_grad/op_kernel/roi_align_rotated_data_grad";

TEST_F(roi_align_rotated_grad_test, test_case_0)
{
  size_t gradOutputByteSize = 8 * 2 * 2 * 6 * sizeof(float);
  size_t roisByteSize = 6 * 8 * sizeof(float);
  size_t gradInputByteSize = 5 * 1 * 1 * 6 * sizeof(float);
  size_t tiling_data_size = sizeof(RoiAlignRotatedGradTilingData);

  uint8_t *grad_input = (uint8_t *)AscendC::GmAlloc(gradInputByteSize);
  uint8_t *rois = (uint8_t *)AscendC::GmAlloc(roisByteSize);
  uint8_t *grad_output = (uint8_t *)AscendC::GmAlloc(gradOutputByteSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(gradOutputByteSize);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

  uint32_t numBlocks = 1;

  RoiAlignRotatedGradTilingData* tilingData = reinterpret_cast<RoiAlignRotatedGradTilingData*>(tiling);
  tilingData->coreRoisNums = 0;
  tilingData->coreRoisTail = 8;
  tilingData->boxSize = 8;
  tilingData->pooledHeight = 2;
  tilingData->pooledWidth = 2;
  tilingData->batchSize = 5;
  tilingData->channelNum = 6;
  tilingData->width = 1;
  tilingData->height = 1;
  tilingData->aligned = 1;
  tilingData->clockwise = 1;
  tilingData->samplingRatio = 1;
  tilingData->spatialScale = 1;
  tilingData->coreNum = 64;

  ICPU_SET_TILING_KEY(-707816656);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(roi_align_rotated_grad, 1, grad_output, rois, grad_input, workspace, tiling);

  AscendC::GmFree(grad_output);
  AscendC::GmFree(rois);
  AscendC::GmFree(grad_input);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}