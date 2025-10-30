/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_roi_align_rotated.cpp
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
#include "../../../op_host/roi_align_rotated_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void roi_align_rotated(GM_ADDR input, GM_ADDR rois, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class roi_align_rotated_test : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    cout << "roi_align_rotated_test SetUp\n"
         << endl;
    const string cmd = "cp -rf " + dataPath + " ./";
    system(cmd.c_str());
    system("chmod -R 755 ./roi_align_rotated_data/");
  }
  static void TearDownTestCase()
  {
    cout << "roi_align_rotated_test TearDown\n"
         << endl;
  }

private:
  const static std::string dataPath;
};
const std::string roi_align_rotated_test::dataPath = "../../../../objdetect/roi_align_rotated/op_kernel/roi_align_rotated_data";

TEST_F(roi_align_rotated_test, test_case_0)
{
  size_t inputByteSize = 1 * 8 * 8 * 8 * sizeof(float);
  size_t roisByteSize = 6 * 8 * sizeof(float);
  size_t outputByteSize = 8 * 8 * 2 * 2 * sizeof(float);
  size_t tiling_data_size = sizeof(RoiAlignRotatedTilingData);

  uint8_t *input = (uint8_t *)AscendC::GmAlloc(inputByteSize);
  uint8_t *rois = (uint8_t *)AscendC::GmAlloc(roisByteSize);
  uint8_t *output = (uint8_t *)AscendC::GmAlloc(outputByteSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 1;

  RoiAlignRotatedTilingData* tilingData = reinterpret_cast<RoiAlignRotatedTilingData*>(tiling);
  tilingData->aligned = 0;
  tilingData->clockwise = 0;
  tilingData->blockDim = 1;
  tilingData->rois_num_per_Lcore = 8;
  tilingData->rois_num_per_Score = 0;
  tilingData->Lcore_num = 1;
  tilingData->Score_num = 63;
  tilingData->input_buffer_size = 32;
  tilingData->tileNum = 8;
  tilingData->batch_size = 1;
  tilingData->channels = 8;
  tilingData->channels_aligned = 8;
  tilingData->input_h = 8;
  tilingData->input_w = 8;
  tilingData->rois_num_aligned = 8;
  tilingData->tail_num = 0;
  tilingData->spatial_scale = 1;
  tilingData->sampling_ratio = 1;
  tilingData->pooled_height = 2;
  tilingData->pooled_width = 2;
  tilingData->ub_total_size = 262144;

  ICPU_SET_TILING_KEY(1);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(roi_align_rotated, blockDim, input, rois, output, workspace, tiling);

  AscendC::GmFree(input);
  AscendC::GmFree(rois);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}