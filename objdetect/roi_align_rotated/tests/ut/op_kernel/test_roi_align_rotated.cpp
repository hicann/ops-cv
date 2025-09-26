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
  size_t roisByteSize = 6 * 2 * sizeof(float);
  size_t outputByteSize = 2 * 8 * 2 * 2 * sizeof(float);
  size_t tiling_data_size = sizeof(RoiAlignRotatedTilingData);

  uint8_t *input = (uint8_t *)AscendC::GmAlloc(inputByteSize);
  uint8_t *rois = (uint8_t *)AscendC::GmAlloc(roisByteSize);
  uint8_t *output = (uint8_t *)AscendC::GmAlloc(outputByteSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

  memset(workspace, 0, 4096);
  system("cd ./roi_align_rotated_data/ && python3 gen_data.py 1 8 8 8 2 6");
  system("cd ./roi_align_rotated_data/ && python3 gen_tiling.py test_case_0");

  char *path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "./roi_align_rotated_data/input.bin", inputByteSize, input, inputByteSize);
  ReadFile(path + "./roi_align_rotated_data/rois.bin", roisByteSize, rois, roisByteSize);
  ReadFile(path + "./roi_align_rotated_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(roi_align_rotated, 40, input, rois, output, workspace, tiling);

  AscendC::GmFree(input);
  AscendC::GmFree(rois);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}

TEST_F(roi_align_rotated_test, test_case_1)
{
  size_t inputByteSize = 2 * 8 * 8 * 8 * sizeof(float);
  size_t roisByteSize = 6 * 3 * sizeof(float);
  size_t outputByteSize = 3 * 8 * 2 * 2 * sizeof(float);
  size_t tiling_data_size = sizeof(RoiAlignRotatedTilingData);

  uint8_t *input = (uint8_t *)AscendC::GmAlloc(inputByteSize);
  uint8_t *rois = (uint8_t *)AscendC::GmAlloc(roisByteSize);
  uint8_t *output = (uint8_t *)AscendC::GmAlloc(outputByteSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

  memset(workspace, 0, 4096);

  system("cd ./roi_align_rotated_data/ && python3 gen_data.py 2 8 8 8 3 6");
  system("cd ./roi_align_rotated_data/ && python3 gen_tiling.py test_case_1");

  char *path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "./roi_align_rotated_data/input.bin", inputByteSize, input, inputByteSize);
  ReadFile(path + "./roi_align_rotated_data/rois.bin", roisByteSize, rois, roisByteSize);
  ReadFile(path + "./roi_align_rotated_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(roi_align_rotated, 40, input, rois, output, workspace, tiling);

  AscendC::GmFree(input);
  AscendC::GmFree(rois);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}

TEST_F(roi_align_rotated_test, test_case_2)
{
  size_t inputByteSize = 3 * 8 * 8 * 8 * sizeof(float);
  size_t roisByteSize = 6 * 8 * sizeof(float);
  size_t outputByteSize = 8 * 8 * 2 * 2 * sizeof(float);
  size_t tiling_data_size = sizeof(RoiAlignRotatedTilingData);

  uint8_t *input = (uint8_t *)AscendC::GmAlloc(inputByteSize);
  uint8_t *rois = (uint8_t *)AscendC::GmAlloc(roisByteSize);
  uint8_t *output = (uint8_t *)AscendC::GmAlloc(outputByteSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

  memset(workspace, 0, 4096);

  system("cd ./roi_align_rotated_data/ && python3 gen_data.py 3 8 8 8 8 6");
  system("cd ./roi_align_rotated_data/ && python3 gen_tiling.py test_case_2");

  char *path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "./roi_align_rotated_data/input.bin", inputByteSize, input, inputByteSize);
  ReadFile(path + "./roi_align_rotated_data/rois.bin", roisByteSize, rois, roisByteSize);
  ReadFile(path + "./roi_align_rotated_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(roi_align_rotated, 40, input, rois, output, workspace, tiling);

  AscendC::GmFree(input);
  AscendC::GmFree(rois);
  AscendC::GmFree(output);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}