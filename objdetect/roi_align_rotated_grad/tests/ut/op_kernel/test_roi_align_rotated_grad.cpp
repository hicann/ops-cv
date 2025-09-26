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
  const static std::string rootPath;
  const static std::string dataPath;
};

const std::string IsFiniteTest::rootPath = "../../../../../../../";
const std::string IsFiniteTest::dataPath = rootPath + "ops/objdetect/roi_align_rotated_grad/tests/ut/op_kernel/roi_align_rotated_grad_data";

TEST_F(roi_align_rotated_grad_test, test_case_0)
{
  size_t gradOutputByteSize = 2 * 8 * 2 * 2 * sizeof(float);
  size_t roisByteSize = 6 * 2 * sizeof(float);
  size_t gradInputByteSize = 1 * 8 * 8 * 8 * sizeof(float);
  size_t tiling_data_size = sizeof(RoiAlignRotatedGradTilingData);

  uint8_t *grad_input = (uint8_t *)AscendC::GmAlloc(gradInputByteSize);
  uint8_t *rois = (uint8_t *)AscendC::GmAlloc(roisByteSize);
  uint8_t *grad_output = (uint8_t *)AscendC::GmAlloc(gradOutputByteSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

  memset(workspace, 0, 4096);

  system("cd ./roi_align_rotated_data/ && python3 gen_data.py 1 8 8 8 2 6");
  system("cd ./roi_align_rotated_data/ && python3 gen_tiling.py test_case_0");

  char *path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "./roi_align_rotated_grad_data/grad_output.bin", gradOutputByteSize, grad_output, gradOutputByteSize);
  ReadFile(path + "./roi_align_rotated_grad_data/rois.bin", roisByteSize, rois, roisByteSize);
  ReadFile(path + "./roi_align_rotated_grad_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(roi_align_rotated_grad, 40, grad_output, rois, grad_input, workspace, tiling);

  AscendC::GmFree(grad_output);
  AscendC::GmFree(rois);
  AscendC::GmFree(grad_input);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(roi_align_rotated_grad_test, test_case_1)
{
  size_t gradOutputByteSize = 3 * 8 * 2 * 2 * sizeof(float);
  size_t roisByteSize = 6 * 3 * sizeof(float);
  size_t gradInputByteSize = 2 * 8 * 8 * 8 * sizeof(float);
  size_t tiling_data_size = sizeof(RoiAlignRotatedGradTilingData);

  uint8_t *grad_output = (uint8_t *)AscendC::GmAlloc(gradOutputByteSize);
  uint8_t *rois = (uint8_t *)AscendC::GmAlloc(roisByteSize);
  uint8_t *grad_input = (uint8_t *)AscendC::GmAlloc(gradInputByteSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

  memset(workspace, 0, 4096);

  system("cd ./roi_align_rotated_data/ && python3 gen_data.py 2 8 8 8 3 6");
  system("cd ./roi_align_rotated_data/ && python3 gen_tiling.py test_case_1");

  char *path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "./roi_align_rotated_grad_data/grad_output.bin", gradOutputByteSize, grad_output, gradOutputByteSize);
  ReadFile(path + "./roi_align_rotated_grad_data/rois.bin", roisByteSize, rois, roisByteSize);
  ReadFile(path + "./roi_align_rotated_grad_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(roi_align_rotated_grad, 40, grad_output, rois, grad_input, workspace, tiling);

  AscendC::GmFree(grad_output);
  AscendC::GmFree(rois);
  AscendC::GmFree(grad_input);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(roi_align_rotated_grad_test, test_case_2)
{
  size_t gradOutputByteSize = 8 * 8 * 2 * 2 * sizeof(float);
  size_t roisByteSize = 6 * 8 * sizeof(float);
  size_t gradInputByteSize = 3 * 8 * 8 * 8 * sizeof(float);
  size_t tiling_data_size = sizeof(RoiAlignRotatedGradTilingData);

  uint8_t *grad_output = (uint8_t *)AscendC::GmAlloc(gradOutputByteSize);
  uint8_t *rois = (uint8_t *)AscendC::GmAlloc(roisByteSize);
  uint8_t *grad_input = (uint8_t *)AscendC::GmAlloc(gradInputByteSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

  memset(workspace, 0, 4096);

  system("cd ./roi_align_rotated_data/ && python3 gen_data.py 3 8 8 8 8 6");
  system("cd ./roi_align_rotated_data/ && python3 gen_tiling.py test_case_2");

  char *path_ = get_current_dir_name();
  string path(path_);
  ReadFile(path + "./roi_align_rotated_grad_data/grad_output.bin", gradOutputByteSize, grad_output, gradOutputByteSize);
  ReadFile(path + "./roi_align_rotated_grad_data/rois.bin", roisByteSize, rois, roisByteSize);
  ReadFile(path + "./roi_align_rotated_grad_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(roi_align_rotated_grad, 40, grad_output, rois, grad_input, workspace, tiling);

  AscendC::GmFree(grad_output);
  AscendC::GmFree(rois);
  AscendC::GmFree(grad_input);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}