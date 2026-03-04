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
 * \file test_roi_pooling_with_arg_max.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_kernel/arch35/roi_pooling_with_arg_max_tiling_key.h"
#include "../../../op_kernel/arch35/roi_pooling_with_arg_max_tiling_data.h"

using namespace std;

class roi_pooling_with_arg_max_test : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    cout << "roi_pooling_with_arg_max_test SetUp\n"
         << endl;
    const string cmd = "cp -rf " + dataPath + " ./";
    system(cmd.c_str());
    system("chmod -R 755 ./roi_pooling_with_arg_max_data/");
  }
  static void TearDownTestCase()
  {
    cout << "roi_pooling_with_arg_max_test TearDown\n"
         << endl;
    system("rm -rf roi_pooling_with_arg_max_data");
  }

private:
  const static std::string dataPath;
};
const std::string roi_pooling_with_arg_max_test::dataPath = "../../../../objdetect/roi_pooling_with_arg_max/tests/ut/op_kernel/roi_pooling_with_arg_max_data";

inline static int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

TEST_F(roi_pooling_with_arg_max_test, test_simt_float32_case0)
{
    std::vector<int64_t> inputXShape = {2, 16, 25, 42};
    std::vector<int64_t> inputRoisShape = {2, 5};
    std::vector<int64_t> outputShape = {2, 16, 3, 3};
    std::string caseName = "test_simt_float32_case0";
    uint64_t tilingKey = GET_TPL_TILING_KEY(ROI_POOLING_WITH_ARG_MAX_TPL_FP32);
    size_t tilingDataSize = sizeof(RoiPoolingWithArgMaxRegBaseTilingData);

    size_t inputXSize = GetShapeSize(inputXShape) * sizeof(float);
    size_t inputRoisSize = GetShapeSize(inputRoisShape) * sizeof(float);
    size_t inputRoiActualNum = 1 * sizeof(int32_t);
    size_t outputYSize = GetShapeSize(outputShape) * sizeof(float);
    size_t outputIdxSize = GetShapeSize(outputShape) * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXSize);
    uint8_t *rois = (uint8_t *)AscendC::GmAlloc(inputRoisSize);
    uint8_t *roiActualNum = (uint8_t *)AscendC::GmAlloc(inputRoiActualNum);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYSize);
    uint8_t *argmax = (uint8_t *)AscendC::GmAlloc(outputIdxSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(4096 * 16);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    RoiPoolingWithArgMaxRegBaseTilingData* tilingDataFromBin =
        reinterpret_cast<RoiPoolingWithArgMaxRegBaseTilingData*>(tiling);
    tilingDataFromBin->channels = 16;
    tilingDataFromBin->fmHeight = 25;
    tilingDataFromBin->fmWidth = 42;
    tilingDataFromBin->roiNumber = 2;
    tilingDataFromBin->poolH = 3;
    tilingDataFromBin->poolW = 3;
    tilingDataFromBin->spatialH = 1.0f;
    tilingDataFromBin->spatialW = 1.0f;

    uint32_t blockDim = 1;

    std::string path = std::filesystem::current_path().string();
    ReadFile(path + "/roi_pooling_with_arg_max_data/x.bin", inputXSize, x, inputXSize);
    ReadFile(path + "/roi_pooling_with_arg_max_data/rois.bin", inputRoisSize, rois, inputRoisSize);

    int32_t roiNum = 2;
    (void)memcpy_s(roiActualNum, inputRoiActualNum, &roiNum, sizeof(roiNum));

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(roi_pooling_with_arg_max<ROI_POOLING_WITH_ARG_MAX_TPL_FP32>, blockDim, x, rois, roiActualNum, y, argmax,
                workspace, tiling);
    
    WriteFile(path + "/roi_pooling_with_arg_max_data/cce_cpu_y.bin", y, outputYSize);
    WriteFile(path + "/roi_pooling_with_arg_max_data/cce_cpu_argmax.bin", argmax, outputIdxSize);

    string cmd = "cmp " + path + "/roi_pooling_with_arg_max_data/cce_cpu_y.bin " + path + "/roi_pooling_with_arg_max_data/y_golden.bin";
    EXPECT_EQ(system(cmd.c_str()), 0);
    cmd = "cmp " + path + "/roi_pooling_with_arg_max_data/cce_cpu_argmax.bin " + path + "/roi_pooling_with_arg_max_data/argmax_golden.bin";
    EXPECT_EQ(system(cmd.c_str()), 0);
    
    AscendC::GmFree(x);
    AscendC::GmFree(rois);
    AscendC::GmFree(roiActualNum);
    AscendC::GmFree(y);
    AscendC::GmFree(argmax);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
