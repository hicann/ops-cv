/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include <cstdint>
#include "../../../op_host/iou_v2_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void iou_v2(
    GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR overlap, GM_ADDR workspace, GM_ADDR tiling);

class iou_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "iou_v2 SetUp\n"
             << endl;
    }
    static void TearDownTestCase()
    {
        cout << "iou_v2_test TearDown\n"
             << endl;
    }
};

// [4, 1020], aligned=true, iou, float32
TEST_F(iou_v2_test, test_aligned_iou_fp32)
{
    size_t bboxesByteSize = 4 * 1020 * sizeof(float);
    size_t gtboxesByteSize = 4 * 1020 * sizeof(float);
    size_t overlapByteSize = 1020 * 1 * sizeof(float);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [4, 1020], aligned=true, iou, float16
TEST_F(iou_v2_test, test_aligned_iou_fp16)
{
    size_t bboxesByteSize = 4 * 1020 * sizeof(half);
    size_t gtboxesByteSize = 4 * 1020 * sizeof(half);
    size_t overlapByteSize = 1020 * 1 * sizeof(half);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [4, 1020], aligned=true, iof, float32
TEST_F(iou_v2_test, test_aligned_iof_fp32)
{
    size_t bboxesByteSize = 4 * 1020 * sizeof(float);
    size_t gtboxesByteSize = 4 * 1020 * sizeof(float);
    size_t overlapByteSize = 1020 * 1 * sizeof(float);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(14);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [4, 1020], aligned=true, iof, float16
TEST_F(iou_v2_test, test_aligned_iof_fp16)
{
    size_t bboxesByteSize = 4 * 1020 * sizeof(half);
    size_t gtboxesByteSize = 4 * 1020 * sizeof(half);
    size_t overlapByteSize = 1020 * 1 * sizeof(half);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(15);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [2020, 4], [20, 4], aligned=false, iou, float32
TEST_F(iou_v2_test, test_not_aligned_iou_fp32)
{
    size_t bboxesByteSize = 2020 * 4 * sizeof(float);
    size_t gtboxesByteSize = 20 * 4 * sizeof(float);
    size_t overlapByteSize = 2020 * 20 * sizeof(float);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(7);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [2020, 4], [20, 4], aligned=false, iou, float16
TEST_F(iou_v2_test, test_not_aligned_iou_fp16)
{
    size_t bboxesByteSize = 2020 * 4 * sizeof(half);
    size_t gtboxesByteSize = 20 * 4 * sizeof(half);
    size_t overlapByteSize = 2020 * 20 * sizeof(half);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(8);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [2020, 4], [20, 4], aligned=false, iof, float32
TEST_F(iou_v2_test, test_not_aligned_iof_fp32)
{
    size_t bboxesByteSize = 2020 * 4 * sizeof(float);
    size_t gtboxesByteSize = 20 * 4 * sizeof(float);
    size_t overlapByteSize = 2020 * 20 * sizeof(float);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(17);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [2020, 4], [20, 4], aligned=false, iof, float16
TEST_F(iou_v2_test, test_not_aligned_iof_fp16)
{
    size_t bboxesByteSize = 2020 * 4 * sizeof(half);
    size_t gtboxesByteSize = 20 * 4 * sizeof(half);
    size_t overlapByteSize = 2020 * 20 * sizeof(half);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(18);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [4, 1020], aligned=true, iou, bfloat16_t
TEST_F(iou_v2_test, test_aligned_iou_bf16)
{
    size_t bboxesByteSize = 4 * 1020 * sizeof(bfloat16_t);
    size_t gtboxesByteSize = 4 * 1020 * sizeof(bfloat16_t);
    size_t overlapByteSize = 1020 * 1 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(6);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [2020, 4], [20, 4], aligned=false, iou, bfloat16_t
TEST_F(iou_v2_test, test_not_aligned_iou_bf16)
{
    size_t bboxesByteSize = 2020 * 4 * sizeof(bfloat16_t);
    size_t gtboxesByteSize = 20 * 4 * sizeof(bfloat16_t);
    size_t overlapByteSize = 2020 * 20 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(9);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [4, 1020], aligned=true, iof, bfloat16_t
TEST_F(iou_v2_test, test_aligned_iof_bf16)
{
    size_t bboxesByteSize = 4 * 1020 * sizeof(bfloat16_t);
    size_t gtboxesByteSize = 4 * 1020 * sizeof(bfloat16_t);
    size_t overlapByteSize = 1020 * 1 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(16);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// [2020, 4], [20, 4], aligned=false, iof, bfloat16_t
TEST_F(iou_v2_test, test_not_aligned_iof_bf16)
{
    size_t bboxesByteSize = 2020 * 4 * sizeof(bfloat16_t);
    size_t gtboxesByteSize = 20 * 4 * sizeof(bfloat16_t);
    size_t overlapByteSize = 2020 * 20 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(IouV2TilingData);

    uint8_t *bboxes = (uint8_t *)AscendC::GmAlloc(bboxesByteSize);
    uint8_t *gtboxes = (uint8_t *)AscendC::GmAlloc(gtboxesByteSize);
    uint8_t *overlap = (uint8_t *)AscendC::GmAlloc(overlapByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    char *path_ = get_current_dir_name();
    string path(path_);

    IouV2TilingData *tilingDatafromBin = reinterpret_cast<IouV2TilingData *>(tiling);

    ICPU_SET_TILING_KEY(19);
    ICPU_RUN_KF(iou_v2, blockDim, bboxes, gtboxes, overlap, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(bboxes);
    AscendC::GmFree(gtboxes);
    AscendC::GmFree(overlap);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

