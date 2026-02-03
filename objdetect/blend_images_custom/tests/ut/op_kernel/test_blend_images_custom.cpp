/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "test_blend_images_custom.h"
#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void blend_images_custom(GM_ADDR rgb, GM_ADDR alpha, GM_ADDR frame, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);
class blend_images_custom_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "blend_images_custom_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "blend_images_custom_test TearDown\n" << endl;
    }
};

TEST_F(blend_images_custom_test, test_case_uint8) {
    size_t rgb_size = 480 * 640 * 3;
    size_t alpha_size = 480 * 640 * 1;
    size_t frame_size = 480 * 640 * 3;
    size_t out_size = 480 * 640 * 3;
    // inputs
    size_t tiling_data_size = sizeof(TilingDataBlendImages);

    uint8_t *rgb = (uint8_t*)AscendC::GmAlloc(rgb_size);
    uint8_t *alpha = (uint8_t*)AscendC::GmAlloc(alpha_size);
    uint8_t *frame = (uint8_t*)AscendC::GmAlloc(frame_size);
    uint8_t *out = (uint8_t*)AscendC::GmAlloc(out_size);
    
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 8;
    system("cp -r ../../../../../../../ops/objdetect/blend_images_custom/tests/ut/op_kernel/blend_images_custom_data ./");
    system("chmod -R 755 ./blend_images_custom_data/");
    system("cd ./blend_images_custom_data/ && rm -rf ./*bin");
    system("cd ./blend_images_custom_data/ && python3 gen_data.py");
    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/blend_images_custom_data/rgb.bin", rgb_size, rgb, rgb_size);
    ReadFile(path + "/blend_images_custom_data/alpha.bin", alpha_size, alpha, alpha_size);
    ReadFile(path + "/blend_images_custom_data/frame.bin", frame_size, frame, frame_size);
    ReadFile(path + "/blend_images_custom_data/out.bin", out_size, out, out_size);
    TilingDataBlendImages* tilingDatafromBin = reinterpret_cast<TilingDataBlendImages*>(tiling);
    tilingDatafromBin->totalAlphaLength = 480 * 640 * 1;
    
    ICPU_RUN_KF(blend_images_custom, numBlocks, rgb, alpha, frame, out, workspace, (uint8_t*)(tilingDatafromBin));
    
    AscendC::GmFree(rgb);
    AscendC::GmFree(alpha);
    AscendC::GmFree(frame);
    AscendC::GmFree(out);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(blend_images_custom_test, test_case_uint8_size_is_zero) {
    size_t rgb_size = 0;
    size_t alpha_size = 0;
    size_t frame_size = 0;
    size_t out_size = 0;
    // inputs
    size_t tiling_data_size = sizeof(TilingDataBlendImages);

    uint8_t *rgb = (uint8_t*)AscendC::GmAlloc(rgb_size);
    uint8_t *alpha = (uint8_t*)AscendC::GmAlloc(alpha_size);
    uint8_t *frame = (uint8_t*)AscendC::GmAlloc(frame_size);
    uint8_t *out = (uint8_t*)AscendC::GmAlloc(out_size);
    
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 8;
    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/blend_images_custom/blend_images_custom_data ./");
    system("chmod -R 755 ./blend_images_custom_data/");
    system("cd ./blend_images_custom_data/ && rm -rf ./*bin");
    system("cd ./blend_images_custom_data/ && python3 gen_data.py");
    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/blend_images_custom_data/rgb.bin", rgb_size, rgb, rgb_size);
    ReadFile(path + "/blend_images_custom_data/alpha.bin", alpha_size, alpha, alpha_size);
    ReadFile(path + "/blend_images_custom_data/frame.bin", frame_size, frame, frame_size);
    ReadFile(path + "/blend_images_custom_data/out.bin", out_size, out, out_size);
    TilingDataBlendImages* tilingDatafromBin = reinterpret_cast<TilingDataBlendImages*>(tiling);
    tilingDatafromBin->totalAlphaLength = 0;
    
    ICPU_RUN_KF(blend_images_custom, numBlocks, rgb, alpha, frame, out, workspace, (uint8_t*)(tilingDatafromBin));
    
    AscendC::GmFree(rgb);
    AscendC::GmFree(alpha);
    AscendC::GmFree(frame);
    AscendC::GmFree(out);
    AscendC::GmFree(tiling);
    free(path_);
}