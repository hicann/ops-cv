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
#include "test_background_replace.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;


extern "C" __global__ __aicore__ void background_replace(GM_ADDR bkg, GM_ADDR src,GM_ADDR mask, GM_ADDR out,
                                                         GM_ADDR workSpace, GM_ADDR tiling);

class background_replace_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "background_replace_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "background_replace_test TearDown\n" << endl;
    }
};

TEST_F(background_replace_test, test_case_float16_c1) {
    size_t bkg_size = 480*640*2;
    size_t src_size = 480*640*2;
    size_t mask_size = 480*640*2;
    size_t out_size = 480*640*2;
    // inputs
    size_t tiling_data_size = sizeof(BackgroundReplaceTilingData);

    uint8_t *bkg = (uint8_t*)AscendC::GmAlloc(bkg_size);
    uint8_t *src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t *mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t *golden = (uint8_t*)AscendC::GmAlloc(out_size);
    uint8_t *out = (uint8_t*)AscendC::GmAlloc(out_size);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;
    system("cp -r ../../../../../../../ops/objdetect/background_replace/tests/ut/op_kernel/background_replace_data ./");
    system("chmod -R 755 ./background_replace_data/");
    system("cd ./background_replace_data/ && rm -rf ./*bin");
    system("cd ./background_replace_data/ && python3 gen_data.py float16 1");
    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/background_replace_data/bkg.bin", bkg_size, bkg, bkg_size);
    ReadFile(path + "/background_replace_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/background_replace_data/mask.bin", mask_size, mask, mask_size);
    ReadFile(path + "/background_replace_data/golden.bin", out_size, golden, out_size);
    BackgroundReplaceTilingData* tilingDatafromBin = reinterpret_cast<BackgroundReplaceTilingData*>(tiling);
    tilingDatafromBin->size = 480*640;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(background_replace, blockDim, bkg, src, mask, out, workspace, (uint8_t*)(tilingDatafromBin));
    
    AscendC::GmFree(bkg);
    AscendC::GmFree(src);
    AscendC::GmFree(mask);
    AscendC::GmFree(out);
    AscendC::GmFree(golden);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(background_replace_test, test_case_u8_c1) {
    size_t bkg_size = 480*640;
    size_t src_size = 480*640;
    size_t mask_size = 480*640*2;
    size_t out_size = 480*640;
    // inputs
    size_t tiling_data_size = sizeof(BackgroundReplaceTilingData);

    uint8_t *bkg = (uint8_t*)AscendC::GmAlloc(bkg_size);
    uint8_t *src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t *mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t *golden = (uint8_t*)AscendC::GmAlloc(out_size);
    uint8_t *out = (uint8_t*)AscendC::GmAlloc(out_size);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;
    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/background_replace/background_replace_data ./");
    system("chmod -R 755 ./background_replace_data/");
    system("cd ./background_replace_data/ && rm -rf ./*bin");
    system("cd ./background_replace_data/ && python3 gen_data.py uint8 1");
    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/background_replace_data/bkg.bin", bkg_size, bkg, bkg_size);
    ReadFile(path + "/background_replace_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/background_replace_data/mask.bin", mask_size, mask, mask_size);
    ReadFile(path + "/background_replace_data/golden.bin", out_size, golden, out_size);
    BackgroundReplaceTilingData* tilingDatafromBin = reinterpret_cast<BackgroundReplaceTilingData*>(tiling);
    tilingDatafromBin->size = 480*640;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(background_replace, blockDim, bkg, src, mask, out, workspace, (uint8_t*)(tilingDatafromBin));
    
    AscendC::GmFree(bkg);
    AscendC::GmFree(src);
    AscendC::GmFree(mask);
    AscendC::GmFree(out);
    AscendC::GmFree(golden);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}



TEST_F(background_replace_test, test_case_float16_c3) {
    size_t bkg_size = 480*640*6;
    size_t src_size = 480*640*6;
    size_t mask_size = 480*640*2;
    size_t out_size = 480*640*6;
    // inputs
    size_t tiling_data_size = sizeof(BackgroundReplaceTilingData);

    uint8_t *bkg = (uint8_t*)AscendC::GmAlloc(bkg_size);
    uint8_t *src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t *mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t *golden = (uint8_t*)AscendC::GmAlloc(out_size);
    uint8_t *out = (uint8_t*)AscendC::GmAlloc(out_size);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;
    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/background_replace/background_replace_data ./");
    system("chmod -R 755 ./background_replace_data/");
    system("cd ./background_replace_data/ && rm -rf ./*bin");
    system("cd ./background_replace_data/ && python3 gen_data.py float16 3");
    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/background_replace_data/bkg.bin", bkg_size, bkg, bkg_size);
    ReadFile(path + "/background_replace_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/background_replace_data/mask.bin", mask_size, mask, mask_size);
    ReadFile(path + "/background_replace_data/golden.bin", out_size, golden, out_size);
    BackgroundReplaceTilingData* tilingDatafromBin = reinterpret_cast<BackgroundReplaceTilingData*>(tiling);
    tilingDatafromBin->size = 480*640;

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(background_replace, blockDim, bkg, src, mask, out, workspace, (uint8_t*)(tilingDatafromBin));
    
    AscendC::GmFree(bkg);
    AscendC::GmFree(src);
    AscendC::GmFree(mask);
    AscendC::GmFree(out);
    AscendC::GmFree(golden);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(background_replace_test, test_case_u8_c3) {
    size_t bkg_size = 480*640*3;
    size_t src_size = 480*640*3;
    size_t mask_size = 480*640*2;
    size_t out_size = 480*640*3;
    // inputs
    size_t tiling_data_size = sizeof(BackgroundReplaceTilingData);

    uint8_t *bkg = (uint8_t*)AscendC::GmAlloc(bkg_size);
    uint8_t *src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t *mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t *golden = (uint8_t*)AscendC::GmAlloc(out_size);
    uint8_t *out = (uint8_t*)AscendC::GmAlloc(out_size);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 8;
    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/background_replace/background_replace_data ./");
    system("chmod -R 755 ./background_replace_data/");
    system("cd ./background_replace_data/ && rm -rf ./*bin");
    system("cd ./background_replace_data/ && python3 gen_data.py uint8 3");
    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/background_replace_data/bkg.bin", bkg_size, bkg, bkg_size);
    ReadFile(path + "/background_replace_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/background_replace_data/mask.bin", mask_size, mask, mask_size);
    ReadFile(path + "/background_replace_data/golden.bin", out_size, golden, out_size);
    BackgroundReplaceTilingData* tilingDatafromBin = reinterpret_cast<BackgroundReplaceTilingData*>(tiling);
    tilingDatafromBin->size = 480*640;

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(background_replace, blockDim, bkg, src, mask, out, workspace, (uint8_t*)(tilingDatafromBin));
    
    AscendC::GmFree(bkg);
    AscendC::GmFree(src);
    AscendC::GmFree(mask);
    AscendC::GmFree(out);
    AscendC::GmFree(golden);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}