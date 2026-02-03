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
#include "test_mrgba_custom.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ void mrgba_custom(GM_ADDR rgb, GM_ADDR alpha, GM_ADDR dst, GM_ADDR workspace, GM_ADDR tiling);
class mrgba_custom_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "mrgba_custom_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "mrgba_custom_test TearDown\n" << endl;
    }
};

TEST_F(mrgba_custom_test, test_case_uint8) {
    // inputs
    size_t tiling_data_size = sizeof(MrgbaCustomTilingData);
    size_t rgb_size = 480*640*3;
    size_t alpha_size = 480*640*1;
    uint32_t numBlocks = 1;

    uint8_t *rgb = (uint8_t*)AscendC::GmAlloc(rgb_size);
    uint8_t *alpha = (uint8_t*)AscendC::GmAlloc(alpha_size);
    uint8_t *dst = (uint8_t*)AscendC::GmAlloc(rgb_size);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    system("cp -r ../../../../../../../ops/objdetect/mrgba_custom/tests/ut/op_kernel/mrgba_custom_data ./");
    system("chmod -R 755 ./mrgba_custom_data/");
    system("cd ./mrgba_custom_data/ && rm -rf ./*bin");
    system("cd ./mrgba_custom_data/ && python3 gen_data.py");
    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/mrgba_custom_data/rgb.bin", rgb_size, rgb, rgb_size);
    ReadFile(path + "/mrgba_custom_data/alpha.bin", alpha_size, alpha, alpha_size);
    MrgbaCustomTilingData* tilingDatafromBin = reinterpret_cast<MrgbaCustomTilingData*>(tiling);
    tilingDatafromBin->alphaLen = 480*640*1;

    ICPU_RUN_KF(mrgba_custom, numBlocks, rgb, alpha, dst, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(rgb);
    AscendC::GmFree(alpha);
    AscendC::GmFree(dst);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}