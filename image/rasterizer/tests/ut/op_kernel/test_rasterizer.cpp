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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "impl/dav_c220/kfc/kfc_log.h"
#include "../../../op_host/rasterizer_tiling.h"
#include "data_utils.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

extern "C" void rasterizer(uint8_t *v, uint8_t *f, uint8_t *d, uint8_t *findices, uint8_t *barycentric, uint8_t *workspace, uint8_t *tiling);

class rasterizer_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rasterizer_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rasterizer_test TearDown\n" << endl;
    }
};

TEST_F(rasterizer_test, test_case_test01)
{
    size_t vByteSize = 3 * 4 * sizeof(float);
    size_t fByteSize = 1 * 3 * sizeof(int32_t);
    size_t dByteSize = 10 * 10 * sizeof(float);
    size_t findicesByteSize = 10 * 10 * sizeof(int32_t);
    size_t baryByteSize = 10 * 10 * 3 * sizeof(float);

    gert::StorageShape v_shape = {{3, 4}, {3, 4}};
    gert::StorageShape f_shape = {{1, 3}, {1, 3}};
    gert::StorageShape d_shape = {{10, 10}, {10, 10}};
    gert::StorageShape findices_shape = {{10, 10}, {10, 10}};
    gert::StorageShape bary_shape = {{10, 10, 3}, {10, 10, 3}};
    RasterizerCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("Rasterizer",
                                                {{v_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {f_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {d_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {{findices_shape, ge::DT_INT32, ge::FORMAT_ND},
                                                {bary_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                {gert::TilingContextPara::OpAttr("width", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
                                                gert::TilingContextPara::OpAttr("height", Ops::Cv::AnyValue::CreateFrom<int64_t>(10)),
                                                gert::TilingContextPara::OpAttr("occlusion_truncation", Ops::Cv::AnyValue::CreateFrom<float>(0.0)),
                                                gert::TilingContextPara::OpAttr("use_depth_prior", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))},
                                                &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint8_t *v = (uint8_t *)AscendC::GmAlloc(vByteSize);
    uint8_t *f = (uint8_t *)AscendC::GmAlloc(fByteSize);
    uint8_t *d = (uint8_t *)AscendC::GmAlloc(dByteSize);
    uint8_t *findices = (uint8_t *)AscendC::GmAlloc(findicesByteSize);
    uint8_t *bary = (uint8_t *)AscendC::GmAlloc(baryByteSize);

    uint32_t numBlocks = 16;

    char *path_ = get_current_dir_name();
    string path(path_);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(1);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(rasterizer, numBlocks, v, f, d, findices, bary, workspace, tiling);

    AscendC::GmFree(v);
    AscendC::GmFree(f);
    AscendC::GmFree(d);
    AscendC::GmFree(findices);
    AscendC::GmFree(bary);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}