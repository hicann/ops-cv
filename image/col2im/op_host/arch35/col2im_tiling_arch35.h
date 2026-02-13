/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file col2im_tiling_arch35.h
 * \brief
 */

#ifndef COL2IM_TILING_ARCH35_H
#define COL2IM_TILING_ARCH35_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/col2im_tiling_data.h"

namespace optiling {

struct Col2imCompileInfo {};

class Col2imTiling {
public:
    explicit Col2imTiling(gert::TilingContext *context) : context_(context){};
    ge::graphStatus DoTiling();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus InitTilingData();
    ge::graphStatus GetInputTensorInfo();
    ge::graphStatus SetAttrParams();
    uint64_t GetTilingKey();
    bool GetDataTypeKey(ge::DataType dataType);

    void SetTilingKey();
    void SetTilingData();

private:
    Col2imRegBaseTilingData* tilingData_{nullptr};
    gert::TilingContext *context_ = nullptr;

    int64_t usrWorkspaceSize_ = 1;
    uint64_t dataTypeTilingKey_ = 0;
    uint64_t tilingKey_{ 0 };
    int64_t useCoreNum_ = 1;
    ge::DataType gardInDType_ = ge::DT_UNDEFINED;

    int64_t totalCoreNum_ = 1;
    int64_t totalLength_ = 1;
    int64_t outputSizeH_ = 1;
    int64_t outputSizeW_ = 1;
    int64_t kernelSizeH_ = 1;
    int64_t kernelSizeW_ = 1;
    int64_t dilationH_ = 1;
    int64_t dilationW_ = 1;
    int64_t paddingH_ = 1;
    int64_t paddingW_ = 1;
    int64_t strideH_ = 1;
    int64_t strideW_ = 1;
    int64_t colH_ = 1;
    int64_t colW_ = 1;
};
} // namespace optiling
#endif
