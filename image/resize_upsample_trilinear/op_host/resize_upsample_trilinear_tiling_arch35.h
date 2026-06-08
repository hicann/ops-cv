/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_upsample_trilinear_tiling_arch35.h
 * \brief resize_upsample_trilinear tiling for A5 architecture
 */
#ifndef RESIZE_UPSAMPLE_TRILINEAR_TILING_ARCH35_H_
#define RESIZE_UPSAMPLE_TRILINEAR_TILING_ARCH35_H_

#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

class ResizeUpsampleTrilinearArch35Tiling : public Ops::Cv::OpTiling::TilingBaseClass {
public:
    explicit ResizeUpsampleTrilinearArch35Tiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    int32_t coreNum_ = 0;
    bool platformInfoCached_ = false;
    int64_t inN_ = 0;
    int64_t inC_ = 0;
    int64_t outN_ = 0;
    int64_t outC_ = 0;
    int64_t batchCount_ = 0;
    int64_t inputD_ = 0;
    int64_t inputH_ = 0;
    int64_t inputW_ = 0;
    int64_t outputD_ = 0;
    int64_t outputH_ = 0;
    int64_t outputW_ = 0;
    float scaleD_ = 0.0f;
    float scaleH_ = 0.0f;
    float scaleW_ = 0.0f;
    int32_t alignCorners_ = 0;
    uint64_t totalElements_ = 0;
    uint32_t threadsPerBlock_ = 512;
    uint32_t elementsPerThread_ = 1;
    uint32_t blockCount_ = 1;
    uint32_t usedCoreNum_ = 1;
    uint32_t baseElementsPerBlock_ = 0;
    uint32_t tailElements_ = 0;
    uint64_t dtypeKey_ = 0;
    int32_t useInt32_ = 0;

    ge::graphStatus ValidateAndGetInputShape();
    ge::graphStatus ValidateAndGetOutputShape();
    ge::graphStatus ValidateShapeValues();
    ge::graphStatus ExtractAttrsAndComputeScales();
    ge::graphStatus DetermineDtypeKey();
    ge::graphStatus ValidateDimensionsAndComputeTotal();
    ge::graphStatus HandleZeroElements();
    ge::graphStatus ComputeThreadBlockConfig();
    ge::graphStatus ComputeFinalTilingConfig();
};

} // namespace optiling

#endif // RESIZE_UPSAMPLE_TRILINEAR_TILING_ARCH35_H_
