/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file resize_bicubic_v2_grad_tiling_arch35.h
 * \brief resize_bicubic_v2_grad_tiling_arch35
 */
#ifndef OPS_IMAGE_RESIZE_BICUBIC_V2_GRAD_OP_HOST_RESIZE_BICUBIC_V2_GRAD_TILING_ARCH35_H_
#define OPS_IMAGE_RESIZE_BICUBIC_V2_GRAD_OP_HOST_RESIZE_BICUBIC_V2_GRAD_TILING_ARCH35_H_

#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_base.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

struct ResizeBicubicV2GradCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
    int64_t ubBlockSize{0};
    int64_t isDetermine{0};
};

struct ResizeBicubicV2GradInputInfo {
    gert::Shape gradsShape;
    ge::DataType gradsDtype{ge::DT_MAX};
    ge::Format gradsFormat{ge::FORMAT_MAX};
    gert::Shape originalImageShape;
    ge::DataType originalImageDtype{ge::DT_MAX};
    ge::Format originalImageFormat{ge::FORMAT_MAX};
    gert::Shape yShape;
    ge::DataType yDtype{ge::DT_MAX};
    ge::Format yFormat{ge::FORMAT_MAX};
    int64_t lenN{0};
    int64_t lenC{0};
    int64_t lenSrcH{0};
    int64_t lenSrcW{0};
    int64_t lenDstH{0};
    int64_t lenDstW{0};
    int64_t format{0};
    int64_t alignCorners{0};
    float oriScaleH{0.0f};
    float oriScaleW{0.0f};
};

struct ResizeBicubicV2GradCalculateInfo {
    int64_t ubBlockNum{0};
    int64_t gradsDtypeSize{0};
    int64_t yDtypeSize{0};
    int64_t gradsShapeSize{0};
    int64_t yShapeSize{0};
    int64_t initYUseCoreNum{0};
    int64_t initYCoreFactor{0};
    int64_t initYCoreTailFactor{0};
    int64_t useCoreNum{0};
    int64_t coreFactor{0};
    int64_t coreTailFactor{0};
    int64_t ubFactor{0};
    int64_t isMatchDetermine{0};
    float scaleH{0.0f};
    float scaleW{0.0f};
    float inverseScaleH{0.0f};
    float inverseScaleW{0.0f};
};

BEGIN_TILING_DATA_DEF(ResizeBicubicV2GradSimtTilingData)
TILING_DATA_FIELD_DEF(int64_t, lenC);
TILING_DATA_FIELD_DEF(int64_t, lenSrcH);
TILING_DATA_FIELD_DEF(int64_t, lenSrcW);
TILING_DATA_FIELD_DEF(int64_t, lenDstH);
TILING_DATA_FIELD_DEF(int64_t, lenDstW);
TILING_DATA_FIELD_DEF(int64_t, format);
TILING_DATA_FIELD_DEF(int64_t, alignCorners);
TILING_DATA_FIELD_DEF(int64_t, initYUseCoreNum);
TILING_DATA_FIELD_DEF(int64_t, initYCoreFactor);
TILING_DATA_FIELD_DEF(int64_t, initYCoreTailFactor);
TILING_DATA_FIELD_DEF(int64_t, useCoreNum);
TILING_DATA_FIELD_DEF(int64_t, coreFactor);
TILING_DATA_FIELD_DEF(int64_t, coreTailFactor);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, scaleW);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ResizeBicubicV2GradSimtDetermineTilingData)
TILING_DATA_FIELD_DEF(int64_t, lenC);
TILING_DATA_FIELD_DEF(int64_t, lenSrcH);
TILING_DATA_FIELD_DEF(int64_t, lenSrcW);
TILING_DATA_FIELD_DEF(int64_t, lenDstH);
TILING_DATA_FIELD_DEF(int64_t, lenDstW);
TILING_DATA_FIELD_DEF(int64_t, format);
TILING_DATA_FIELD_DEF(int64_t, alignCorners);
TILING_DATA_FIELD_DEF(int64_t, useCoreNum);
TILING_DATA_FIELD_DEF(int64_t, coreFactor);
TILING_DATA_FIELD_DEF(int64_t, coreTailFactor);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, inverseScaleH);
TILING_DATA_FIELD_DEF(float, inverseScaleW);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(ResizeBicubicV2GradAllCopyTilingData)
TILING_DATA_FIELD_DEF(int64_t, useCoreNum);
TILING_DATA_FIELD_DEF(int64_t, coreFactor);
TILING_DATA_FIELD_DEF(int64_t, coreTailFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResizeBicubicV2Grad, ResizeBicubicV2GradSimtTilingData)
REGISTER_TILING_DATA_CLASS(ResizeBicubicV2Grad_20000, ResizeBicubicV2GradSimtDetermineTilingData)
REGISTER_TILING_DATA_CLASS(ResizeBicubicV2Grad_20001, ResizeBicubicV2GradSimtDetermineTilingData)
REGISTER_TILING_DATA_CLASS(ResizeBicubicV2Grad_30000, ResizeBicubicV2GradAllCopyTilingData)

class ResizeBicubicV2GradBaseTiling : public Ops::Cv::OpTiling::TilingBaseClass {
public:
    explicit ResizeBicubicV2GradBaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~ResizeBicubicV2GradBaseTiling() override
    {}

    ResizeBicubicV2GradCompileInfo compileInfo_;
    ResizeBicubicV2GradInputInfo inputInfo_;
    ResizeBicubicV2GradCalculateInfo calcInfo_;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus GetTensorInfo();
    ge::graphStatus CheckDtypeValid();
    ge::graphStatus CheckFormatValid();
    ge::graphStatus CheckShapeValid();
    ge::graphStatus GetAttrInfo();
    void SetScales();
    bool IsUseIdx32() const;
};

class ResizeBicubicV2GradSimtTiling : public ResizeBicubicV2GradBaseTiling {
public:
    explicit ResizeBicubicV2GradSimtTiling(gert::TilingContext* context) : ResizeBicubicV2GradBaseTiling(context)
    {}
    ~ResizeBicubicV2GradSimtTiling() override
    {}

    ResizeBicubicV2GradSimtTilingData tilingData_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    void SetTilingData();
    void PrintTilingData();
};

class ResizeBicubicV2GradSimtDetermineTiling : public ResizeBicubicV2GradBaseTiling {
public:
    explicit ResizeBicubicV2GradSimtDetermineTiling(gert::TilingContext* context)
        : ResizeBicubicV2GradBaseTiling(context)
    {}
    ~ResizeBicubicV2GradSimtDetermineTiling() override
    {}

    ResizeBicubicV2GradSimtDetermineTilingData tilingData_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    void SetTilingData();
    void PrintTilingData();
};

class ResizeBicubicV2GradAllCopyTiling : public ResizeBicubicV2GradBaseTiling {
public:
    explicit ResizeBicubicV2GradAllCopyTiling(gert::TilingContext* context) : ResizeBicubicV2GradBaseTiling(context)
    {}
    ~ResizeBicubicV2GradAllCopyTiling() override
    {}

    ResizeBicubicV2GradAllCopyTilingData tilingData_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    void SetTilingData();
    void PrintTilingData();
};

} // namespace optiling

#endif // OPS_IMAGE_RESIZE_BICUBIC_V2_GRAD_OP_HOST_RESIZE_BICUBIC_V2_GRAD_TILING_ARCH35_H_
