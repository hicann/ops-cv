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
 * \file resize_bilinear_v2_grad_tiling_arch35.h
 * \brief resize_bilinear_v2_grad_tiling_arch35
 */
#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_RESIZE_BILINEAR_V2_GRAD_TILING_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_RESIZE_BILINEAR_V2_GRAD_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ResizeBilinearV2GradTilingData)
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, alignCorners);
TILING_DATA_FIELD_DEF(int64_t, halfPixelCenters);
TILING_DATA_FIELD_DEF(int64_t, lenN);
TILING_DATA_FIELD_DEF(int64_t, lenC);
TILING_DATA_FIELD_DEF(int64_t, lenSrcH);
TILING_DATA_FIELD_DEF(int64_t, lenSrcW);
TILING_DATA_FIELD_DEF(int64_t, lenDesH);
TILING_DATA_FIELD_DEF(int64_t, lenDesW);
TILING_DATA_FIELD_DEF(int64_t, nFactor);
TILING_DATA_FIELD_DEF(int64_t, hFactor);
TILING_DATA_FIELD_DEF(int64_t, wFactor);
TILING_DATA_FIELD_DEF(int64_t, cFactor);
TILING_DATA_FIELD_DEF(int64_t, hwFactor);
TILING_DATA_FIELD_DEF(int64_t, ubNFactor);
TILING_DATA_FIELD_DEF(int64_t, ubHFactor);
TILING_DATA_FIELD_DEF(int64_t, ubWFactor);
TILING_DATA_FIELD_DEF(int64_t, ubCFactor);
TILING_DATA_FIELD_DEF(int64_t, ubHWFactor);
TILING_DATA_FIELD_DEF(float, scaleH);
TILING_DATA_FIELD_DEF(float, scaleW);
TILING_DATA_FIELD_DEF(float, inverseScaleH);               // for simt determine
TILING_DATA_FIELD_DEF(float, inverseScaleW);               // for simt determine
TILING_DATA_FIELD_DEF(int64_t, initYRealCoreNum);          // for init y
TILING_DATA_FIELD_DEF(int64_t, initYSplitBlockFactor);     // for init y
TILING_DATA_FIELD_DEF(int64_t, initYSplitBlockTailFactor); // for init y
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);               // for simt
TILING_DATA_FIELD_DEF(int64_t, splitBlockFactor);          // for simt
TILING_DATA_FIELD_DEF(int64_t, splitBlockTailFactor);      // for simt
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ResizeBilinearV2Grad, ResizeBilinearV2GradTilingData)

struct ResizeBilinearV2GradCompileInfo {
    int32_t coreNum;
    int32_t ubSize;
};

class ResizeBilinearV2GradTilingAscendC {
public:
    explicit ResizeBilinearV2GradTilingAscendC(gert::TilingContext* context)
        : context_(context), nodeName_(context->GetNodeName()){};

    ge::graphStatus Init(const ResizeBilinearV2GradCompileInfo* compileInfo);
    ge::graphStatus DoTiling();

private:
    inline int64_t Min(int64_t x, int64_t y);
    ge::graphStatus GetPlatformInfo(const ResizeBilinearV2GradCompileInfo* compileInfo);
    ge::graphStatus GetTensorInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus CheckDtypeValid();
    ge::graphStatus CheckFormatValid();
    ge::graphStatus CheckShapeValid();

    void SetScales(bool isDetermine);
    void SetFactors();

    bool IsMatchAllCopy();
    bool IsMatchPointCopy();
    bool IsMatchSimtDetermine();
    bool IsMatchCParallel();
    void SetSimtTilingKey(bool isDetermine);
    void SetTilingKey();
    void DoTilingInitY();
    void DoTilingSimtNotDetermine();
    void DoTilingSimtDetermine();
    void DoTilingAllCopy();
    int64_t FindBest2DTiling(int64_t lenM, int64_t lenN);
    void DoTilingPointCopy();
    void DoTilingCParallel();
    void DoTilingStrategy();

    void FillTilingData();
    void PrintTilingData();

private:
    int32_t gradsDtypeSize_ = 0;
    int32_t yDtypeSize_ = 0;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t ubBlockNum_ = 0;
    int64_t isDetermine_ = 0;
    int64_t tilingKey_ = 0;
    ge::DataType gradsDtype_ = ge::DT_MAX;
    ge::DataType originalImageDtype_ = ge::DT_MAX;
    ge::DataType yDtype_ = ge::DT_MAX;
    ge::Format gradsFormat_ = ge::FORMAT_MAX;
    ge::Format originalImageFormat_ = ge::FORMAT_MAX;
    ge::Format yFormat_ = ge::FORMAT_MAX;
    gert::Shape gradsShape_;
    gert::Shape originalImageShape_;
    gert::Shape yShape_;

    int64_t alignCorners_ = 0;
    int64_t halfPixelCenters_ = 0;
    float originalScaleW_ = 0.0f;
    float originalScaleH_ = 0.0f;

    int64_t lenC_ = 0;
    int64_t lenN_ = 0;
    int64_t lenSrcH_ = 0;
    int64_t lenSrcW_ = 0;
    int64_t lenDesH_ = 0;
    int64_t lenDesW_ = 0;

    int64_t nFactor_ = 0;
    int64_t hFactor_ = 0;
    int64_t wFactor_ = 0;
    int64_t cFactor_ = 0;
    int64_t hwFactor_ = 0;
    int64_t ubNFactor_ = 0;
    int64_t ubHFactor_ = 0;
    int64_t ubWFactor_ = 0;
    int64_t ubCFactor_ = 0;
    int64_t ubHWFactor_ = 0;

    float scaleW_ = 0.0f;
    float scaleH_ = 0.0f;
    float inverseScaleW_ = 0.0f;
    float inverseScaleH_ = 0.0f;

    int64_t lenCAlign_ = 0;
    bool isAlign_ = false;
    bool isNeedInitY_ = false;
    int64_t initYRealCoreNum_ = 0;
    int64_t initYSplitBlockFactor_ = 0;
    int64_t initYSplitBlockTailFactor_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t splitBlockFactor_ = 0;
    int64_t splitBlockTailFactor_ = 0;

    gert::TilingContext* context_ = nullptr;
    std::string nodeName_ = "ResizeBilinearV2Grad";
    ResizeBilinearV2GradTilingData tilingData_;
};

} // namespace optiling

#endif // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_RESIZE_BILINEAR_V2_GRAD_TILING_H_