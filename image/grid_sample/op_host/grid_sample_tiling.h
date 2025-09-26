/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grid_sample_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLE_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLE_H
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_key.h"
#include "tiling_base/tiling_util.h"

namespace optiling {

struct GridSampleCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

BEGIN_TILING_DATA_DEF(GridSampleTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNumVar);
TILING_DATA_FIELD_DEF(int64_t, inN);
TILING_DATA_FIELD_DEF(int64_t, inC);
TILING_DATA_FIELD_DEF(int64_t, inD);
TILING_DATA_FIELD_DEF(int64_t, inH);
TILING_DATA_FIELD_DEF(int64_t, inW);
TILING_DATA_FIELD_DEF(int64_t, outD);
TILING_DATA_FIELD_DEF(int64_t, outH);
TILING_DATA_FIELD_DEF(int64_t, outW);
TILING_DATA_FIELD_DEF(int64_t, interpolationMode);
TILING_DATA_FIELD_DEF(int64_t, paddingMode);
TILING_DATA_FIELD_DEF(int64_t, alignCorners);
TILING_DATA_FIELD_DEF(int64_t, channelLast);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, preCoreNum);
TILING_DATA_FIELD_DEF(int64_t, preNumPerCore);
TILING_DATA_FIELD_DEF(int64_t, postNumPerCore);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GridSample, GridSampleTilingData)

enum class GridSampleDtypeKey : int32_t { FLOAT16 = 1, FLOAT32 = 2, BFLOAT16 = 3 };

class GridSampleTiling : public Ops::Cv::OpTiling::TilingBaseClass {
public:
    explicit GridSampleTiling(gert::TilingContext *context) : Ops::Cv::OpTiling::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    ge::DataType xDtype{ge::DT_FLOAT};
    int64_t coreNumVar{0};
    int64_t inN{0};
    int64_t inD{0};
    int64_t inC{0};
    int64_t inH{0};
    int64_t inW{0};
    int64_t outD{0};
    int64_t outH{0};
    int64_t outW{0};
    int64_t interpolationMode{0};
    int64_t paddingMode{0};
    int64_t alignCorners{0};
    int64_t channelLast{0};
    int64_t schedulerMode{0};
    int64_t needCoreNum{0};
    int64_t dimValue{0};
    int64_t tempType{1};
    int64_t templateCNum{0};
    int64_t hwFactor{512};
    int64_t dimension{0};
    GridSampleTilingData tilingData;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLE_H