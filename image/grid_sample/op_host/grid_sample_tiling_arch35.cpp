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
 * \file grid_sample_tiling_arch35.cpp
 * \brief grid_sample_tiling_arch35 impl info
 */
#include "grid_sample_tiling_arch35.h"
#include "grid_sample_tiling.h"

namespace optiling {
static constexpr uint32_t VF_MAX_THREAD_NUM = 512;
static constexpr uint32_t SIMT_COMMON_TILING_KEY = 1000;
static const std::set<ge::DataType> supportDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

bool GridSampleArch35Tiling::IsCapable()
{
    return true;
}

ge::graphStatus GridSampleArch35Tiling::DoOpTiling()
{
    tilingData.set_coreNumVar(coreNumVar);
    tilingData.set_inN(inN);
    tilingData.set_inC(inC);
    tilingData.set_inH(inH);
    tilingData.set_inW(inW);
    tilingData.set_outH(outH);
    tilingData.set_outW(outW);
    tilingData.set_interpolationMode(interpolationMode);
    tilingData.set_paddingMode(paddingMode);
    tilingData.set_alignCorners(alignCorners);
    tilingData.set_channelLast(channelLast);

    // output format is [N, C, H, W]
    int64_t outputSize = inN * inC * outH * outW;
    needCoreNum = (outputSize + VF_MAX_THREAD_NUM - 1) / VF_MAX_THREAD_NUM;
    tilingData.set_blockNum(needCoreNum);
    needCoreNum = std::min(needCoreNum, coreNumVar);
    tilingData.set_needCoreNum(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GridSampleArch35Tiling::GetWorkspaceSize()
{
    workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GridSampleArch35Tiling::PostTiling()
{
    context_->SetBlockDim(tilingData.get_needCoreNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;

    gert::TilingData* rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        rawTilingData == nullptr, OP_LOGE(context_->GetNodeType(), "GetRawTilingData failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tilingData.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_, "actual tiling data size %zu > context tiling data size %zu", tilingData.GetDataSize(),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

uint64_t GridSampleArch35Tiling::GetTilingKey() const
{
    return SIMT_COMMON_TILING_KEY;
}

REGISTER_TILING_TEMPLATE("GridSample", GridSampleArch35Tiling, 2000);

} // namespace optiling