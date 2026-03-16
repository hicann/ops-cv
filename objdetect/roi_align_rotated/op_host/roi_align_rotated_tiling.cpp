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
 * \file roi_align_rotated_tiling.cc
 * \brief
 */
#include "roi_align_rotated_tiling.h"
#include <cmath>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "util/math_util.h"
#include "log/log.h"

using namespace ge;
using namespace std;

namespace optiling
{
    const uint32_t INPUT_INDEX = 0;
    const uint32_t ROIS_INDEX = 1;
    const uint32_t OUTPUT_INDEX = 0;
    const uint32_t ROIS_NUM_INDEX = 1;

    const uint32_t BS_INDEX = 0;
    const uint32_t H_INDEX = 1;
    const uint32_t W_INDEX = 2;
    const uint32_t CHANNEL_INDEX = 3;

    const uint32_t PH_INDEX = 0;
    const uint32_t PW_INDEX = 1;
    const uint32_t SPATIAL_INDEX = 2;
    const uint32_t SAMPLING_INDEX = 3;
    const uint32_t ALIGNED_INDEX = 4;
    const uint32_t CLOCKWISE_INDEX = 5;

    const uint32_t ALIGN_VALUE = 8;
    const uint32_t TILING_KEY = 1;
    const uint32_t TILE_NUM = 8;

    static ge::graphStatus GetRoiAlignRotatedPlatformInfo(void *platform,
        uint32_t &BLOCK_DIM, uint64_t &ub_total_size)
    {
        if (platform == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto platform_info = platform_ascendc::PlatformAscendC(static_cast<fe::PlatFormInfos*>(platform));
        platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_total_size);
        BLOCK_DIM = platform_info.GetCoreNumAiv();
        if (BLOCK_DIM == 0) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    static uint32_t AlignRoisNum(uint32_t rois_num)
    {
        if (static_cast<uint32_t>(rois_num % ALIGN_VALUE) == 0) {
            return rois_num;
        }
        return (static_cast<uint32_t>(rois_num / ALIGN_VALUE) + 1) * ALIGN_VALUE;
    }

    static ge::graphStatus ParseTilingInputs(gert::TilingContext *context,
        uint32_t &batch_size, uint32_t &input_h, uint32_t &input_w, uint32_t &channels, uint32_t &channels_aligned,
        float &spatial_scale, int32_t &sampling_ratio, int32_t &pooled_height, int32_t &pooled_width,
        bool &aligned, bool &clockwise, uint32_t &rois_num)
    {
        auto inputTensorPtr = context->GetInputTensor(INPUT_INDEX);
        auto RoisTensorPtr = context->GetInputTensor(ROIS_INDEX);
        if (inputTensorPtr == nullptr || RoisTensorPtr == nullptr) {
            return ge::GRAPH_FAILED;
        }
        batch_size = inputTensorPtr->GetStorageShape().GetDim(BS_INDEX);
        input_h = inputTensorPtr->GetStorageShape().GetDim(H_INDEX);
        input_w = inputTensorPtr->GetStorageShape().GetDim(W_INDEX);
        channels = inputTensorPtr->GetStorageShape().GetDim(CHANNEL_INDEX);
        channels_aligned = (static_cast<uint32_t>(channels % ALIGN_VALUE) == 0)
            ? channels
            : (static_cast<uint32_t>(channels / ALIGN_VALUE) + 1) * ALIGN_VALUE;

        auto attrsPtr = context->GetAttrs();
        if (attrsPtr == nullptr) {
            return ge::GRAPH_FAILED;
        }
        spatial_scale = *(attrsPtr->GetAttrPointer<float>(SPATIAL_INDEX));
        sampling_ratio = *(attrsPtr->GetAttrPointer<int32_t>(SAMPLING_INDEX));
        pooled_height = *(attrsPtr->GetAttrPointer<int32_t>(PH_INDEX));
        pooled_width = *(attrsPtr->GetAttrPointer<int32_t>(PW_INDEX));
        aligned = *(attrsPtr->GetAttrPointer<bool>(ALIGNED_INDEX));
        clockwise = *(attrsPtr->GetAttrPointer<bool>(CLOCKWISE_INDEX));

        rois_num = RoisTensorPtr->GetStorageShape().GetDim(ROIS_NUM_INDEX);
        if (rois_num == 0) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus ComputeRoiBlockPartition(uint32_t rois_num, uint32_t &BLOCK_DIM,
        uint32_t &rois_num_aligned, uint32_t &tail_num, uint32_t &rois_num_per_Score, uint32_t &rois_num_per_Lcore,
        uint32_t &Score_num, uint32_t &Lcore_num)
    {
        if (BLOCK_DIM == 0) {
            return ge::GRAPH_FAILED;  // 避免 rois_num_aligned / BLOCK_DIM 除零
        }
        rois_num_aligned = AlignRoisNum(rois_num);
        tail_num = rois_num_aligned - rois_num;
        rois_num_per_Score = (rois_num_aligned / BLOCK_DIM / ALIGN_VALUE) * ALIGN_VALUE;
        rois_num_per_Lcore = rois_num_per_Score + ALIGN_VALUE;
        Score_num = (BLOCK_DIM * (ALIGN_VALUE + rois_num_per_Score) - rois_num_aligned) / ALIGN_VALUE;
        Lcore_num = BLOCK_DIM - Score_num;
        if (rois_num_per_Score == 0) {
            BLOCK_DIM = BLOCK_DIM - Score_num;
        }
        if (rois_num_per_Lcore == 0) {
            BLOCK_DIM = BLOCK_DIM - Lcore_num;
        }
        return ge::GRAPH_SUCCESS;
    }

    static void FillTilingAndFlushToContext(gert::TilingContext *context, uint32_t BLOCK_DIM, uint64_t ub_total_size,
        uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, uint32_t channels_aligned,
        uint32_t rois_num_aligned, uint32_t tail_num, float spatial_scale, int32_t sampling_ratio,
        int32_t pooled_height, int32_t pooled_width, bool aligned, bool clockwise,
        uint32_t rois_num_per_Lcore, uint32_t rois_num_per_Score, uint32_t Lcore_num, uint32_t Score_num,
        uint32_t input_buffer_size)
    {
        RoiAlignRotatedTilingData tiling;
        tiling.set_numBlocks(BLOCK_DIM);
        tiling.set_ub_total_size(ub_total_size);
        tiling.set_tileNum(TILE_NUM);
        tiling.set_batch_size(batch_size);
        tiling.set_channels(channels);
        tiling.set_channels_aligned(channels_aligned);
        tiling.set_input_h(input_h);
        tiling.set_input_w(input_w);
        tiling.set_rois_num_aligned(rois_num_aligned);
        tiling.set_tail_num(tail_num);
        tiling.set_spatial_scale(spatial_scale);
        tiling.set_sampling_ratio(sampling_ratio);
        tiling.set_pooled_height(pooled_height);
        tiling.set_pooled_width(pooled_width);
        tiling.set_aligned(aligned);
        tiling.set_clockwise(clockwise);
        tiling.set_rois_num_per_Lcore(rois_num_per_Lcore);
        tiling.set_rois_num_per_Score(rois_num_per_Score);
        tiling.set_Lcore_num(Lcore_num);
        tiling.set_Score_num(Score_num);
        tiling.set_input_buffer_size(input_buffer_size);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetBlockDim(BLOCK_DIM);
        context->SetTilingKey(TILING_KEY);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = static_cast<size_t>(0);
    }

    static ge::graphStatus TilingPrepare4RoiAlignRotated(gert::TilingParseContext *context)
    {
        uint32_t BLOCK_DIM = 0;
        uint64_t ub_total_size = 0;
        ge::graphStatus ret = GetRoiAlignRotatedPlatformInfo(context->GetPlatformInfo(), BLOCK_DIM, ub_total_size);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }

        auto compileInfo = context->GetCompiledInfo<RoiAlignRotatedCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        compileInfo->totalCoreNum = BLOCK_DIM;
        compileInfo->ubSizePlatForm = static_cast<int64_t>(ub_total_size);

        return ge::GRAPH_SUCCESS;
    }
    static ge::graphStatus TilingForRoiAlignRotated(gert::TilingContext *context)
    {
        uint32_t batch_size = 0, input_h = 0, input_w = 0, channels = 0, channels_aligned = 0, rois_num = 0;
        float spatial_scale = 0.0f;
        int32_t sampling_ratio = 0, pooled_height = 0, pooled_width = 0;
        bool aligned = false, clockwise = false;
        if (ParseTilingInputs(context, batch_size, input_h, input_w, channels, channels_aligned,
                spatial_scale, sampling_ratio, pooled_height, pooled_width, aligned, clockwise, rois_num)
            != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        uint32_t BLOCK_DIM = 0;
        uint64_t ub_total_size = 0;
        if (GetRoiAlignRotatedPlatformInfo(context->GetPlatformInfo(), BLOCK_DIM, ub_total_size) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        uint32_t rois_num_aligned = 0, tail_num = 0, rois_num_per_Score = 0, rois_num_per_Lcore = 0;
        uint32_t Score_num = 0, Lcore_num = 0;
        if (ComputeRoiBlockPartition(rois_num, BLOCK_DIM, rois_num_aligned, tail_num,
                rois_num_per_Score, rois_num_per_Lcore, Score_num, Lcore_num) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        uint32_t input_buffer_size =
            static_cast<uint32_t>(ceil(static_cast<float>(channels_aligned) / ALIGN_VALUE)) * ALIGN_VALUE * sizeof(float);

        FillTilingAndFlushToContext(context, BLOCK_DIM, ub_total_size, batch_size, input_h, input_w,
            channels, channels_aligned, rois_num_aligned, tail_num, spatial_scale, sampling_ratio,
            pooled_height, pooled_width, aligned, clockwise, rois_num_per_Lcore, rois_num_per_Score,
            Lcore_num, Score_num, input_buffer_size);
        return ge::GRAPH_SUCCESS;
    }

    IMPL_OP_OPTILING(RoiAlignRotated)
        .Tiling(TilingForRoiAlignRotated)
        .TilingParse<RoiAlignRotatedCompileInfo>(TilingPrepare4RoiAlignRotated);
}