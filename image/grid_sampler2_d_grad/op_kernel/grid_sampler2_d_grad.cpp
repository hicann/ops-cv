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
 * \file grid_sampler2_d_grad.cpp
 * \brief
 */

#if __CCE_AICORE__ == 220
#include "grid_sampler2_d_grad.h"
#include "grid_sampler2_d_grad_fp16.h"
#include "grid_sampler2_d_grad_cast.h"
#include "grid_sampler2_d_grad_bicubic.h"
#include "grid_sampler2_d_grad_bicubic_fp16.h"
#else
#include "grid_sampler2_d_grad.h"
#include "grid_sampler2_d_grad_bicubic.h"
#include "arch35/grid_sampler2_d_grad_simt.h"
#include "arch35/grid_sampler2_d_grad_simt_det.h"
using namespace GridSampler2DSimtA5;
using namespace GridSampler2DSimtA5Det;
#endif

__aicore__ inline void InitWorkspace(const GridSampler2DGradTilingData& tiling, GM_ADDR workSpace)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t computePNum = 0;
    uint32_t castOffset = 0;
    float initParam = 0.0f;
    uint32_t tailPNumCast = tiling.tailPNumCast;
    uint32_t pNumPerCoreCast = tiling.pNumPerCoreCast;
    GlobalTensor<float> indexCountGm;
    if (blockIdx < tailPNumCast) {
        computePNum = pNumPerCoreCast + 1;
        castOffset = blockIdx * computePNum;
    } else {
        computePNum = pNumPerCoreCast;
        castOffset = blockIdx * computePNum + tailPNumCast;
    }
    indexCountGm.SetGlobalBuffer((__gm__ float*)workSpace + castOffset);
    InitOutput(indexCountGm, computePNum, initParam);
}

// kernel function
extern "C" __global__ __aicore__ void grid_sampler2_d_grad(
    GM_ADDR grad, GM_ADDR x, GM_ADDR grid, GM_ADDR dx, GM_ADDR dgrid, GM_ADDR workspace, GM_ADDR tiling)
{
    #if __CCE_AICORE__ != 220
 	     if ASCEND_IS_AIC {
 	         return;
 	     }
 	#endif
    
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR gmTensor[6] = {grad, x, grid, dx, dgrid, workspace};
#if __CCE_AICORE__ == 220
    if (TILING_KEY_IS(1)) {
        GridSampler2DGrad<float, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitBilinearLocalTensor();
        op.Process();
    }
    if (TILING_KEY_IS(2)) {
        GridSampler2DGrad<float, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitNearestLocalTensor();
        op.Process();
    }
    if (TILING_KEY_IS(3)) {
        InitWorkspace(tilingData, workspace);
        SyncAll();
        GridSampler2DGradFP16<float, half, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitBilinearLocalTensor();
        op.Process();
        SyncAll();
        pipe.Destroy();
        TPipe tpipe;
        GridSampler2DGradCast<half, GridSampler2DGradTilingData> castOp;
        castOp.Init(tilingData, gmTensor, &tpipe);
        castOp.Process();
    }
    if (TILING_KEY_IS(4)) {
        InitWorkspace(tilingData, workspace);
        SyncAll();
        GridSampler2DGradFP16<float, half, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitNearestLocalTensor();
        op.Process();
        SyncAll();
        pipe.Destroy();
        TPipe tpipe;
        GridSampler2DGradCast<half, GridSampler2DGradTilingData> castOp;
        castOp.Init(tilingData, gmTensor, &tpipe);
        castOp.Process();
    }
    if (TILING_KEY_IS(5)) {
        InitWorkspace(tilingData, workspace);
        SyncAll();
        GridSampler2DGradFP16<float, bfloat16_t, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitBilinearLocalTensor();
        op.Process();
        SyncAll();
        pipe.Destroy();
        TPipe tpipe;
        GridSampler2DGradCast<bfloat16_t, GridSampler2DGradTilingData> castOp;
        castOp.Init(tilingData, gmTensor, &tpipe);
        castOp.Process();
    }
    if (TILING_KEY_IS(6)) {
        InitWorkspace(tilingData, workspace);
        SyncAll();
        GridSampler2DGradFP16<float, bfloat16_t, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitNearestLocalTensor();
        op.Process();
        SyncAll();
        pipe.Destroy();
        TPipe tpipe;
        GridSampler2DGradCast<bfloat16_t, GridSampler2DGradTilingData> castOp;
        castOp.Init(tilingData, gmTensor, &tpipe);
        castOp.Process();
    }
    if (TILING_KEY_IS(7)) {
        GridSampler2DGradBicubic<float, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitBicubicLocalTensor();
        op.Process();
    }
    if (TILING_KEY_IS(8)) {
        InitWorkspace(tilingData, workspace);
        SyncAll();
        GridSampler2DGradBicubicFP16<float, half, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitBicubicLocalTensor();
        op.Process();
        SyncAll();
        pipe.Destroy();
        TPipe tpipe;
        GridSampler2DGradCast<half, GridSampler2DGradTilingData> castOp;
        castOp.Init(tilingData, gmTensor, &tpipe);
        castOp.Process();
    }
    if (TILING_KEY_IS(9)) {
        InitWorkspace(tilingData, workspace);
        SyncAll();
        GridSampler2DGradBicubicFP16<float, bfloat16_t, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitBicubicLocalTensor();
        op.Process();
        SyncAll();
        pipe.Destroy();
        TPipe tpipe;
        GridSampler2DGradCast<bfloat16_t, GridSampler2DGradTilingData> castOp;
        castOp.Init(tilingData, gmTensor, &tpipe);
        castOp.Process();
    }
#else
    TILING_KEY_IS(1);
    TILING_KEY_IS(2);
    TILING_KEY_IS(3);
    TILING_KEY_IS(4);
    TILING_KEY_IS(5);
    TILING_KEY_IS(6);
    TILING_KEY_IS(7);
    TILING_KEY_IS(8);
    TILING_KEY_IS(9);
    TILING_KEY_IS(10);
    TILING_KEY_IS(11);
    TILING_KEY_IS(12);
    TILING_KEY_IS(13);
    TILING_KEY_IS(14);
    TILING_KEY_IS(15);
    TILING_KEY_IS(16);
    TILING_KEY_IS(17);
    TILING_KEY_IS(18);
    TILING_KEY_IS(19);
    TILING_KEY_IS(20);
    TILING_KEY_IS(21);
    #if TILING_KEY_VAR == 1 || TILING_KEY_VAR == 2 
        GridSampler2DGradSimt<float> op;
        op.Init(&tilingData, gmTensor);
        op.Process();
    #elif TILING_KEY_VAR == 3 || TILING_KEY_VAR == 4 
        GridSampler2DGradSimt<half> op;
        op.Init(&tilingData, gmTensor);
        op.Process();
    #elif TILING_KEY_VAR == 5 || TILING_KEY_VAR == 6 
        GridSampler2DGradSimt<bfloat16_t> op;
        op.Init(&tilingData, gmTensor);
        op.Process();
    #elif TILING_KEY_VAR == 7 || TILING_KEY_VAR == 8 
        GridSampler2DGradSimtDet<float> op;
        op.Init(&tilingData, gmTensor);
        SyncAll();
        op.Process();
    #elif TILING_KEY_VAR == 9 || TILING_KEY_VAR == 10
        GridSampler2DGradSimtDet<half> op;
        op.Init(&tilingData, gmTensor);
        SyncAll();
        op.Process();
    #elif TILING_KEY_VAR == 11 || TILING_KEY_VAR == 12
        GridSampler2DGradSimtDet<bfloat16_t> op;
        op.Init(&tilingData, gmTensor);
        SyncAll();
        op.Process();
    #elif TILING_KEY_VAR == 13
        // bicubic, float32, non-deterministic
        GridSampler2DGradSimt<float> op;
        op.Init(&tilingData, gmTensor);
        op.Process();
    #elif TILING_KEY_VAR == 14
        // bicubic, float16, non-deterministic
        GridSampler2DGradSimt<half> op;
        op.Init(&tilingData, gmTensor);
        op.Process();
    #elif TILING_KEY_VAR == 15
        // bicubic, bfloat16, non-deterministic
        GridSampler2DGradSimt<bfloat16_t> op;
        op.Init(&tilingData, gmTensor);
        op.Process();
    #elif TILING_KEY_VAR == 16
        // bicubic, float32, deterministic
        GridSampler2DGradSimtDet<float> op;
        op.Init(&tilingData, gmTensor);
        SyncAll();
        op.Process();
    #elif TILING_KEY_VAR == 17
        // bicubic, float16, deterministic
        GridSampler2DGradSimtDet<half> op;
        op.Init(&tilingData, gmTensor);
        SyncAll();
        op.Process();
    #elif TILING_KEY_VAR == 18
        // bicubic, bfloat16, deterministic
        GridSampler2DGradSimtDet<bfloat16_t> op;
        op.Init(&tilingData, gmTensor);
        SyncAll();
        op.Process();
    #elif TILING_KEY_VAR == 19
        // bilinear, float32, simd
        GridSampler2DGrad<float, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitBilinearLocalTensor();
        op.Process();
    #elif TILING_KEY_VAR == 20
        // nearest, float32, simd
        GridSampler2DGrad<float, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitNearestLocalTensor();
        op.Process();
    #elif TILING_KEY_VAR == 21
        // bicubic, float32, simd
        GridSampler2DGradBicubic<float, GridSampler2DGradTilingData> op;
        op.Init(tilingData, gmTensor);
        op.InitBuffer(&pipe);
        op.InitBicubicLocalTensor();
        op.Process();
    #endif
#endif
}