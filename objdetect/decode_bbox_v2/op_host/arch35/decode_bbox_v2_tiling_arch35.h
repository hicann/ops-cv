/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DECODE_BBOX_V2_TILING_ARCH35_H
#define DECODE_BBOX_V2_TILING_ARCH35_H

#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"

namespace optiling {

/**
 * TilingFuncDecodeBboxV2 — public TilingFunc entry.
 *
 * Workflow:
 *   1. GetPlatformInfo: totalUb / availableCoreNum (dynamic, no hardcoding).
 *   2. CheckInputs: dtype/format/rank/attr/shape validation + layout normalization
 *      (dim0 = reversedBox ? shape[1] : shape[0]).
 *   3. ComputeUbSplit: per-box 256B alignment, numCalcBufs fp16=3 / fp32=0.
 *   4. ComputeMultiCoreSplit: 512 box alignment, dim0=0 short circuit.
 *   5. FillAndLogTilingData: all fields + scales/decodeClip/invScales/halfVal.
 *   6. SetBlockDim + SetTilingKey (reversedBox ? F4N(1) : N4(0)).
 *
 * Returns: ge::GRAPH_SUCCESS / ge::GRAPH_FAILED.
 */
ge::graphStatus TilingFuncDecodeBboxV2(gert::TilingContext* context);

} // namespace optiling

#endif
