/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BOUNDING_BOX_DECODE_TILING_ARCH35_H
#define BOUNDING_BOX_DECODE_TILING_ARCH35_H

namespace optiling {

/**
 * TilingFunc: the tiling callback invoked by the CANN framework before kernel launch.
 *
 * Parameters:
 *   context — [in/out] tiling context providing input shapes, dtypes, and
 *             accepting the computed BoundingBoxDecodeTilingData.
 *
 * Returns:
 *   ge::GRAPH_SUCCESS on success.
 *
 * Side effects:
 *   - Reads input tensor metadata from context.
 *   - Computes totalLength, blockLength, tileLength, numBlocks.
 *   - Writes these values into BoundingBoxDecodeTilingData via context->GetTilingData.
 *   - Sets the block dimension via context->SetBlockDim.
 *   - Optionally sets workspace sizes via context->GetWorkspaceSizes.
 */
ge::graphStatus TilingFunc(gert::TilingContext* context);

} // namespace optiling

#endif
