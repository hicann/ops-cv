/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file image_projective_transform_apt.cpp
 * \brief Kernel entry for image_projective_transform operator
 *
 * dtype is handled by DTYPE_IMAGES macro (auto-instantiated per dtype).
 * Interpolation mode is dispatched at compile-time via TilingKey.
 */

#include "arch35/image_projective_transform_simt.h"

template <uint32_t interpMode>
__global__ __aicore__ void image_projective_transform(GM_ADDR images, GM_ADDR transforms, GM_ADDR output_shape,
                                                      GM_ADDR transformed_images, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ImageProjectiveTransformTilingData);
    GET_TILING_DATA_WITH_STRUCT(ImageProjectiveTransformTilingData, tilingData, tiling);

    // Use tiling data as primary source (host-side tiling validates
    // output_shape has >= 2 elements before storing hOut/wOut).
    // Only read from GM when tiling data values are non-positive,
    // which indicates the host-side could not resolve output_shape.
    int32_t actualHOut = tilingData.hOut;
    int32_t actualWOut = tilingData.wOut;
    if (actualHOut <= 0 || actualWOut <= 0) {
        // Fallback: host tiling could not resolve output_shape (dynamic shape).
        // Validate the GM_ADDR before dereferencing; if invalid, fall back to
        // input image H/W (consistent with host-side ResolveOutputShape).
        if (output_shape != nullptr) {
            __gm__ int32_t* outputShapeGm = (__gm__ int32_t*)output_shape;
            actualHOut = outputShapeGm[0];
            actualWOut = outputShapeGm[1];
        }
        if (actualHOut <= 0 || actualWOut <= 0) {
            actualHOut = tilingData.hIn;
            actualWOut = tilingData.wIn;
        }
    }

    if constexpr (interpMode == IPT_TPL_BILINEAR) {
        NsImageProjectiveTransform::Process<DTYPE_IMAGES, IPT_TPL_BILINEAR>(images, transforms, transformed_images,
                                                                            &tilingData, actualHOut, actualWOut);
    } else {
        NsImageProjectiveTransform::Process<DTYPE_IMAGES, IPT_TPL_NEAREST>(images, transforms, transformed_images,
                                                                           &tilingData, actualHOut, actualWOut);
    }
}
