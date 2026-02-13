/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file col2im_tiling_data.h
 * \brief col2im_tiling_data.h
 */

#ifndef _COL2IM_REGBASE_TILING_DATA_H_
#define _COL2IM_REGBASE_TILING_DATA_H_

#include <string>

struct Col2imRegBaseTilingData {
    int64_t totalLength;
    int64_t outputSizeH;
    int64_t outputSizeW;
    int64_t kernelSizeH;
    int64_t kernelSizeW;
    int64_t dilationH;
    int64_t dilationW;
    int64_t paddingH;
    int64_t paddingW;
    int64_t strideH;
    int64_t strideW;
    int64_t colH;
    int64_t colW;

    std::string toString() const {
        return "totalLength = " + std::to_string(totalLength) + ", outputSizeH = " + std::to_string(outputSizeH) 
                + ", outputSizeW = " + std::to_string(outputSizeW) + ", kernelSizeH = " + std::to_string(kernelSizeH)
                + ", kernelSizeW = " + std::to_string(kernelSizeW) + ", dilationH = " + std::to_string(dilationH) 
                + ", dilationW = " + std::to_string(dilationW) + ", paddingH = " + std::to_string(paddingH)
                + ", paddingW = " + std::to_string(paddingW) + ", strideH = " + std::to_string(strideH) 
                + ", strideW = " + std::to_string(strideW) + ", colH = " + std::to_string(colH)
                + ", colW = " + std::to_string(colW);
    }
};
#endif
