/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file nms_with_mask_tiling_arch35.h
 * \brief nms_with_mask_tiling_arch35 info
 */

#ifndef NMS_WITH_MASK_REGBASE_TILING_ARCH35_H_
#define NMS_WITH_MASK_REGBASE_TILING_ARCH35_H_

#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/nms_with_mask_tiling_data.h"

namespace optiling {
class NMSWithMaskRegbaseTiling {
public:
    explicit NMSWithMaskRegbaseTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckOutputShape();
    ge::graphStatus SetTilingData();

private:
    NMSWithMaskTilingData* tilingData_;
    gert::TilingContext* tilingContext_;
    ge::DataType inputDtype_;
    int64_t boxesNum_ = 0;
    int64_t groupSize_ = 0;
    int64_t groupNum_ = 0;
    int64_t blockNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t headCoreNum_ = 0;
    int64_t blockPerHead_ = 0;
    float iouThreshold_ = 0.0f;
};

struct NMSWithMaskCompileInfo {
    int32_t maxBoxesNum;
    int64_t coreNum;
    int64_t ubSize;
    // bool isAscendC = false;
};

} // namespace optiling

#endif //  NMS_WITH_MASK_REGBASE_TILING_ARCH35_H_