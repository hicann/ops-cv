/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_TILING_ARCH35_H_
#define BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_TILING_ARCH35_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/batch_multi_class_non_max_suppression_tiling_data.h"

namespace optiling {
struct BatchMultiClassNonMaxSuppressionCompileInfo {
    uint32_t coreNum{0};
    uint64_t ubSize{0};
};

class BatchMultiClassNonMaxSuppressionTiling {
public:
    explicit BatchMultiClassNonMaxSuppressionTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus CheckAndParse();
    ge::graphStatus SetTilingData();

    gert::TilingContext* context_;
    BatchMultiClassNonMaxSuppressionTilingData* tilingData_{nullptr};
    int64_t batch_{0};
    int64_t boxesNum_{0};
    int64_t classesNum_{0};
    int64_t boxClassesNum_{0};
    int64_t maxSizePerClass_{0};
    int64_t maxTotalSize_{0};
    float scoreThreshold_{0.0F};
    float iouThreshold_{0.0F};
    bool hasClipWindow_{false};
    bool hasNumValidBoxes_{false};
    bool changeCoordinateFrame_{false};
    bool transposeBox_{false};
};
} // namespace optiling

#endif // BATCH_MULTI_CLASS_NON_MAX_SUPPRESSION_TILING_ARCH35_H_
