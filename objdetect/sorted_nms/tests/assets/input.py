#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def customize_inputs(
    boxes,
    sorted_scores,
    input_indices,
    max_output_size,
    iou_threshold,
    score_threshold,
    **kwargs,
):
    """Generate deterministic SortedNMS inputs with retain and suppress paths."""
    boxes_num = boxes.shape[0]
    box_ids = np.arange(boxes_num, dtype=np.int64)
    group_ids = box_ids // 4

    # Four boxes in each group are identical, while different groups are
    # spatially separated. This guarantees both suppress and retain paths.
    x1 = (group_ids % 64) * 8
    y1 = (group_ids // 64) * 8
    generated_boxes = np.stack((x1, y1, x1 + 4, y1 + 4), axis=1)
    boxes[...] = generated_boxes.astype(boxes.dtype)

    # Scores are already sorted by position as required by SortedNMS.
    if boxes_num > 0:
        scores = np.linspace(0.99, 0.01, boxes_num, dtype=np.float32)
        sorted_scores[...] = scores.astype(sorted_scores.dtype)
        # A fixed coprime affine permutation exercises the input-index map.
        multiplier = boxes_num - 1 if boxes_num % 2 == 0 else boxes_num - 2
        multiplier = max(multiplier, 1)
        input_indices[...] = ((box_ids * multiplier + 1) % boxes_num).astype(np.int32)

    return (
        boxes,
        sorted_scores,
        input_indices,
        max_output_size,
        iou_threshold,
        score_threshold,
    )
