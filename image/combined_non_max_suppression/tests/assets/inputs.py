#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK input validation for CombinedNonMaxSuppression."""

import numpy as np


__input__ = {
    "kernel": {
        "combined_non_max_suppression": "combined_non_max_suppression_inputs",
    },
}


def _scalar_value(tensor, name):
    """Read a scalar tensor while retaining the original tensor for TTK."""
    value = np.asarray(tensor)
    if value.size != 1:
        raise ValueError(f"{name} must contain exactly one element")
    return value.reshape(-1)[0]


def combined_non_max_suppression_inputs(
    boxes,
    scores,
    max_output_size_per_class,
    max_total_size,
    iou_threshold,
    score_threshold,
    **_unused,
):
    """Validate and return mutually consistent kernel inputs.

    TTK creates tensors independently.  This hook rejects invalid shape and
    scalar combinations before they reach host tiling, while preserving the
    generated box and score values (reversed box corners are valid for this
    operator).
    """
    if boxes.ndim != 4 or scores.ndim != 3:
        raise ValueError("boxes must be rank 4 and scores must be rank 3")

    batch, num_boxes, box_classes, box_coords = boxes.shape
    score_batch, score_boxes, num_classes = scores.shape
    if batch <= 0 or num_boxes <= 0 or num_classes <= 0:
        raise ValueError("batch, num_boxes and num_classes must be positive")
    if box_coords != 4:
        raise ValueError("the last dimension of boxes must be 4")
    if (score_batch, score_boxes) != (batch, num_boxes):
        raise ValueError("boxes and scores batch/num_boxes dimensions must match")
    if box_classes not in (1, num_classes):
        raise ValueError("the q dimension of boxes must be 1 or num_classes")
    if num_boxes > 200000 or num_classes > 200:
        raise ValueError("num_boxes must be <= 200000 and num_classes must be <= 200")

    max_per_class = int(
        _scalar_value(max_output_size_per_class, "max_output_size_per_class")
    )
    max_total = int(_scalar_value(max_total_size, "max_total_size"))
    iou = float(_scalar_value(iou_threshold, "iou_threshold"))
    _scalar_value(score_threshold, "score_threshold")

    if not 1 <= max_per_class <= 1000:
        raise ValueError("max_output_size_per_class must be in [1, 1000]")
    if not 1 <= max_total <= 1000:
        raise ValueError("max_total_size must be in [1, 1000]")
    if not np.isfinite(iou) or not 0.0 <= iou <= 1.0:
        raise ValueError("iou_threshold must be finite and in [0, 1]")

    return (
        boxes,
        scores,
        max_output_size_per_class,
        max_total_size,
        iou_threshold,
        score_threshold,
    )
