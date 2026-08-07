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
"""TTK input generator for BatchMultiClassNonMaxSuppression.

Random coordinates do not necessarily form valid ``[y_min, x_min, y_max,
x_max]`` boxes.  NMS would then filter most candidates before the suppression
path is exercised.  This generator normalizes the generated data into valid
boxes and, when optional inputs are present, makes their values valid too.
"""

import numpy as np


__input__ = {
    "kernel": {
        "batch_multi_class_non_max_suppression": (
            "batch_multi_class_non_max_suppression_inputs"
        ),
    },
}


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _copy_to(destination, value):
    if hasattr(destination, "copy_"):
        destination.copy_(destination.new_tensor(value))
    else:
        destination[...] = value.astype(destination.dtype, copy=False)


def _transpose_box_from_args(extra_args, kwargs):
    del extra_args
    return bool(kwargs["transpose_box"]) if "transpose_box" in kwargs else None


def _valid_boxes(boxes, transpose_box):
    values = np.nan_to_num(_to_numpy(boxes).astype(np.float32, copy=True))
    if values.ndim != 4:
        raise ValueError("boxes must be rank 4")

    if transpose_box is None:
        if values.shape[-1] == 4 and values.shape[2] != 4:
            transpose_box = False
        elif values.shape[2] == 4 and values.shape[-1] != 4:
            transpose_box = True
        elif values.shape[-1] == 4:
            transpose_box = False
        else:
            raise ValueError("boxes must contain a coordinate dimension of size 4")

    canonical = values.transpose(0, 3, 1, 2) if transpose_box else values
    if canonical.shape[-1] != 4:
        raise ValueError("boxes must use [B, N, q, 4] or [B, q, 4, N] layout")

    mins = np.minimum(canonical[..., :2], canonical[..., 2:])
    maxs = np.maximum(canonical[..., :2], canonical[..., 2:])
    # A sizable minimum extent survives float16 conversion and guarantees that
    # every generated candidate passes the geometric validity check.
    canonical[..., :2] = mins
    canonical[..., 2:] = np.maximum(maxs, mins + 0.125)
    normalized = canonical.transpose(0, 2, 3, 1) if transpose_box else canonical
    return normalized, canonical


def _valid_scores(scores):
    values = np.nan_to_num(_to_numpy(scores).astype(np.float32, copy=True))
    if values.ndim != 3:
        raise ValueError("scores must be rank 3")
    # CSV cases commonly request [0, 1].  For broader default ranges, sigmoid
    # gives score_threshold meaningful coverage without changing valid ranges.
    if np.any(values < 0.0) or np.any(values > 1.0):
        values = 1.0 / (1.0 + np.exp(-np.clip(values, -20.0, 20.0)))
    return values


def _valid_clip_window(clip_window, canonical_boxes):
    if clip_window is None:
        return
    values = _to_numpy(clip_window)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("clip_window must have shape [B, 4]")
    if values.shape[0] != canonical_boxes.shape[0]:
        raise ValueError("clip_window batch size must match boxes")

    y_min = canonical_boxes[..., 0].min(axis=(1, 2)) - 1.0
    x_min = canonical_boxes[..., 1].min(axis=(1, 2)) - 1.0
    y_max = canonical_boxes[..., 2].max(axis=(1, 2)) + 1.0
    x_max = canonical_boxes[..., 3].max(axis=(1, 2)) + 1.0
    _copy_to(clip_window, np.stack((y_min, x_min, y_max, x_max), axis=1))


def _valid_num_boxes(num_valid_boxes, batch_size, boxes_num):
    if num_valid_boxes is None:
        return
    values = _to_numpy(num_valid_boxes)
    if values.ndim != 1 or values.shape[0] != batch_size:
        raise ValueError("num_valid_boxes must have shape [B]")
    # Cover both the full-input and truncated-input paths when B > 1.
    counts = boxes_num - (np.arange(batch_size) % min(boxes_num, 2))
    _copy_to(num_valid_boxes, counts)


def batch_multi_class_non_max_suppression_inputs(
    boxes, scores, clip_window=None, num_valid_boxes=None, *extra_args, **kwargs
):
    """Make generated NMS inputs mutually valid in place for TTK."""
    transpose_box = _transpose_box_from_args(extra_args, kwargs)
    valid_boxes, canonical_boxes = _valid_boxes(boxes, transpose_box)
    score_values = _valid_scores(scores)

    if canonical_boxes.shape[0] != score_values.shape[0]:
        raise ValueError("boxes and scores batch sizes must match")
    if canonical_boxes.shape[1] != score_values.shape[1]:
        raise ValueError("boxes and scores candidate counts must match")
    if canonical_boxes.shape[2] not in (1, score_values.shape[2]):
        raise ValueError("boxes class dimension must be 1 or match scores")

    _copy_to(boxes, valid_boxes)
    _copy_to(scores, score_values)
    _valid_clip_window(clip_window, canonical_boxes)
    _valid_num_boxes(
        num_valid_boxes, canonical_boxes.shape[0], canonical_boxes.shape[1]
    )
    # Kernel direct-call mode consumes the normalized input arrays.
    return [boxes, scores, clip_window, num_valid_boxes]
