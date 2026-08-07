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
"""NumPy golden for the CombinedNonMaxSuppression kernel used by TTK."""

import numpy as np


__golden__ = {
    "kernel": {
        "combined_non_max_suppression": "combined_non_max_suppression_golden",
    },
}


def _scalar(tensor, name):
    value = np.asarray(tensor)
    if value.size != 1:
        raise ValueError(f"{name} must contain exactly one element")
    return value.reshape(-1)[0]


def _iou(lhs, rhs):
    """Compute IoU with the same float32 operation order as the kernel."""
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)

    lhs_y_min = np.minimum(lhs[0], lhs[2])
    lhs_x_min = np.minimum(lhs[1], lhs[3])
    lhs_y_max = np.maximum(lhs[0], lhs[2])
    lhs_x_max = np.maximum(lhs[1], lhs[3])
    rhs_y_min = np.minimum(rhs[0], rhs[2])
    rhs_x_min = np.minimum(rhs[1], rhs[3])
    rhs_y_max = np.maximum(rhs[0], rhs[2])
    rhs_x_max = np.maximum(rhs[1], rhs[3])

    intersect_h = np.maximum(
        np.minimum(lhs_y_max, rhs_y_max) - np.maximum(lhs_y_min, rhs_y_min),
        np.float32(0.0),
    )
    intersect_w = np.maximum(
        np.minimum(lhs_x_max, rhs_x_max) - np.maximum(lhs_x_min, rhs_x_min),
        np.float32(0.0),
    )
    intersection = np.float32(intersect_h * intersect_w)
    lhs_area = np.float32(
        np.maximum(lhs_y_max - lhs_y_min, np.float32(0.0))
        * np.maximum(lhs_x_max - lhs_x_min, np.float32(0.0))
    )
    rhs_area = np.float32(
        np.maximum(rhs_y_max - rhs_y_min, np.float32(0.0))
        * np.maximum(rhs_x_max - rhs_x_min, np.float32(0.0))
    )
    union_area = np.float32(np.float32(lhs_area + rhs_area) - intersection)
    if union_area > np.float32(0.0):
        return np.float32(intersection / union_area)
    return np.float32(0.0)


def _select_class(boxes, scores, max_output, iou_threshold, score_threshold):
    """Greedy per-class NMS; score ties prefer the lower anchor index."""
    suppressed = np.zeros(scores.shape[0], dtype=np.bool_)
    selected = []

    for _ in range(max_output):
        best_index = -1
        best_score = np.float32(-np.finfo(np.float32).max)
        for anchor, score in enumerate(scores):
            score = np.float32(score)
            if suppressed[anchor] or not score > score_threshold:
                continue
            if score > best_score or (
                score == best_score and (best_index < 0 or anchor < best_index)
            ):
                best_score = score
                best_index = anchor

        if best_index < 0:
            break
        selected.append((best_score, best_index))
        for anchor in range(scores.shape[0]):
            if not suppressed[anchor] and (
                anchor == best_index
                or _iou(boxes[best_index], boxes[anchor]) > iou_threshold
            ):
                suppressed[anchor] = True

    return selected


def combined_non_max_suppression_golden(
    boxes,
    scores,
    max_output_size_per_class,
    max_total_size,
    iou_threshold,
    score_threshold,
    *,
    pad_per_class=False,
    clip_boxes=True,
    **_unused,
):
    """Run per-class NMS and merge each batch by descending score."""
    boxes_f32 = np.asarray(boxes, dtype=np.float32)
    scores_f32 = np.asarray(scores, dtype=np.float32)
    if boxes_f32.ndim != 4 or scores_f32.ndim != 3:
        raise ValueError("boxes must be rank 4 and scores must be rank 3")

    batch_size, num_boxes, box_classes, coords = boxes_f32.shape
    if coords != 4 or scores_f32.shape[:2] != (batch_size, num_boxes):
        raise ValueError("invalid boxes/scores shape relationship")
    num_classes = scores_f32.shape[2]
    if box_classes not in (1, num_classes):
        raise ValueError("the q dimension of boxes must be 1 or num_classes")

    max_per_class = int(_scalar(max_output_size_per_class, "max_output_size_per_class"))
    max_total = int(_scalar(max_total_size, "max_total_size"))
    iou_limit = np.float32(_scalar(iou_threshold, "iou_threshold"))
    score_limit = np.float32(_scalar(score_threshold, "score_threshold"))
    output_size = (
        min(max_total, max_per_class * num_classes) if pad_per_class else max_total
    )

    nmsed_boxes = np.zeros((batch_size, output_size, 4), dtype=np.float32)
    nmsed_scores = np.zeros((batch_size, output_size), dtype=np.float32)
    nmsed_classes = np.zeros((batch_size, output_size), dtype=np.float32)
    valid_detections = np.zeros((batch_size,), dtype=np.int32)

    for batch in range(batch_size):
        candidates = []
        for class_index in range(num_classes):
            box_class = 0 if box_classes == 1 else class_index
            selected = _select_class(
                boxes_f32[batch, :, box_class, :],
                scores_f32[batch, :, class_index],
                max_per_class,
                iou_limit,
                score_limit,
            )
            for class_rank, (score, anchor) in enumerate(selected):
                candidate_index = class_index * max_per_class + class_rank
                candidates.append(
                    (score, candidate_index, class_index, anchor, box_class)
                )

        # This is equivalent to the kernel's k-way merge. candidate_index
        # reproduces its tie-break rule for equal scores.
        candidates.sort(key=lambda item: (-float(item[0]), item[1]))
        valid = min(output_size, len(candidates))
        valid_detections[batch] = valid
        for output_index, candidate in enumerate(candidates[:valid]):
            score, _, class_index, anchor, box_class = candidate
            output_box = boxes_f32[batch, anchor, box_class, :]
            if clip_boxes:
                output_box = np.clip(output_box, np.float32(0.0), np.float32(1.0))
            nmsed_boxes[batch, output_index, :] = output_box
            nmsed_scores[batch, output_index] = score
            nmsed_classes[batch, output_index] = np.float32(class_index)

    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
