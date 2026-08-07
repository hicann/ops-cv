#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Torch CPU golden plugin for kernel-direct TTK Promote validation."""

from __future__ import annotations

import numpy as np
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

__spec__ = {
    "batch_multi_class_non_max_suppression": "BatchMultiClassNonMaxSuppressionTestSpec"
}


def _iou(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    mins = torch.maximum(boxes[:, :2], box[:2])
    maxes = torch.minimum(boxes[:, 2:], box[2:])
    inter = torch.clamp(maxes - mins, min=0).prod(dim=1)
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
    box_area = (box[2:] - box[:2]).prod()
    return inter / torch.clamp(areas + box_area - inter, min=1e-12)


def _nms(boxes, scores, score_threshold, iou_threshold, max_size):
    valid = scores > score_threshold
    valid &= boxes[:, 2] > boxes[:, 0]
    valid &= boxes[:, 3] > boxes[:, 1]
    indices = torch.where(valid)[0]
    if indices.numel() == 0:
        return []
    boxes = boxes[indices]
    scores = scores[indices]
    order = torch.argsort(scores, descending=True, stable=True)
    boxes = boxes[order]
    scores = scores[order]
    suppressed = torch.zeros(scores.numel(), dtype=torch.bool)
    kept = []
    for index in range(scores.numel()):
        if bool(suppressed[index]):
            continue
        kept.append((scores[index], boxes[index].clone()))
        if len(kept) >= max_size:
            break
        remaining = torch.where(~suppressed)[0]
        remaining = remaining[remaining > index]
        if remaining.numel():
            suppressed[
                remaining[_iou(boxes[remaining], boxes[index]) > iou_threshold]
            ] = True
    return kept


def _torch_golden(
    boxes,
    scores,
    clip_window=None,
    num_valid_boxes=None,
    score_threshold=0.5,
    iou_threshold=0.5,
    max_size_per_class=100,
    max_total_size=100,
    change_coordinate_frame=False,
    transpose_box=False,
    **kwargs,
):
    boxes = torch.as_tensor(boxes, device="cpu")
    scores = torch.as_tensor(scores, device="cpu")
    clip_window = (
        None if clip_window is None else torch.as_tensor(clip_window, device="cpu")
    )
    num_valid_boxes = (
        None
        if num_valid_boxes is None
        else torch.as_tensor(num_valid_boxes, device="cpu")
    )
    dtype = boxes.dtype
    batch_size, num_classes = scores.shape[0], scores.shape[2]
    num_boxes = boxes.shape[3] if transpose_box else boxes.shape[1]
    box_classes = boxes.shape[1] if transpose_box else boxes.shape[2]
    out_boxes = torch.zeros((batch_size, max_total_size, 4), dtype=dtype)
    out_scores = torch.zeros((batch_size, max_total_size), dtype=dtype)
    out_classes = torch.zeros((batch_size, max_total_size), dtype=dtype)
    out_num = torch.zeros(batch_size, dtype=torch.int32)
    for batch in range(batch_size):
        valid_count = (
            num_boxes
            if num_valid_boxes is None
            else max(0, min(int(num_valid_boxes[batch]), num_boxes))
        )
        candidates = []
        for class_id in range(num_classes):
            class_scores = scores[batch, :, class_id].clone()
            class_scores[valid_count:] = -torch.inf
            box_class = 0 if box_classes == 1 else class_id
            class_boxes = (
                boxes[batch, box_class].T.contiguous().clone()
                if transpose_box
                else boxes[batch, :, box_class].clone()
            )
            if clip_window is not None:
                y_min, x_min, y_max, x_max = clip_window[batch]
                class_boxes[:, 0].clamp_(min=y_min)
                class_boxes[:, 1].clamp_(min=x_min)
                class_boxes[:, 2].clamp_(max=y_max)
                class_boxes[:, 3].clamp_(max=x_max)
                if change_coordinate_frame:
                    if bool(y_max <= y_min) or bool(x_max <= x_min):
                        continue
                    class_boxes[:, (0, 2)] = (class_boxes[:, (0, 2)] - y_min) / (
                        y_max - y_min
                    )
                    class_boxes[:, (1, 3)] = (class_boxes[:, (1, 3)] - x_min) / (
                        x_max - x_min
                    )
            for score, box in _nms(
                class_boxes,
                class_scores,
                float(score_threshold),
                float(iou_threshold),
                int(max_size_per_class),
            ):
                candidates.append((float(score), class_id, box))
        candidates.sort(key=lambda item: -item[0])
        count = min(len(candidates), int(max_total_size))
        out_num[batch] = count
        for index, (score, class_id, box) in enumerate(candidates[:count]):
            out_boxes[batch, index] = box
            out_scores[batch, index] = score
            out_classes[batch, index] = class_id
    return [value.numpy() for value in (out_boxes, out_scores, out_classes, out_num)]


def _golden(*inputs, **attrs):
    """Run Torch in a clean process, outside TTK's CANN-initialized worker."""
    with tempfile.TemporaryDirectory(prefix="batch_nms_cpu_golden_") as directory:
        request = Path(directory) / "request.pkl"
        response = Path(directory) / "response.pkl"
        request.write_bytes(
            pickle.dumps((inputs, attrs), protocol=pickle.HIGHEST_PROTOCOL)
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--torch-worker",
                str(request),
                str(response),
            ],
            check=True,
        )
        return pickle.loads(response.read_bytes())


class BatchMultiClassNonMaxSuppressionTestSpec:
    golden = staticmethod(_golden)
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }

    @staticmethod
    def pre_compare(*outputs, **kwargs):
        half = len(outputs) // 2
        for offset in (0, half):
            boxes, scores, classes, count = outputs[offset : offset + 4]
            for batch in range(boxes.shape[0]):
                valid = int(np.asarray(count).reshape(-1)[batch])
                if valid <= 1:
                    continue
                order = np.argsort(
                    -np.asarray(scores)[batch, :valid].astype(np.float64), kind="stable"
                )
                boxes[batch, :valid] = boxes[batch, order]
                scores[batch, :valid] = scores[batch, order]
                classes[batch, :valid] = classes[batch, order]


def _worker(request: str, response: str) -> int:
    global torch
    import torch

    inputs, attrs = pickle.loads(Path(request).read_bytes())
    outputs = _torch_golden(*inputs, **attrs)
    Path(response).write_bytes(pickle.dumps(outputs, protocol=pickle.HIGHEST_PROTOCOL))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--torch-worker":
        raise SystemExit("expected --torch-worker REQUEST RESPONSE")
    raise SystemExit(_worker(sys.argv[2], sys.argv[3]))
