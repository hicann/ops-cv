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

import torch
import numpy

__spec__ = {
    "batch_multi_class_non_max_suppression": "BatchMultiClassNonMaxSuppressionTestSpec",
}


def _compute_iou_vectorized(boxes_a, box_b):
    iy_min = torch.max(boxes_a[:, 0], box_b[0])
    ix_min = torch.max(boxes_a[:, 1], box_b[1])
    iy_max = torch.min(boxes_a[:, 2], box_b[2])
    ix_max = torch.min(boxes_a[:, 3], box_b[3])
    inter_h = torch.clamp(iy_max - iy_min, min=0.0)
    inter_w = torch.clamp(ix_max - ix_min, min=0.0)
    inter_area = inter_h * inter_w
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = torch.clamp(area_a + area_b - inter_area, min=1e-12)
    return inter_area / union_area


def _nms_single_class(
    class_boxes, class_scores, score_threshold, iou_threshold, max_size
):
    valid_mask = class_scores > score_threshold
    valid_mask &= class_boxes[:, 2] > class_boxes[:, 0]
    valid_mask &= class_boxes[:, 3] > class_boxes[:, 1]
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return []
    valid_scores = class_scores[valid_indices]
    valid_boxes = class_boxes[valid_indices]
    sorted_order = torch.argsort(valid_scores, descending=True, stable=True)
    sorted_scores = valid_scores[sorted_order]
    sorted_boxes = valid_boxes[sorted_order]
    num_candidates = len(sorted_scores)
    suppressed = torch.zeros(num_candidates, dtype=torch.bool)
    kept = []
    for i in range(num_candidates):
        if suppressed[i]:
            continue
        kept.append((sorted_scores[i].item(), sorted_boxes[i].clone()))
        if len(kept) >= max_size:
            break
        if i < num_candidates - 1:
            rem_indices = torch.where(~suppressed)[0]
            rem_indices = rem_indices[rem_indices > i]
            if len(rem_indices) == 0:
                continue
            iou = _compute_iou_vectorized(sorted_boxes[rem_indices], sorted_boxes[i])
            suppressed[rem_indices[iou > iou_threshold]] = True
    return kept


def _golden_impl(
    boxes,
    scores,
    clip_window,
    num_valid_boxes,
    score_threshold,
    iou_threshold,
    max_size_per_class,
    max_total_size,
    change_coordinate_frame,
    transpose_box,
):
    out_dtype = boxes.dtype
    batch_size = scores.shape[0]
    num_classes = scores.shape[2]
    out_boxes = torch.zeros(batch_size, max_total_size, 4, dtype=torch.float32)
    out_scores = torch.zeros(batch_size, max_total_size, dtype=torch.float32)
    out_classes = torch.zeros(batch_size, max_total_size, dtype=torch.float32)
    out_num = torch.zeros(batch_size, dtype=torch.int32)
    if transpose_box:
        num_boxes = boxes.shape[3]
        box_classes = boxes.shape[1]
    else:
        num_boxes = boxes.shape[1]
        box_classes = boxes.shape[2]
    for b in range(batch_size):
        valid_count = num_boxes
        if num_valid_boxes is not None:
            nvb = int(num_valid_boxes[b].item())
            valid_count = max(0, min(nvb, num_boxes))
        has_clip = clip_window is not None
        if has_clip:
            cw = clip_window[b].float()
            cy_min, cx_min, cy_max, cx_max = (
                cw[0].item(),
                cw[1].item(),
                cw[2].item(),
                cw[3].item(),
            )
            if change_coordinate_frame and (cy_max <= cy_min or cx_max <= cx_min):
                continue
        candidates = []
        for c in range(num_classes):
            class_scores = scores[b, :, c].float().clone()
            if valid_count < num_boxes:
                class_scores[valid_count:] = float("-inf")
            box_class_idx = 0 if box_classes == 1 else c
            if transpose_box:
                class_boxes = (
                    boxes[b, box_class_idx, :, :].float().T.contiguous().clone()
                )
            else:
                class_boxes = boxes[b, :, box_class_idx, :].float().clone()
            if has_clip:
                class_boxes[:, 0] = torch.clamp(class_boxes[:, 0], min=cy_min)
                class_boxes[:, 1] = torch.clamp(class_boxes[:, 1], min=cx_min)
                class_boxes[:, 2] = torch.clamp(class_boxes[:, 2], max=cy_max)
                class_boxes[:, 3] = torch.clamp(class_boxes[:, 3], max=cx_max)
                if change_coordinate_frame:
                    win_h = cy_max - cy_min
                    win_w = cx_max - cx_min
                    class_boxes[:, 0] = (class_boxes[:, 0] - cy_min) / win_h
                    class_boxes[:, 2] = (class_boxes[:, 2] - cy_min) / win_h
                    class_boxes[:, 1] = (class_boxes[:, 1] - cx_min) / win_w
                    class_boxes[:, 3] = (class_boxes[:, 3] - cx_min) / win_w
            kept = _nms_single_class(
                class_boxes,
                class_scores,
                score_threshold,
                iou_threshold,
                max_size_per_class,
            )
            for score_val, box_val in kept:
                candidates.append((score_val, c, box_val))
        candidates.sort(key=lambda x: -x[0])
        num_valid = min(len(candidates), max_total_size)
        out_num[b] = num_valid
        for i in range(num_valid):
            score_val, class_id, box_val = candidates[i]
            out_scores[b, i] = score_val
            out_classes[b, i] = float(class_id)
            out_boxes[b, i] = box_val
    return [
        out_boxes.to(out_dtype),
        out_scores.to(out_dtype),
        out_classes.to(out_dtype),
        out_num,
    ]


def _to_numpy(arr):
    """Convert torch tensor or numpy array to numpy."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return numpy.asarray(arr)


def _to_torch(arr):
    """Adapt kernel-direct numpy inputs to torch tensors."""
    if arr is None or isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(numpy.asarray(arr))


def _valid_count(counts, batch_index, batch_size):
    """Read logical count from either [B] or fusion-internal [B,8] storage."""
    values = _to_numpy(counts)
    if values.ndim >= 2 and values.shape[0] == batch_size:
        return int(values[batch_index].reshape(-1)[0])
    return int(values.reshape(-1)[batch_index])


def _maximum_match_count(allowed_pairs):
    """Return the maximum one-to-one match count for a boolean pair matrix."""
    row_count, column_count = allowed_pairs.shape
    if row_count == 0 or column_count == 0:
        return 0
    column_owner = numpy.full(column_count, -1, dtype=numpy.int64)

    def augment(row, visited):
        for column in numpy.flatnonzero(allowed_pairs[row]):
            if visited[column]:
                continue
            visited[column] = True
            owner = int(column_owner[column])
            if owner == -1 or augment(owner, visited):
                column_owner[column] = row
                return True
        return False

    matched = 0
    for row in range(row_count):
        if augment(row, numpy.zeros(column_count, dtype=numpy.bool_)):
            matched += 1
    return matched


def _comparison_tolerance(*arrays):
    """Use the precision thresholds carried by the TTK case matrix."""
    if any(_to_numpy(value).dtype == numpy.float16 for value in arrays):
        return 1e-2, 1e-2
    return 1e-4, 1e-4


class BatchMultiClassNonMaxSuppressionTestSpec:
    def golden(
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
        *args,
        **kwargs,
    ):
        # XPU server appends metadata such as input_formats positionally when
        # invoking a third-party golden.  It is not an operator attribute and
        # must not affect the reference computation.
        is_torch_input = isinstance(boxes, torch.Tensor)
        results = _golden_impl(
            _to_torch(boxes),
            _to_torch(scores),
            _to_torch(clip_window),
            _to_torch(num_valid_boxes),
            float(score_threshold),
            float(iou_threshold),
            int(max_size_per_class),
            int(max_total_size),
            bool(change_coordinate_frame),
            bool(transpose_box),
        )
        if is_torch_input:
            return results
        return [result.cpu().numpy() for result in results]

    def pre_compare(*outputs, **kwargs):
        """Sort detections by score descending in both NPU and golden outputs.

        Layout: func(*npu_outputs, *golden_outputs)
          outputs[0]=npu_boxes, outputs[1]=npu_scores, outputs[2]=npu_classes, outputs[3]=npu_num
          outputs[4]=golden_boxes, outputs[5]=golden_scores, outputs[6]=golden_classes, outputs[7]=golden_num
        In-place mode: modify via [:], return None.
        """
        npu_boxes = _to_numpy(outputs[0])
        npu_scores = _to_numpy(outputs[1])
        npu_classes = _to_numpy(outputs[2])
        npu_num = _to_numpy(outputs[3])
        golden_boxes = _to_numpy(outputs[4])
        golden_scores = _to_numpy(outputs[5])
        golden_classes = _to_numpy(outputs[6])
        golden_num = _to_numpy(outputs[7])

        batch_size = npu_boxes.shape[0]
        for b in range(batch_size):
            n_cnt = _valid_count(npu_num, b, batch_size)
            g_cnt = _valid_count(golden_num, b, batch_size)

            if n_cnt > 1:
                n_order = numpy.argsort(
                    -npu_scores[b, :n_cnt].astype(numpy.float64), kind="stable"
                )
                npu_boxes[b, :n_cnt] = npu_boxes[b, n_order]
                npu_scores[b, :n_cnt] = npu_scores[b, n_order]
                npu_classes[b, :n_cnt] = npu_classes[b, n_order]

            if g_cnt > 1:
                g_order = numpy.argsort(
                    -golden_scores[b, :g_cnt].astype(numpy.float64), kind="stable"
                )
                golden_boxes[b, :g_cnt] = golden_boxes[b, g_order]
                golden_scores[b, :g_cnt] = golden_scores[b, g_order]
                golden_classes[b, :g_cnt] = golden_classes[b, g_order]

    def compare(*outputs, **kwargs):
        """Strict, order-independent comparison for valid NMS detections.

        Layout: func(*npu_outputs, *golden_outputs)
        Returns list[dict], one per output (boxes, scores, classes, num).
        """
        npu_boxes_raw = _to_numpy(outputs[0])
        npu_scores_raw = _to_numpy(outputs[1])
        golden_boxes_raw = _to_numpy(outputs[4])
        golden_scores_raw = _to_numpy(outputs[5])
        rtol, atol = _comparison_tolerance(
            npu_boxes_raw, npu_scores_raw, golden_boxes_raw, golden_scores_raw
        )
        npu_boxes = npu_boxes_raw.astype(numpy.float64)
        npu_scores = npu_scores_raw.astype(numpy.float64)
        npu_classes = _to_numpy(outputs[2]).astype(numpy.float64)
        npu_num = _to_numpy(outputs[3])
        golden_boxes = golden_boxes_raw.astype(numpy.float64)
        golden_scores = golden_scores_raw.astype(numpy.float64)
        golden_classes = _to_numpy(outputs[6]).astype(numpy.float64)
        golden_num = _to_numpy(outputs[7])

        batch_size = npu_boxes.shape[0]
        counts_equal = golden_boxes.shape[0] == batch_size
        counts_in_range = counts_equal
        total_npu = 0
        total_golden = 0
        total_box_matches = 0
        total_tuple_matches = 0
        count_mismatches = []

        for b in range(batch_size):
            n_cnt = _valid_count(npu_num, b, batch_size)
            g_cnt = _valid_count(golden_num, b, batch_size)
            total_npu += max(n_cnt, 0)
            total_golden += max(g_cnt, 0)
            if n_cnt != g_cnt:
                counts_equal = False
                count_mismatches.append(f"batch {b}: npu={n_cnt}, golden={g_cnt}")

            n_capacity = min(
                npu_boxes.shape[1], npu_scores.shape[1], npu_classes.shape[1]
            )
            g_capacity = min(
                golden_boxes.shape[1],
                golden_scores.shape[1],
                golden_classes.shape[1],
            )
            if n_cnt < 0 or g_cnt < 0 or n_cnt > n_capacity or g_cnt > g_capacity:
                counts_in_range = False
                continue
            if n_cnt == 0 or g_cnt == 0:
                continue

            n_boxes = npu_boxes[b, :n_cnt]
            g_boxes = golden_boxes[b, :g_cnt]
            n_scores_b = npu_scores[b, :n_cnt]
            g_scores_b = golden_scores[b, :g_cnt]
            n_classes_b = npu_classes[b, :n_cnt]
            g_classes_b = golden_classes[b, :g_cnt]

            box_close = numpy.all(
                numpy.isclose(
                    n_boxes[:, None, :],
                    g_boxes[None, :, :],
                    rtol=rtol,
                    atol=atol,
                    equal_nan=False,
                ),
                axis=2,
            )
            class_equal = n_classes_b[:, None] == g_classes_b[None, :]
            score_close = numpy.isclose(
                n_scores_b[:, None],
                g_scores_b[None, :],
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
            total_box_matches += _maximum_match_count(box_close)
            total_tuple_matches += _maximum_match_count(
                box_close & class_equal & score_close
            )

        expected = total_golden
        denominator = max(expected, total_npu, 1)
        box_rate = total_box_matches / denominator * 100.0
        tuple_rate = total_tuple_matches / denominator * 100.0
        box_pass = counts_equal and counts_in_range and total_box_matches == expected
        tuple_pass = (
            counts_equal and counts_in_range and total_tuple_matches == expected
        )
        count_pass = counts_equal and counts_in_range

        boxes_result = {
            "pass": box_pass,
            "precision": round(box_rate, 2),
            "error_info": f"box_matches={total_box_matches}/{expected} "
            f"npu_det={total_npu} rtol={rtol} atol={atol}",
            "metrics": {
                "box_match_rate": box_rate,
                "total_npu_det": total_npu,
                "total_golden_det": total_golden,
                "total_box_matches": total_box_matches,
            },
        }
        scores_result = {
            "pass": tuple_pass,
            "precision": round(tuple_rate, 2),
            "error_info": f"complete_detection_matches={total_tuple_matches}/{expected}",
            "metrics": {"complete_detection_match_rate": tuple_rate},
        }
        classes_result = {
            "pass": tuple_pass,
            "precision": round(tuple_rate, 2),
            "error_info": f"complete_detection_matches={total_tuple_matches}/{expected}",
            "metrics": {"complete_detection_match_rate": tuple_rate},
        }
        num_result = {
            "pass": count_pass,
            "precision": 100.0 if count_pass else 0.0,
            "error_info": "; ".join(count_mismatches)
            if count_mismatches
            else f"counts_equal={counts_equal} counts_in_range={counts_in_range}",
            "metrics": {
                "counts_equal": counts_equal,
                "counts_in_range": counts_in_range,
                "batch_size": batch_size,
            },
        }
        return [boxes_result, scores_result, classes_result, num_result]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
    }
