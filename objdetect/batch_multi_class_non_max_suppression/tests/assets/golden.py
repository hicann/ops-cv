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


def _numpy_iou_matrix(boxes_a, boxes_b):
    """Compute IoU matrix between two sets of boxes [N,4] and [M,4] in yxyx format."""
    n = boxes_a.shape[0]
    m = boxes_b.shape[0]
    if n == 0 or m == 0:
        return numpy.zeros((n, m), dtype=numpy.float64)
    iy_min = numpy.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    ix_min = numpy.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    iy_max = numpy.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    ix_max = numpy.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    inter_h = numpy.clip(iy_max - iy_min, 0, None)
    inter_w = numpy.clip(ix_max - ix_min, 0, None)
    inter_area = inter_h * inter_w
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union_area = numpy.clip(area_a[:, None] + area_b[None, :] - inter_area, 1e-12, None)
    return inter_area / union_area


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
        **kwargs,
    ):
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
            n_cnt = int(npu_num.flatten()[b])
            g_cnt = int(golden_num.flatten()[b])

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
        """IoU-based matching comparison for NMS outputs.

        Layout: func(*npu_outputs, *golden_outputs)
        Returns list[dict], one per output (boxes, scores, classes, num).
        """
        npu_boxes = _to_numpy(outputs[0]).astype(numpy.float64)
        npu_scores = _to_numpy(outputs[1]).astype(numpy.float64)
        npu_classes = _to_numpy(outputs[2]).astype(numpy.float64)
        npu_num = _to_numpy(outputs[3]).flatten()
        golden_boxes = _to_numpy(outputs[4]).astype(numpy.float64)
        golden_scores = _to_numpy(outputs[5]).astype(numpy.float64)
        golden_classes = _to_numpy(outputs[6]).astype(numpy.float64)
        golden_num = _to_numpy(outputs[7]).flatten()

        batch_size = npu_boxes.shape[0]
        iou_match_threshold = 0.5
        score_rtol = 0.05

        total_npu_det = 0
        total_golden_det = 0
        total_matched = 0
        total_score_close = 0
        total_class_match = 0
        total_box_iou_sum = 0.0
        num_diff_sum = 0

        for b in range(batch_size):
            n_cnt = int(npu_num[b])
            g_cnt = int(golden_num[b])
            total_npu_det += n_cnt
            total_golden_det += g_cnt
            num_diff_sum += abs(n_cnt - g_cnt)

            if n_cnt == 0 and g_cnt == 0:
                continue
            if n_cnt == 0 or g_cnt == 0:
                continue

            n_boxes = npu_boxes[b, :n_cnt]
            g_boxes = golden_boxes[b, :g_cnt]
            n_scores_b = npu_scores[b, :n_cnt]
            g_scores_b = golden_scores[b, :g_cnt]
            n_classes_b = npu_classes[b, :n_cnt]
            g_classes_b = golden_classes[b, :g_cnt]

            iou_mat = _numpy_iou_matrix(n_boxes, g_boxes)
            # A single geometric box may legitimately be retained by several
            # classes.  Pure IoU ordering then pairs identical boxes
            # arbitrarily and can report an otherwise exact result as having
            # wrong scores/classes.  Prefer semantic agreement before IoU.
            class_equal = n_classes_b[:, None] == g_classes_b[None, :]
            score_close = numpy.isclose(
                n_scores_b[:, None], g_scores_b[None, :], rtol=score_rtol, atol=1e-3
            )
            matched_n = set()
            matched_g = set()
            pairs = []
            flat_order = numpy.lexsort(
                (
                    numpy.arange(iou_mat.size),
                    -iou_mat.flatten(),
                    -score_close.astype(numpy.int8).flatten(),
                    -class_equal.astype(numpy.int8).flatten(),
                )
            )
            for idx in flat_order:
                ni = int(idx // g_cnt)
                gi = int(idx % g_cnt)
                if iou_mat[ni, gi] < iou_match_threshold:
                    continue
                if ni in matched_n or gi in matched_g:
                    continue
                matched_n.add(ni)
                matched_g.add(gi)
                pairs.append((ni, gi, iou_mat[ni, gi]))

            total_matched += len(pairs)
            for ni, gi, iou_val in pairs:
                total_box_iou_sum += iou_val
                if numpy.isclose(
                    n_scores_b[ni], g_scores_b[gi], rtol=score_rtol, atol=1e-3
                ):
                    total_score_close += 1
                if n_classes_b[ni] == g_classes_b[gi]:
                    total_class_match += 1

        max_det = max(total_npu_det, total_golden_det, 1)
        if total_npu_det == 0 and total_golden_det == 0:
            # Empty detection sets are equivalent; treating the absence of
            # pairs as a 0% match creates a false negative for valid inputs.
            match_rate = 100.0
            avg_iou = 1.0
            score_rate = 100.0
            class_rate = 100.0
        else:
            match_rate = total_matched / max_det * 100.0
            avg_iou = total_box_iou_sum / max(total_matched, 1)
            score_rate = total_score_close / max(total_matched, 1) * 100.0
            class_rate = total_class_match / max(total_matched, 1) * 100.0
        num_accuracy = max(0.0, 100.0 - num_diff_sum / max(batch_size, 1) * 10)

        boxes_result = {
            "pass": match_rate >= 70.0 and avg_iou >= 0.8,
            "precision": round(match_rate, 2),
            "error_info": f"match_rate={match_rate:.1f}% avg_iou={avg_iou:.3f} "
            f"npu_det={total_npu_det} golden_det={total_golden_det} matched={total_matched}",
            "metrics": {
                "match_rate": match_rate,
                "avg_iou": avg_iou,
                "total_npu_det": total_npu_det,
                "total_golden_det": total_golden_det,
                "total_matched": total_matched,
            },
        }
        scores_result = {
            "pass": score_rate >= 80.0,
            "precision": round(score_rate, 2),
            "error_info": f"score_close={total_score_close}/{total_matched}",
            "metrics": {"score_match_rate": score_rate},
        }
        classes_result = {
            "pass": class_rate >= 80.0,
            "precision": round(class_rate, 2),
            "error_info": f"class_match={total_class_match}/{total_matched}",
            "metrics": {"class_match_rate": class_rate},
        }
        num_result = {
            "pass": num_accuracy >= 50.0,
            "precision": round(num_accuracy, 2),
            "error_info": f"num_diff_sum={num_diff_sum} over {batch_size} batches",
            "metrics": {"num_diff_sum": int(num_diff_sum), "batch_size": batch_size},
        }
        return [boxes_result, scores_result, classes_result, num_result]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
    }
