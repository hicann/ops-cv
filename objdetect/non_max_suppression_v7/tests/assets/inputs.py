#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Input normalization for NonMaxSuppressionV7 TTK cases."""

import numpy as np

__input__ = {"kernel": {"non_max_suppression_v7": "non_max_suppression_v7_inputs"}}


def _copy(dst, value):
    if hasattr(dst, "copy_"):
        dst.copy_(dst.new_tensor(value))
    else:
        dst[...] = np.asarray(value, dtype=dst.dtype)


def _scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return float(np.asarray(value).reshape(-1)[0])


def _set_box(boxes, box_index, y_min, x_min, y_max, x_max, center_point_box):
    if center_point_box:
        boxes[:, box_index, 0] = (x_min + x_max) * 0.5
        boxes[:, box_index, 1] = (y_min + y_max) * 0.5
        boxes[:, box_index, 2] = x_max - x_min
        boxes[:, box_index, 3] = y_max - y_min
    else:
        boxes[:, box_index, :] = (y_min, x_min, y_max, x_max)


def _special_case_inputs(testcase_name, boxes, scores, center_point_box, source_dtype):
    prefix = "nms_v7_special_"
    if not testcase_name.startswith(prefix):
        return
    try:
        mode = testcase_name[len(prefix) :].rsplit("_", 1)[0]
    except ValueError as error:
        raise ValueError(f"invalid special testcase name: {testcase_name}") from error

    box_count = boxes.shape[1]
    for box_index in range(box_count):
        origin = float(box_index * 4)
        _set_box(
            boxes,
            box_index,
            origin,
            origin,
            origin + 2.0,
            origin + 2.0,
            center_point_box,
        )
    score_axis = np.linspace(1.0, 0.1, box_count, dtype=np.float32)
    scores[...] = score_axis[None, None, :]

    if mode == "identical":
        for box_index in range(box_count):
            _set_box(boxes, box_index, 0.0, 0.0, 2.0, 2.0, center_point_box)
    elif mode == "zero_area":
        _set_box(boxes, 0, 0.0, 0.0, 0.0, 2.0, center_point_box)
        if box_count > 1:
            _set_box(boxes, 1, 4.0, 4.0, 6.0, 4.0, center_point_box)
    elif mode == "large_coord":
        for box_index in range(box_count):
            origin = 4096.0 + float(box_index * 64)
            _set_box(
                boxes,
                box_index,
                origin,
                -origin,
                origin + 32.0,
                -origin + 32.0,
                center_point_box,
            )
    elif mode == "tiny_iou":
        extent = 1.0e-3 if source_dtype == np.float16 else 1.0e-7
        _set_box(boxes, 0, 0.0, 0.0, extent, extent, center_point_box)
        if box_count > 1:
            inner = extent * 0.9
            _set_box(boxes, 1, 0.0, 0.0, inner, inner, center_point_box)
    elif mode == "score_boundary":
        scores.fill(-1.0e-3)
        scores[..., 0] = 0.0
        if box_count > 1:
            scores[..., 1] = 1.0e-3
    elif mode == "iou_boundary":
        _set_box(boxes, 0, 0.0, 0.0, 2.0, 2.0, center_point_box)
        if box_count > 1:
            _set_box(boxes, 1, 0.0, 0.0, 2.0, 1.0, center_point_box)
    elif mode == "score_extreme":
        scores.fill(-65504.0)
        scores[..., 0] = 65504.0
        if box_count > 1:
            scores[..., 1] = 0.0
    else:
        raise ValueError(f"unsupported special mode: {mode}")


def non_max_suppression_v7_inputs(
    boxes,
    scores,
    max_output_size=None,
    iou_threshold=None,
    score_threshold=None,
    index_id=None,
    *args,
    center_point_box=0,
    **kwargs,
):
    del args
    testcase_name = str(kwargs.get("testcase_name", ""))
    source_boxes = np.asarray(
        boxes.detach().cpu() if hasattr(boxes, "detach") else boxes
    )
    source_dtype = source_boxes.dtype
    b = np.nan_to_num(source_boxes.astype(np.float32, copy=True))
    s = np.nan_to_num(
        np.asarray(
            scores.detach().cpu() if hasattr(scores, "detach") else scores,
            dtype=np.float32,
        )
    )
    if (
        b.ndim != 3
        or b.shape[-1] != 4
        or s.ndim != 3
        or b.shape[0] != s.shape[0]
        or b.shape[1] != s.shape[2]
    ):
        raise ValueError("expected boxes [B,N,4] and scores [B,C,N]")
    if center_point_box:
        b[..., 2:] = np.abs(b[..., 2:])
    else:
        lo = np.minimum(b[..., :2], b[..., 2:])
        hi = np.maximum(b[..., :2], b[..., 2:])
        b[..., :2] = lo
        b[..., 2:] = hi

    _special_case_inputs(testcase_name, b, s, bool(center_point_box), source_dtype)
    if testcase_name.startswith("nms_v7_index_"):
        s.fill(-1.0)
        s[..., 0] = 1.0
    _copy(boxes, b)
    _copy(scores, np.clip(s, -1.0e4, 1.0e4))
    if max_output_size is not None:
        _copy(
            max_output_size,
            np.array(
                [
                    max(
                        0,
                        min(
                            int(np.prod(np.asarray(max_output_size).shape)) * 0
                            + int(np.asarray(max_output_size).flat[0]),
                            b.shape[1],
                        ),
                    )
                ],
                dtype=np.int32,
            ),
        )
    if iou_threshold is not None:
        _copy(iou_threshold, np.array([_scalar(iou_threshold)], dtype=np.float32))
    if score_threshold is not None:
        _copy(score_threshold, np.array([_scalar(score_threshold)], dtype=np.float32))
    if index_id is not None:
        index_shape = np.asarray(index_id).shape
        if len(index_shape) != 4 or index_shape[-1] not in (3, 4):
            raise ValueError("index_id must have shape [B, C, N, 3] or [B, C, N, 4]")
        ids = np.zeros(index_shape, dtype=np.float16)
        ids[..., 0] = np.arange(index_shape[0], dtype=np.float16)[:, None, None]
        ids[..., 1] = np.arange(index_shape[1], dtype=np.float16)[None, :, None]
        if index_shape[-1] == 3:
            ids[..., 2] = np.arange(index_shape[2], dtype=np.float16)[None, None, :]
        else:
            ids[..., 2] = (np.arange(index_shape[2]) // 1000).astype(np.float16)[
                None, None, :
            ]
            ids[..., 3] = (np.arange(index_shape[2]) % 1000).astype(np.float16)[
                None, None, :
            ]
        if testcase_name == "nms_v7_index_negative":
            ids[..., 0] = -3.0
            ids[..., 1] = -2.0
            ids[..., 2] = -1.0
            if index_shape[-1] == 4:
                ids[..., 3] = -7.0
        elif testcase_name == "nms_v7_index_fractional":
            ids[..., 0] = -3.5
            ids[..., 1] = 2.75
            ids[..., 2] = -1.5
            if index_shape[-1] == 4:
                ids[..., 3] = 7.75
        elif testcase_name == "nms_v7_index_nan":
            ids[..., 0] = np.nan
            ids[..., 1] = np.inf
            ids[..., 2] = -np.inf
            if index_shape[-1] == 4:
                ids[..., 3] = np.nan
        _copy(index_id, ids)
    return [boxes, scores, max_output_size, iou_threshold, score_threshold, index_id]
