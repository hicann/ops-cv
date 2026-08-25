#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TTK input generator for valid and deterministic rotated-box cases.

Most CSV rows start from TTK's seeded random data and are normalised into
valid rectangles.  Rows whose names begin with ``rotated_overlaps_f32_`` and
describe a geometric boundary case are filled with explicit coordinates.  The
same coordinates then reach both the independent PyTorch golden and the device,
which makes the important geometry semantics reproducible instead of relying
on an accidental random sample.
"""

import hashlib

import numpy as np

__input__ = {"kernel": {"rotated_overlaps": "rotated_overlaps_inputs"}}


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _copy_to(destination, value):
    if hasattr(destination, "copy_"):
        destination.copy_(destination.new_tensor(value))
    else:
        destination[...] = value.astype(destination.dtype, copy=False)


def _normalise(values, trans):
    result = np.nan_to_num(
        _to_numpy(values).astype(np.float32, copy=True),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    )
    if result.ndim != 3 or result.shape[1] != 5:
        raise ValueError("RotatedOverlaps inputs must have shape [B,5,N]")
    cx = np.clip(result[:, 0, :], -8.0, 8.0)
    cy = np.clip(result[:, 1, :], -8.0, 8.0)
    width = np.abs(result[:, 2, :]) * 0.5 + 0.5
    height = np.abs(result[:, 3, :]) * 0.5 + 0.5
    theta = np.tanh(result[:, 4, :]) * 135.0
    if trans:
        result[:, 0, :] = cx - width * 0.5
        result[:, 1, :] = cy - height * 0.5
        result[:, 2, :] = cx + width * 0.5
        result[:, 3, :] = cy + height * 0.5
    else:
        result[:, 0, :] = cx
        result[:, 1, :] = cy
        result[:, 2, :] = width
        result[:, 3, :] = height
    result[:, 4, :] = theta
    return result


def _fill_valid_grid(shape, trans):
    """Return deterministic, valid boxes for a `[B,5,N]` tensor."""
    values = np.zeros(shape, dtype=np.float32)
    batch, _, count = shape
    for batch_index in range(batch):
        locations = np.arange(count, dtype=np.float32)
        if trans:
            values[batch_index, 0, :] = locations * 0.75 - 1.0
            values[batch_index, 1, :] = locations * 0.5 - 1.0
            values[batch_index, 2, :] = values[batch_index, 0, :] + 2.0
            values[batch_index, 3, :] = values[batch_index, 1, :] + 2.0
        else:
            values[batch_index, 0, :] = locations * 0.75
            values[batch_index, 1, :] = locations * 0.5
            values[batch_index, 2, :] = 2.0
            values[batch_index, 3, :] = 2.0
        values[batch_index, 4, :] = (locations % 4.0) * 15.0
    return values


def _set_box(values, index, box):
    values[:, :, index] = np.asarray(box, dtype=np.float32)[None, :]


def _fixed_case_inputs(testcase_name, boxes_shape, queries_shape, trans):
    """Build one of the named, hand-auditable geometry cases when requested."""
    if not testcase_name.startswith("rotated_overlaps_f32_"):
        return None

    boxes = _fill_valid_grid(boxes_shape, trans)
    queries = _fill_valid_grid(queries_shape, trans)
    if testcase_name == "rotated_overlaps_f32_identical":
        _set_box(boxes, 0, (0.0, 0.0, 2.0, 2.0, 0.0))
        _set_box(queries, 0, (0.0, 0.0, 2.0, 2.0, 0.0))
    elif testcase_name == "rotated_overlaps_f32_disjoint":
        _set_box(boxes, 0, (0.0, 0.0, 2.0, 2.0, 0.0))
        _set_box(queries, 0, (10.0, 10.0, 2.0, 2.0, 0.0))
    elif testcase_name == "rotated_overlaps_f32_contained":
        _set_box(boxes, 0, (0.0, 0.0, 4.0, 4.0, 0.0))
        _set_box(queries, 0, (0.0, 0.0, 2.0, 2.0, 0.0))
    elif testcase_name == "rotated_overlaps_f32_edge_touch":
        _set_box(boxes, 0, (0.0, 0.0, 2.0, 2.0, 0.0))
        _set_box(queries, 0, (2.0, 0.0, 2.0, 2.0, 0.0))
    elif testcase_name == "rotated_overlaps_f32_positive_sliver":
        _set_box(boxes, 0, (0.0, 0.0, 2.0, 2.0, 0.0))
        _set_box(queries, 0, (1.9999999, 0.0, 2.0, 2.0, 0.0))
    elif testcase_name == "rotated_overlaps_f32_theta45":
        _set_box(boxes, 0, (0.0, 0.0, 2.0, 2.0, 0.0))
        _set_box(queries, 0, (0.0, 0.0, 2.0, 2.0, 45.0))
    elif testcase_name == "rotated_overlaps_f32_xyxyt_identical":
        if not trans:
            raise ValueError("xyxyt fixed case requires trans=True")
        _set_box(boxes, 0, (-1.0, -1.0, 1.0, 1.0, 0.0))
        _set_box(queries, 0, (-1.0, -1.0, 1.0, 1.0, 0.0))
    elif testcase_name == "rotated_overlaps_f32_invalid_tail":
        if trans:
            raise ValueError("invalid-tail fixed case is encoded as xywht")
        # Last two queries deliberately violate the frozen validity contract.
        queries[:, 2, -2] = 0.0
        queries[:, 2, -1] = -1.0
    elif testcase_name == "rotated_overlaps_f32_invalid_middle":
        if trans:
            raise ValueError("invalid-middle fixed case is encoded as xywht")
        queries[:, 3, 1] = 0.0
    elif testcase_name == "rotated_overlaps_f32_invalid_nan":
        if trans:
            raise ValueError("invalid-nan fixed case is encoded as xywht")
        queries[:, 0, 0] = np.nan
    else:
        return None
    return boxes, queries


_GENERALIZED_PREFIX = "rotated_overlaps_gen_"


def _geometry(shape, rng, coordinate_scale=8.0, min_extent=0.25, max_extent=5.0):
    """Create valid xywht geometry before encoding the public layout."""
    batch, _, count = shape
    return {
        "cx": rng.uniform(-coordinate_scale, coordinate_scale, (batch, count)).astype(
            np.float32
        ),
        "cy": rng.uniform(-coordinate_scale, coordinate_scale, (batch, count)).astype(
            np.float32
        ),
        "width": rng.uniform(min_extent, max_extent, (batch, count)).astype(np.float32),
        "height": rng.uniform(min_extent, max_extent, (batch, count)).astype(
            np.float32
        ),
        "theta": rng.uniform(-179.0, 179.0, (batch, count)).astype(np.float32),
    }


def _copy_geometry(source, count):
    """Repeat source boxes as needed while preserving every geometric field."""
    indices = np.arange(count, dtype=np.int64) % source["cx"].shape[1]
    return {name: values[:, indices].copy() for name, values in source.items()}


def _encode_geometry(geometry, trans):
    batch, count = geometry["cx"].shape
    values = np.empty((batch, 5, count), dtype=np.float32)
    if trans:
        values[:, 0, :] = geometry["cx"] - geometry["width"] * 0.5
        values[:, 1, :] = geometry["cy"] - geometry["height"] * 0.5
        values[:, 2, :] = geometry["cx"] + geometry["width"] * 0.5
        values[:, 3, :] = geometry["cy"] + geometry["height"] * 0.5
    else:
        values[:, 0, :] = geometry["cx"]
        values[:, 1, :] = geometry["cy"]
        values[:, 2, :] = geometry["width"]
        values[:, 3, :] = geometry["height"]
    values[:, 4, :] = geometry["theta"]
    return values


def _invalidate_extent(values, trans, case_index):
    """Inject finite zero/negative extents at start, middle and tail."""
    count = values.shape[2]
    indices = tuple(dict.fromkeys((0, count // 2, count - 1)))
    for offset, index in enumerate(indices):
        kind = (case_index + offset) % 2
        if trans:
            if kind == 0:
                values[:, 2, index] = values[:, 0, index]
            else:
                values[:, 3, index] = values[:, 1, index] - 1.0
        else:
            if kind == 0:
                values[:, 2, index] = 0.0
            else:
                values[:, 3, index] = -1.0


def _invalidate_nonfinite(values, case_index):
    """Inject NaN, +Inf and -Inf at distinct geometric fields."""
    count = values.shape[2]
    indices = tuple(dict.fromkeys((0, count // 2, count - 1)))
    special_values = (np.nan, np.inf, -np.inf)
    for offset, index in enumerate(indices):
        field = (case_index + offset) % 5
        values[:, field, index] = special_values[offset % len(special_values)]


def _generalized_case_inputs(testcase_name, boxes_shape, queries_shape, trans):
    """Return one deterministic generalized case, or None for ordinary rows."""
    if not testcase_name.startswith(_GENERALIZED_PREFIX):
        return None
    try:
        mode, index_text = testcase_name[len(_GENERALIZED_PREFIX) :].rsplit("_", 1)
        case_index = int(index_text)
    except ValueError as error:
        raise ValueError(
            f"invalid generalized testcase name: {testcase_name}"
        ) from error

    digest = hashlib.sha256(testcase_name.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    boxes_geo = _geometry(boxes_shape, rng)
    queries_geo = _geometry(queries_shape, rng)
    query_count = queries_shape[2]

    if mode == "identical":
        queries_geo = _copy_geometry(boxes_geo, query_count)
    elif mode == "disjoint":
        boxes_geo["cx"] = rng.uniform(-40.0, -24.0, boxes_geo["cx"].shape).astype(
            np.float32
        )
        boxes_geo["cy"] = rng.uniform(-40.0, -24.0, boxes_geo["cy"].shape).astype(
            np.float32
        )
        queries_geo["cx"] = rng.uniform(24.0, 40.0, queries_geo["cx"].shape).astype(
            np.float32
        )
        queries_geo["cy"] = rng.uniform(24.0, 40.0, queries_geo["cy"].shape).astype(
            np.float32
        )
    elif mode == "positive_sliver":
        boxes_geo["cx"][:, 0] = 0.0
        boxes_geo["cy"][:, 0] = 0.0
        boxes_geo["width"][:, 0] = 2.0
        boxes_geo["height"][:, 0] = 2.0
        boxes_geo["theta"][:, 0] = 0.0
        queries_geo["cx"][:, 0] = np.float32(1.9999999)
        queries_geo["cy"][:, 0] = 0.0
        queries_geo["width"][:, 0] = 2.0
        queries_geo["height"][:, 0] = 2.0
        queries_geo["theta"][:, 0] = 0.0
    elif mode == "contained":
        queries_geo = _copy_geometry(boxes_geo, query_count)
        queries_geo["width"] *= rng.uniform(
            0.10, 0.70, queries_geo["width"].shape
        ).astype(np.float32)
        queries_geo["height"] *= rng.uniform(
            0.10, 0.70, queries_geo["height"].shape
        ).astype(np.float32)
    elif mode == "edge_touch":
        boxes_geo["theta"].fill(0.0)
        queries_geo = _copy_geometry(boxes_geo, query_count)
        queries_geo["theta"].fill(0.0)
        queries_geo["width"] = rng.uniform(
            0.25, 5.0, queries_geo["width"].shape
        ).astype(np.float32)
        queries_geo["height"] = rng.uniform(
            0.25, 5.0, queries_geo["height"].shape
        ).astype(np.float32)
        queries_geo["cx"] += (
            boxes_geo["width"][:, np.arange(query_count) % boxes_shape[2]] * 0.5
        )
        queries_geo["cx"] += queries_geo["width"] * 0.5
    elif mode == "near_parallel":
        queries_geo = _copy_geometry(boxes_geo, query_count)
        queries_geo["cx"] += rng.uniform(-0.05, 0.05, queries_geo["cx"].shape).astype(
            np.float32
        )
        queries_geo["cy"] += rng.uniform(-0.05, 0.05, queries_geo["cy"].shape).astype(
            np.float32
        )
        queries_geo["theta"] += rng.uniform(
            -0.02, 0.02, queries_geo["theta"].shape
        ).astype(np.float32)
    elif mode == "angle_sweep":
        boxes_geo["theta"] = np.tile(
            np.linspace(-179.0, 179.0, boxes_shape[2], dtype=np.float32),
            (boxes_shape[0], 1),
        )
        queries_geo["theta"] = np.tile(
            np.linspace(179.0, -179.0, queries_shape[2], dtype=np.float32),
            (queries_shape[0], 1),
        )
    elif mode == "large_coord":
        boxes_geo = _geometry(
            boxes_shape, rng, coordinate_scale=512.0, min_extent=1.0, max_extent=24.0
        )
        queries_geo = _geometry(
            queries_shape, rng, coordinate_scale=512.0, min_extent=1.0, max_extent=24.0
        )
    elif mode == "tiny_rotated":
        boxes_geo = _geometry(
            boxes_shape,
            rng,
            coordinate_scale=0.01,
            min_extent=5.0e-4,
            max_extent=2.0e-3,
        )
        extent = rng.uniform(5.0e-4, 2.0e-3, boxes_geo["width"].shape).astype(
            np.float32
        )
        boxes_geo["width"] = extent
        boxes_geo["height"] = extent.copy()
        queries_geo = _copy_geometry(boxes_geo, query_count)
        queries_geo["theta"] += 45.0
    elif mode == "invalid_query":
        pass
    elif mode == "invalid_box":
        pass
    elif mode == "nonfinite_query":
        pass
    elif mode == "nonfinite_box":
        pass
    elif mode != "random":
        raise ValueError(f"unsupported generalized mode: {mode}")

    boxes = _encode_geometry(boxes_geo, trans)
    queries = _encode_geometry(queries_geo, trans)
    if mode == "invalid_query":
        _invalidate_extent(queries, trans, case_index)
    elif mode == "invalid_box":
        _invalidate_extent(boxes, trans, case_index)
    elif mode == "nonfinite_query":
        _invalidate_nonfinite(queries, case_index)
    elif mode == "nonfinite_box":
        _invalidate_nonfinite(boxes, case_index)
    return boxes, queries


def rotated_overlaps_inputs(boxes, query_boxes, *extra_args, **kwargs):
    """Normalise generated inputs or inject a deterministic named case."""
    del extra_args
    trans = bool(kwargs.get("trans", False))
    testcase_name = str(kwargs.get("testcase_name", ""))
    fixed = _fixed_case_inputs(
        testcase_name, tuple(boxes.shape), tuple(query_boxes.shape), trans
    )
    if fixed is None:
        fixed = _generalized_case_inputs(
            testcase_name, tuple(boxes.shape), tuple(query_boxes.shape), trans
        )
    if fixed is None:
        fixed = (_normalise(boxes, trans), _normalise(query_boxes, trans))
    _copy_to(boxes, fixed[0])
    _copy_to(query_boxes, fixed[1])
    return [boxes, query_boxes]
