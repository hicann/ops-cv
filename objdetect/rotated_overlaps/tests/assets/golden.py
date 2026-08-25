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
"""PyTorch reference and CUDA third-party implementation for RotatedOverlaps.

PyTorch and TensorFlow do not expose an operator with the exact public
semantics used here (intersection area, degree angles and both xywht/xyxyt
layouts). The reference therefore follows the repository fallback rule and
uses independent PyTorch small-op composition instead of NumPy geometry.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_inputs_plugin():
    input_path = Path(__file__).with_name("inputs.py")
    spec = importlib.util.spec_from_file_location(
        "rotated_overlaps_test_inputs", input_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rotated_overlaps_inputs


_rotated_overlaps_inputs_impl = _load_inputs_plugin()


def rotated_overlaps_inputs(*args, **kwargs):
    """Expose a source-defined input hook so TTK plugin discovery can find it."""
    return _rotated_overlaps_inputs_impl(*args, **kwargs)


def _normalise_boxes(values, trans):
    """Return `(cx, cy, width, height, theta)` fields and a validity mask."""
    if values.ndim != 3 or values.shape[1] != 5:
        raise ValueError("RotatedOverlaps inputs must have shape [B,5,N]")
    raw = values.transpose(1, 2)
    valid = torch.isfinite(raw).all(dim=-1)
    if trans:
        x1, y1, x2, y2, theta = raw.unbind(dim=-1)
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
    else:
        center_x, center_y, width, height, theta = raw.unbind(dim=-1)
    valid = valid & (width > 0.0) & (height > 0.0)
    return center_x, center_y, width, height, theta, valid


def _corners(fields):
    """Construct four counter-clockwise corners for every rotated box."""
    center_x, center_y, width, height, theta, _ = fields
    radians = torch.deg2rad(theta)
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    half_width = width * 0.5
    half_height = height * 0.5
    offset_x = torch.stack((-half_width, half_width, half_width, -half_width), dim=-1)
    offset_y = torch.stack(
        (-half_height, -half_height, half_height, half_height), dim=-1
    )
    x = (
        center_x.unsqueeze(-1)
        + offset_x * cosine.unsqueeze(-1)
        - offset_y * sine.unsqueeze(-1)
    )
    y = (
        center_y.unsqueeze(-1)
        + offset_x * sine.unsqueeze(-1)
        + offset_y * cosine.unsqueeze(-1)
    )
    return torch.stack((x, y), dim=-1)


def _cross(lhs, rhs):
    return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]


def _points_inside(points, rectangles):
    """Test each candidate point against all four CCW rectangle edges."""
    starts = rectangles
    ends = torch.roll(rectangles, shifts=-1, dims=-2)
    edges = ends - starts
    relative = points.unsqueeze(-2) - starts.unsqueeze(-3)
    crosses = (
        edges[..., 0].unsqueeze(-2) * relative[..., 1]
        - edges[..., 1].unsqueeze(-2) * relative[..., 0]
    )
    return (crosses >= 0.0).all(dim=-1)


def _edge_intersections(first, second):
    """Return the 16 edge-pair intersection candidates and valid flags."""
    first_start = first.unsqueeze(-2)
    first_vector = (torch.roll(first, shifts=-1, dims=-2) - first).unsqueeze(-2)
    second_start = second.unsqueeze(-3)
    second_vector = (torch.roll(second, shifts=-1, dims=-2) - second).unsqueeze(-3)
    relative = second_start - first_start
    denominator = _cross(first_vector, second_vector)
    nonzero = denominator != 0.0
    safe_denominator = torch.where(nonzero, denominator, torch.ones_like(denominator))
    first_ratio = _cross(relative, second_vector) / safe_denominator
    second_ratio = _cross(relative, first_vector) / safe_denominator
    valid = (
        nonzero
        & (first_ratio >= 0.0)
        & (first_ratio <= 1.0)
        & (second_ratio >= 0.0)
        & (second_ratio <= 1.0)
    )
    points = first_start + first_ratio.unsqueeze(-1) * first_vector
    return points.flatten(start_dim=-3, end_dim=-2), valid.flatten(start_dim=-2)


def _intersection_area(first, second, pair_valid):
    """Calculate every pair area from 24 independently generated candidates."""
    first_pairs = first.unsqueeze(2).expand(-1, -1, second.shape[1], -1, -1)
    second_pairs = second.unsqueeze(1).expand(-1, first.shape[1], -1, -1, -1)
    first_inside = _points_inside(first_pairs, second_pairs)
    second_inside = _points_inside(second_pairs, first_pairs)
    intersections, intersection_valid = _edge_intersections(first_pairs, second_pairs)
    candidates = torch.cat((first_pairs, second_pairs, intersections), dim=-2)
    candidate_valid = torch.cat(
        (first_inside, second_inside, intersection_valid), dim=-1
    ) & pair_valid.unsqueeze(-1)

    masked_candidates = torch.where(
        candidate_valid.unsqueeze(-1), candidates, torch.zeros_like(candidates)
    )
    candidate_count = candidate_valid.sum(dim=-1)
    center = masked_candidates.sum(dim=-2) / candidate_count.clamp_min(1).unsqueeze(-1)
    relative = candidates - center.unsqueeze(-2)
    angle = torch.atan2(relative[..., 1], relative[..., 0])
    angle = torch.where(candidate_valid, angle, torch.full_like(angle, 4.0))
    order = torch.argsort(angle, dim=-1)
    ordered = torch.gather(candidates, -2, order.unsqueeze(-1).expand(*order.shape, 2))
    ordered_valid = torch.gather(candidate_valid, -1, order)
    first_point = ordered[..., :1, :]
    ordered = torch.where(ordered_valid.unsqueeze(-1), ordered, first_point)
    ordered = ordered - center.unsqueeze(-2)
    following = torch.roll(ordered, shifts=-1, dims=-2)
    doubled_area = _cross(ordered, following).sum(dim=-1).abs()
    area = doubled_area * 0.5
    return torch.where(
        pair_valid & (candidate_count >= 3), area, torch.zeros_like(area)
    )


def _rotated_overlaps_torch(boxes, query_boxes, trans=False):
    if boxes.ndim != 3 or query_boxes.ndim != 3:
        raise ValueError("boxes and query_boxes must both be rank 3")
    if boxes.shape[0] != query_boxes.shape[0]:
        raise ValueError("boxes and query_boxes batch dimensions must match")
    first_fields = _normalise_boxes(boxes, trans)
    second_fields = _normalise_boxes(query_boxes, trans)
    pair_valid = first_fields[-1].unsqueeze(2) & second_fields[-1].unsqueeze(1)
    return _intersection_area(
        _corners(first_fields), _corners(second_fields), pair_valid
    )


KERNEL_OUTPUT_TOLERANCE = {"float32": {"standard": "cross_check", "level": "L1"}}
GEIR_OUTPUT_TOLERANCE = {"float32": {"standard": "cross_check", "level": "L1"}}
ONNX_OUTPUT_TOLERANCE = {"float32": {"standard": "cross_check", "level": "L1"}}


def _golden_impl(boxes, query_boxes, *, trans=False, **kwargs):
    """High-precision CPU composition shared by delivery-route wrappers."""
    del kwargs
    boxes_tensor = torch.from_numpy(boxes).to(torch.float64)
    queries_tensor = torch.from_numpy(query_boxes).to(torch.float64)
    result = _rotated_overlaps_torch(boxes_tensor, queries_tensor, bool(trans))
    return result.to(torch.float32).numpy()


def rotated_overlaps_kernel_golden(boxes, query_boxes, trans=False, **kwargs):
    return _golden_impl(boxes, query_boxes, trans=trans, **kwargs)


def rotated_overlaps_geir_golden(boxes, query_boxes, trans=False, **kwargs):
    return _golden_impl(boxes, query_boxes, trans=trans, **kwargs)


def rotated_overlaps_onnx_golden(boxes, query_boxes, trans=False, **kwargs):
    """Standalone ONNX importer golden; TTK does not execute this route."""
    return _golden_impl(boxes, query_boxes, trans=trans, **kwargs)


class _RotatedOverlapsTorchBaseline:
    """PyTorch small-op baseline with a conversion-free timed call."""

    def __init__(self, *, trans=False, **kwargs):
        del kwargs
        self.trans = bool(trans)

    def __call__(self, boxes, query_boxes, **kwargs):
        del kwargs
        return _rotated_overlaps_torch(boxes, query_boxes, self.trans)


class RotatedOverlapsKernelThirdParty(_RotatedOverlapsTorchBaseline):
    """Third-party baseline for the Kernel route."""


class RotatedOverlapsGeirThirdParty(_RotatedOverlapsTorchBaseline):
    """Third-party baseline for the GEIR route."""


class RotatedOverlapsOnnxThirdParty(_RotatedOverlapsTorchBaseline):
    """Third-party baseline for the standalone ONNX importer route."""


class RotatedOverlapsTestSpec:
    """Kernel TestSpec; GEIR uses the same TTK lookup key."""

    golden = staticmethod(rotated_overlaps_kernel_golden)
    third_party = {"torch": RotatedOverlapsKernelThirdParty}
    tolerance = KERNEL_OUTPUT_TOLERANCE


class RotatedOverlapsGeirTestSpec:
    """Standalone GEIR route specification for route-level review."""

    golden = staticmethod(rotated_overlaps_geir_golden)
    third_party = {"torch": RotatedOverlapsGeirThirdParty}
    tolerance = GEIR_OUTPUT_TOLERANCE


class RotatedOverlapsOnnxTestSpec:
    """Standalone ONNX route specification for the dedicated verifier."""

    golden = staticmethod(rotated_overlaps_onnx_golden)
    third_party = {"torch": RotatedOverlapsOnnxThirdParty}
    tolerance = ONNX_OUTPUT_TOLERANCE


__golden__ = {
    "kernel": {"rotated_overlaps": "rotated_overlaps_kernel_golden"},
    "geir": {"rotated_overlaps": "rotated_overlaps_geir_golden"},
}
__input__ = {
    "kernel": {"rotated_overlaps": "rotated_overlaps_inputs"},
    "geir": {"rotated_overlaps": "rotated_overlaps_inputs"},
}
__spec__ = {"rotated_overlaps": "RotatedOverlapsTestSpec"}


def _self_test():
    cases = (
        ((0.0, 0.0, 2.0, 2.0, 0.0), (0.0, 0.0, 2.0, 2.0, 0.0), 4.0),
        ((0.0, 0.0, 2.0, 2.0, 0.0), (10.0, 10.0, 2.0, 2.0, 0.0), 0.0),
        ((0.0, 0.0, 2.0, 2.0, 0.0), (0.0, 0.0, 2.0, 2.0, 45.0), 3.3137085),
        (
            (0.0, 0.0, 0.001, 0.001, 0.0),
            (0.0, 0.0, 0.001, 0.001, 45.0),
            8.284271e-7,
        ),
        (
            (0.0, 0.0, 2.0, 2.0, 0.0),
            (
                float(torch.tensor(1.9999999, dtype=torch.float32)),
                0.0,
                2.0,
                2.0,
                0.0,
            ),
            2.384185791015625e-7,
        ),
    )
    for first, second, expected in cases:
        boxes = torch.tensor(first, dtype=torch.float64).reshape(1, 5, 1)
        queries = torch.tensor(second, dtype=torch.float64).reshape(1, 5, 1)
        actual = float(_rotated_overlaps_torch(boxes, queries)[0, 0, 0])
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-12)


if __name__ == "__main__":
    _self_test()
