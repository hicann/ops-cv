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
    # Reduce the represented input in degrees before multiplying by pi/180.
    # fmod also preserves tiny negative angles (positive remainder may round
    # them to 360). Direct conversion loses phase for large finite inputs.
    radians = torch.deg2rad(torch.fmod(theta, 360.0))
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


def _clipped_pair_area(first, second, valid):
    """Clip convex polygons without dividing by nearly parallel edge pairs.

    A crossing is computed only when consecutive signed distances have
    opposite signs. Its denominator is then a sum of magnitudes, avoiding
    the cancellation in the former 16 edge-pair intersection candidates.
    """
    # Start with the smaller rectangle. Clipping a near-float32-max rectangle
    # down to a small contained one subtracts huge, almost equal coordinates
    # and can lose the entire intersection even with float64 arithmetic.
    first_area = _cross(first[:, 1] - first[:, 0], first[:, 3] - first[:, 0]).abs()
    second_area = _cross(second[:, 1] - second[:, 0], second[:, 3] - second[:, 0]).abs()
    swap = (first_area > second_area).unsqueeze(-1).unsqueeze(-1)
    first, second = torch.where(swap, second, first), torch.where(swap, first, second)
    capacity = 16
    slots = torch.arange(capacity, device=first.device).unsqueeze(0)
    polygon = torch.zeros(
        (first.shape[0], capacity, 2), dtype=first.dtype, device=first.device
    )
    polygon[:, :4] = first
    count = torch.where(valid, 4, 0)
    for edge_index in range(4):
        active = slots < count.unsqueeze(-1)
        previous_index = torch.where(slots == 0, count.unsqueeze(-1) - 1, slots - 1)
        previous = torch.gather(
            polygon, 1, previous_index.clamp_min(0).unsqueeze(-1).expand(-1, -1, 2)
        )
        start = second[:, edge_index].unsqueeze(1)
        edge = (second[:, (edge_index + 1) % 4] - second[:, edge_index]).unsqueeze(1)
        # Signed-distance scale cancels in the crossing ratio. Normalising
        # the edge avoids overflow in float32 for very large valid boxes.
        scale = edge.abs().amax(dim=-1, keepdim=True)
        edge = edge / torch.where(scale > 0, scale, torch.ones_like(scale))
        distance = _cross(edge, polygon - start)
        previous_distance = _cross(edge, previous - start)
        inside = distance >= 0
        crossing = active & (inside != (previous_distance >= 0))
        denominator = torch.where(
            crossing, previous_distance - distance, torch.ones_like(distance)
        )
        ratio = torch.where(
            crossing, previous_distance / denominator, torch.zeros_like(distance)
        )
        intersection = previous + ratio.unsqueeze(-1) * (polygon - previous)
        candidates = torch.stack((intersection, polygon), dim=2).flatten(1, 2)
        keep = torch.stack((crossing, active & inside), dim=2).flatten(1, 2)
        positions = keep.to(torch.int64).cumsum(dim=1) - 1
        count = keep.sum(dim=1)
        polygon = torch.zeros_like(polygon).scatter_add(
            1,
            positions.clamp(0, capacity - 1).unsqueeze(-1).expand(-1, -1, 2),
            torch.where(keep.unsqueeze(-1), candidates, torch.zeros_like(candidates)),
        )
    active = slots < count.unsqueeze(-1)
    following_index = torch.where(slots + 1 < count.unsqueeze(-1), slots + 1, 0)
    relative = polygon - polygon[:, :1]
    following = torch.gather(
        relative, 1, following_index.unsqueeze(-1).expand(-1, -1, 2)
    )
    area = torch.where(active, _cross(relative, following), 0).sum(dim=1).abs() * 0.5
    return torch.where(valid & (count >= 3), area, 0)


def _rotated_overlaps_torch(boxes, query_boxes, trans=False):
    if boxes.ndim != 3 or query_boxes.ndim != 3:
        raise ValueError("boxes and query_boxes must both be rank 3")
    if boxes.shape[0] != query_boxes.shape[0]:
        raise ValueError("boxes and query_boxes batch dimensions must match")
    first_fields = _normalise_boxes(boxes, trans)
    second_fields = _normalise_boxes(query_boxes, trans)
    first = _corners(first_fields)
    second = _corners(second_fields)
    batch, num_boxes, num_queries = boxes.shape[0], boxes.shape[2], query_boxes.shape[2]
    result = torch.empty(
        (batch, num_boxes, num_queries), dtype=boxes.dtype, device=boxes.device
    )
    # Bound temporary geometry storage even for multi-million-pair cases.
    flat = result.reshape(-1)
    for begin in range(0, flat.numel(), 4096):
        indices = torch.arange(
            begin, min(begin + 4096, flat.numel()), device=boxes.device
        )
        batches = indices // (num_boxes * num_queries)
        box_indices = (indices // num_queries) % num_boxes
        query_indices = indices % num_queries
        valid = (
            first_fields[-1][batches, box_indices]
            & second_fields[-1][batches, query_indices]
        )
        flat[begin : begin + indices.numel()] = _clipped_pair_area(
            first[batches, box_indices], second[batches, query_indices], valid
        )
    return result


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
    for angle in (
        1.0e20,
        -1.0e20,
        torch.finfo(torch.float32).max,
        -torch.finfo(torch.float32).max,
        360.0,
        -360.0,
        -1.0e-20,
    ):
        for trans in (False, True):
            values = (-2, -1, 2, 1, angle) if trans else (0, 0, 4, 2, angle)
            query = (-1, -1, 3, 1, 0) if trans else (1, 0, 4, 2, 0)
            # All external inputs are float32; float64 is reference arithmetic.
            first = torch.tensor(values, dtype=torch.float32).reshape(1, 5, 1)
            second = torch.tensor(query, dtype=torch.float32).reshape(1, 5, 1)
            reduced = first.clone()
            reduced[:, 4] = torch.fmod(reduced[:, 4], 360.0)
            for dtype in (torch.float32, torch.float64):
                torch.testing.assert_close(
                    _rotated_overlaps_torch(first.to(dtype), second.to(dtype), trans),
                    _rotated_overlaps_torch(reduced.to(dtype), second.to(dtype), trans),
                    rtol=0,
                    atol=0,
                )
    cases = (
        # The former near-parallel edge-pair division returned 84.91 here,
        # exceeding the contained square's area (30.25).
        ((8.0, 8.0, 5.5, 74.78156, 135.0), (-8.0, -8.0, 5.5, 5.5, -135.0), 30.25),
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
    for angle in (0.0, 45.0, 90.0, 135.0, -135.0, 179.999):
        first = torch.tensor(
            (8.0, 8.0, 100.0, 100.0, angle), dtype=torch.float64
        ).reshape(1, 5, 1)
        second = torch.tensor(
            (8.0, 8.0, 5.5, 5.5, angle + 90), dtype=torch.float64
        ).reshape(1, 5, 1)
        for a, b in ((first, second), (second, first)):
            torch.testing.assert_close(
                _rotated_overlaps_torch(a, b),
                torch.full((1, 1, 1), 30.25, dtype=torch.float64),
                rtol=1.0e-12,
                atol=1.0e-12,
            )

    # A narrow xyxyt intersection from L1_019, independently checked with
    # polygon geometry. Start with the actual float32 endpoint values.
    first = (
        torch.tensor(
            (-0.2423699647, -0.2500007749, 0.2603048384, 0.2549992204, 1.2994835377),
            dtype=torch.float32,
        )
        .reshape(1, 5, 1)
        .to(torch.float64)
    )
    second = (
        torch.tensor(
            (-21.5855388641, -2.9623384476, 5.5855388641, 18.9623374939, -135.0),
            dtype=torch.float32,
        )
        .reshape(1, 5, 1)
        .to(torch.float64)
    )
    for a, b in ((first, second), (second, first)):
        torch.testing.assert_close(
            float(_rotated_overlaps_torch(a, b, trans=True)[0, 0, 0]),
            6.253878989896318e-8,
            rtol=1.0e-8,
            atol=1.0e-15,
        )

    for dtype in (torch.float32, torch.float64):
        large = torch.tensor((0, 0, 1.7e38, 1.7e38, 135), dtype=dtype).reshape(1, 5, 1)
        small = torch.tensor((2, -2, 2, 3, 15), dtype=dtype).reshape(1, 5, 1)
        for a, b in ((large, small), (small, large)):
            torch.testing.assert_close(
                float(_rotated_overlaps_torch(a, b)[0, 0, 0]),
                6.0,
                rtol=1.0e-6,
                atol=1.0e-7,
            )


if __name__ == "__main__":
    _self_test()
