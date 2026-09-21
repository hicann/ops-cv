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
"""DeformableRoiPoolGrad kernel/GEIR golden and MMCV CUDA reference.

Kernel/GEIR golden inputs are numpy.ndarray and are evaluated by an explicit
PyTorch CPU small-op composition. The third-party path uses MMCV's public
``deform_roi_pool`` CUDA API plus ``torch.autograd.grad``.

MMCV does not expose a public ``DeformableRoiPoolGrad`` Python API. Its public
forward operator's autograd backward dispatches to ``deform_roi_pool_backward``
and produces the two outputs required here: grad_x and grad_offset.
"""

import math
import torch


__spec__ = {
    "deformable_roi_pool_grad": "DeformableRoiPoolGradKernelSpec",
}


def _output_size_list(output_size):
    if isinstance(output_size, torch.Tensor):
        output_size = output_size.detach().cpu().tolist()
    elif isinstance(output_size, int):
        output_size = [output_size, output_size]
    else:
        output_size = list(output_size)
    if len(output_size) != 2:
        raise ValueError(f"output_size must contain two values, got {output_size}")
    result = [int(output_size[0]), int(output_size[1])]
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"output_size values must be positive, got {result}")
    return result


def _cpu_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    return torch.as_tensor(value).cpu().contiguous()


def _adaptive_grid_size(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value > 46340.0:
        return 0
    return math.ceil(value)


def _finite_or_zero(value):
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value))


def _deformable_roi_pool_grad_cpu(
    grad,
    x,
    rois,
    offset,
    output_size,
    spatial_scale=1.0,
    sampling_ratio=0,
    gamma=0.1,
):
    """Explicit CPU small-op composition aligned with MMCV CUDA semantics."""
    pooled_height, pooled_width = _output_size_list(output_size)
    grad_f = _cpu_tensor(grad).to(torch.float32)
    x_src = _cpu_tensor(x)
    x_f = x_src.to(torch.float32)
    rois_f = _cpu_tensor(rois).to(torch.float32)

    has_offset = offset is not None and _cpu_tensor(offset).numel() > 0
    offset_src = _cpu_tensor(offset) if has_offset else None
    offset_f = offset_src.to(torch.float32) if has_offset else None

    batch, channels, height, width = map(int, x_f.shape)
    num_rois = int(rois_f.shape[0])
    grad_x = torch.zeros((batch, channels, height, width), dtype=torch.float32)
    grad_offset = torch.zeros(
        (num_rois, 2, pooled_height, pooled_width), dtype=torch.float32
    )
    if batch <= 0:
        return grad_x, grad_offset

    spatial_scale_f = torch.tensor(float(spatial_scale), dtype=torch.float32)
    gamma_f = torch.tensor(float(gamma), dtype=torch.float32)
    half_f = torch.tensor(0.5, dtype=torch.float32)
    one_f = torch.tensor(1.0, dtype=torch.float32)

    for roi_idx in range(num_rois):
        roi_batch_value = float(rois_f[roi_idx, 0].item())
        roi_batch_idx = int(roi_batch_value) if math.isfinite(roi_batch_value) else 0
        roi_batch_idx = max(0, min(roi_batch_idx, batch - 1))

        roi_start_w = _finite_or_zero(rois_f[roi_idx, 1]) * spatial_scale_f - half_f
        roi_start_h = _finite_or_zero(rois_f[roi_idx, 2]) * spatial_scale_f - half_f
        roi_end_w = _finite_or_zero(rois_f[roi_idx, 3]) * spatial_scale_f - half_f
        roi_end_h = _finite_or_zero(rois_f[roi_idx, 4]) * spatial_scale_f - half_f
        roi_width = roi_end_w - roi_start_w
        roi_height = roi_end_h - roi_start_h
        bin_size_h = roi_height / float(pooled_height)
        bin_size_w = roi_width / float(pooled_width)

        if int(sampling_ratio) > 0:
            grid_h = int(sampling_ratio)
            grid_w = int(sampling_ratio)
        else:
            grid_h = _adaptive_grid_size(roi_height / float(pooled_height))
            grid_w = _adaptive_grid_size(roi_width / float(pooled_width))
        count = grid_h * grid_w
        if count <= 0:
            continue

        for channel_idx in range(channels):
            for pooled_h_idx in range(pooled_height):
                for pooled_w_idx in range(pooled_width):
                    current_start_w = roi_start_w
                    current_start_h = roi_start_h
                    if has_offset:
                        offset_w = offset_f[roi_idx, 0, pooled_h_idx, pooled_w_idx]
                        offset_h = offset_f[roi_idx, 1, pooled_h_idx, pooled_w_idx]
                        offset_w_finite = bool(torch.isfinite(offset_w).item())
                        offset_h_finite = bool(torch.isfinite(offset_h).item())
                        current_start_w = current_start_w + (
                            gamma_f * roi_width * _finite_or_zero(offset_w)
                        )
                        current_start_h = current_start_h + (
                            gamma_f * roi_height * _finite_or_zero(offset_h)
                        )

                    grad_bin = grad_f[
                        roi_idx, channel_idx, pooled_h_idx, pooled_w_idx
                    ] / float(count)

                    for sample_h_idx in range(grid_h):
                        raw_y = (
                            current_start_h
                            + float(pooled_h_idx) * bin_size_h
                            + (float(sample_h_idx) + 0.5) * bin_size_h / float(grid_h)
                        )
                        for sample_w_idx in range(grid_w):
                            raw_x = (
                                current_start_w
                                + float(pooled_w_idx) * bin_size_w
                                + (float(sample_w_idx) + 0.5)
                                * bin_size_w
                                / float(grid_w)
                            )

                            if (
                                raw_y < -1.0
                                or raw_y > float(height)
                                or raw_x < -1.0
                                or raw_x > float(width)
                            ):
                                continue

                            sample_y = torch.clamp_min(raw_y, 0.0)
                            sample_x = torch.clamp_min(raw_x, 0.0)
                            y_low = int(sample_y.item())
                            x_low = int(sample_x.item())

                            if y_low >= height - 1:
                                y_low = height - 1
                                y_high = height - 1
                                sample_y = torch.tensor(
                                    float(y_low), dtype=torch.float32
                                )
                            else:
                                y_high = y_low + 1

                            if x_low >= width - 1:
                                x_low = width - 1
                                x_high = width - 1
                                sample_x = torch.tensor(
                                    float(x_low), dtype=torch.float32
                                )
                            else:
                                x_high = x_low + 1

                            ly = sample_y - float(y_low)
                            lx = sample_x - float(x_low)
                            hy = one_f - ly
                            hx = one_f - lx
                            weights = (hy * hx, hy * lx, ly * hx, ly * lx)
                            positions = (
                                (y_low, x_low),
                                (y_low, x_high),
                                (y_high, x_low),
                                (y_high, x_high),
                            )
                            for weight, (y_idx, x_idx) in zip(weights, positions):
                                grad_x[roi_batch_idx, channel_idx, y_idx, x_idx] += (
                                    grad_bin * weight
                                )

                            if not has_offset:
                                continue

                            input_00 = x_f[roi_batch_idx, channel_idx, y_low, x_low]
                            input_10 = x_f[roi_batch_idx, channel_idx, y_low, x_high]
                            input_01 = x_f[roi_batch_idx, channel_idx, y_high, x_low]
                            input_11 = x_f[roi_batch_idx, channel_idx, y_high, x_high]

                            # MMCV clamps coordinates only for indices/weights.
                            # Offset derivatives intentionally use raw_x/raw_y.
                            grad_offset_x = (
                                gamma_f
                                * roi_width
                                * grad_bin
                                * (
                                    input_11 * (raw_y - float(y_low))
                                    + input_10 * (float(y_high) - raw_y)
                                    + input_01 * (float(y_low) - raw_y)
                                    + input_00 * (raw_y - float(y_high))
                                )
                            )
                            grad_offset_y = (
                                gamma_f
                                * roi_height
                                * grad_bin
                                * (
                                    input_11 * (raw_x - float(x_low))
                                    + input_01 * (float(x_high) - raw_x)
                                    + input_10 * (float(x_low) - raw_x)
                                    + input_00 * (raw_x - float(x_high))
                                )
                            )
                            if offset_w_finite:
                                grad_offset[roi_idx, 0, pooled_h_idx, pooled_w_idx] += (
                                    grad_offset_x
                                )
                            if offset_h_finite:
                                grad_offset[roi_idx, 1, pooled_h_idx, pooled_w_idx] += (
                                    grad_offset_y
                                )

    return grad_x.to(x_src.dtype), grad_offset.to(x_src.dtype)


def _kernel_cpu_golden(
    grad,
    x,
    rois,
    offset,
    *,
    output_size,
    spatial_scale=1.0,
    sampling_ratio=0,
    gamma=0.1,
    **kwargs,
):
    grad_x, grad_offset = _deformable_roi_pool_grad_cpu(
        grad,
        x,
        rois,
        offset,
        output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        gamma=gamma,
    )
    return [grad_x.numpy(), grad_offset.numpy()]


class MmcvCudaDeformableRoiPoolGrad:
    """MMCV CUDA reference through public forward API and PyTorch autograd."""

    def __init__(
        self,
        *,
        output_size=None,
        spatial_scale=1.0,
        sampling_ratio=0,
        gamma=0.1,
        **kwargs,
    ):
        self.output_size = (
            _output_size_list(output_size) if output_size is not None else None
        )
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.gamma = float(gamma)

    def __call__(
        self,
        grad,
        x,
        rois,
        offset,
        **kwargs,
    ):
        from mmcv.ops import deform_roi_pool

        if not isinstance(x, torch.Tensor) or not x.is_cuda:
            raise RuntimeError(
                "MMCV third-party golden requires torch.Tensor inputs on an NVIDIA CUDA device"
            )
        if x.dtype != torch.float32:
            raise TypeError(f"x must be torch.float32, got {x.dtype}")

        tensor_inputs = {"grad": grad, "rois": rois}
        if offset is not None and int(offset.numel()) > 0:
            tensor_inputs["offset"] = offset
        for name, value in tensor_inputs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.device != x.device or value.dtype != x.dtype:
                raise ValueError(f"{name} must already match x.device and x.dtype")

        output_size = self.output_size or [int(grad.shape[-2]), int(grad.shape[-1])]
        if int(rois.shape[0]) == 0:
            return [
                torch.zeros_like(x),
                torch.zeros(
                    (0, 2, int(output_size[0]), int(output_size[1])),
                    device=x.device,
                    dtype=x.dtype,
                ),
            ]
        with torch.enable_grad():
            x_ref = x.detach().contiguous().requires_grad_(True)
            rois_ref = rois.detach().contiguous()
            grad_ref = grad.detach().contiguous()

            has_offset = offset is not None and int(offset.numel()) > 0
            if has_offset:
                offset_ref = offset.detach().contiguous().requires_grad_(True)
            else:
                offset_ref = None

            output = deform_roi_pool(
                x_ref,
                rois_ref,
                offset_ref,
                tuple(output_size),
                self.spatial_scale,
                self.sampling_ratio,
                self.gamma,
            )

            if has_offset:
                result_grad_x, result_grad_offset = torch.autograd.grad(
                    output,
                    (x_ref, offset_ref),
                    grad_outputs=grad_ref,
                    create_graph=False,
                    retain_graph=False,
                )
            else:
                (result_grad_x,) = torch.autograd.grad(
                    output,
                    (x_ref,),
                    grad_outputs=grad_ref,
                    create_graph=False,
                    retain_graph=False,
                )
                result_grad_offset = torch.zeros(
                    (
                        int(rois.shape[0]),
                        2,
                        int(output_size[0]),
                        int(output_size[1]),
                    ),
                    device=x.device,
                    dtype=x.dtype,
                )

        return [result_grad_x.detach(), result_grad_offset.detach()]


class DeformableRoiPoolGradKernelSpec:
    """Kernel/GEIR: numpy CPU golden; MMCV receives CUDA torch tensors."""

    golden = _kernel_cpu_golden
    third_party = {"torch": MmcvCudaDeformableRoiPoolGrad}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }
