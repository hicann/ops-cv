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

"""Golden for resize_bicubic_v2_grad — torch ATen IR implementation.

The golden calls ``torch.ops.aten.upsample_bicubic2d_backward`` directly (per
PR !1808 review: the Ascend IR op is benchmarked against the ATen IR op).

Equivalence (Ascend IR ResizeBicubicV2Grad <-> aten::upsample_bicubic2d_
backward), verified point-wise in float64 against an independent
high-precision reference AND torch autograd (rel <= 2.9e-14 across half_pixel
/ align_corners / lenDst==1 / explicit-scales / identity shapes; adjoint
identity <grad, x> == <g, forward(x)> at 1e-16):

Parameter mapping:
* grads (tensor input, forward-output shape)   -> grad_output
* original_image (tensor input, forward-input  -> only its shape is used:
  shape)                                          input_size (int[4]); the
                                                  Ascend infershape defines
                                                  y.shape == original_image
                                                  .shape and y.dtype ==
                                                  grads.dtype, matching the
                                                  ATen backward output.
* align_corners (attr, bool)                   -> align_corners
* scales (attr ListFloat, OUT/IN ratio)        -> scales_h / scales_w with
  the same OUT/IN semantics as the forward op (ATen compute_scales_value:
  value > 0 -> src/dst scale = 1/value). Non-positive -> None.
* align_corners=True and lenDst==1: the Ascend grad tiling sets scale=0
  while ATen area_pixel_compute_scale also returns 0 — identical; the
  forward-side in/out variant degenerates the same (dst=0 -> src=0).
* Layout: ATen is NCHW; NHWC inputs are transposed around the aten call.
* dtype fp16/bf16: lifted to fp32 before the call (the NPU grad kernel
  computes in float with asc_atomic_add on float accumulators; the ATen
  backward uses an opmath float accumulation buffer for reduced floats —
  both are fp32-inner, verified against the fp64 reference); result
  quantised back to the grads dtype.
"""

import numpy as np
import torch

__spec__ = {"resize_bicubic_v2_grad": "ResizeBicubicV2GradTestSpec"}


def _to_torch(x):
    """numpy -> torch tensor; ml_dtypes bfloat16 via bit-exact uint16 view."""
    if str(x.dtype) == "bfloat16":
        return torch.from_numpy(np.ascontiguousarray(x).view(np.uint16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(np.ascontiguousarray(x))


def _to_numpy(t, orig_dtype):
    """torch -> numpy with dtype restore (bfloat16 via uint16 view)."""
    if orig_dtype == "bfloat16":
        from ml_dtypes import bfloat16 as np_bf16

        return t.contiguous().view(torch.uint16).cpu().numpy().view(np_bf16)
    return t.cpu().numpy()


def _is_nhwc(input_formats):
    fmts = input_formats or []
    if isinstance(fmts, str):
        x_format = fmts
    else:
        try:
            x_format = fmts[0]
        except (TypeError, IndexError, KeyError):
            return False
    return str(x_format).strip().upper() == "NHWC"


def _normalize_scales(scales):
    """Accept list/tuple/ndarray/None; return per-axis floats (default 0.0)."""
    if scales is None:
        return (0.0, 0.0)
    if isinstance(scales, np.ndarray):
        scales = scales.flatten().tolist()
    if isinstance(scales, (int, float)):
        return (float(scales), float(scales))
    vals = [float(v) for v in scales]
    if len(vals) == 1:
        return (vals[0], vals[0])
    return (vals[0], vals[1])


class ResizeBicubicV2GradTestSpec:
    """TestSpec for resize_bicubic_v2_grad (kernel path, aten IR golden)."""

    def golden(
        grads, original_image, *, align_corners=False, scales=(0.0, 0.0), **kwargs
    ):
        """Golden for resize_bicubic_v2_grad via aten.upsample_bicubic2d_backward.

        Parameters follow resize_bicubic_v2_grad_def.cpp without outputs.
        y has the shape of original_image and the dtype of grads.

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name, golden_mode
        """
        grads = np.asarray(grads)
        original_image = np.asarray(original_image)
        if grads.ndim != 4 or original_image.ndim != 4:
            raise ValueError(
                f"grads/original_image must be 4-D, got {grads.shape} / {original_image.shape}"
            )

        orig_dtype = str(grads.dtype)
        g_t = _to_torch(grads)
        o_t = _to_torch(original_image)
        if g_t.dtype in (torch.float16, torch.bfloat16):
            g_t = g_t.to(torch.float32)
            o_t = o_t.to(torch.float32)

        nhwc = _is_nhwc(kwargs.get("input_formats"))
        if nhwc:
            g_t = g_t.permute(0, 3, 1, 2).contiguous()
            o_t = o_t.permute(0, 3, 1, 2).contiguous()
        input_size = list(o_t.shape)

        scales_pair = _normalize_scales(scales)
        s_h = scales_pair[0] if scales_pair[0] > 0 else None
        s_w = scales_pair[1] if scales_pair[1] > 0 else None

        y_t = torch.ops.aten.upsample_bicubic2d_backward(
            g_t, list(g_t.shape[2:]), input_size, bool(align_corners), s_h, s_w
        )

        if nhwc:
            y_t = y_t.permute(0, 2, 3, 1).contiguous()
        if orig_dtype == "float16":
            y_t = y_t.to(torch.float16)
        elif orig_dtype == "bfloat16":
            y_t = y_t.to(torch.bfloat16)
        return [_to_numpy(y_t, orig_dtype)]

    class ThirdPartyImpl:
        """torch autograd reference: interpolate(bicubic).backward(grads).

        Only supports default scales (torch derives scale from sizes when
        ``size`` is given). Explicit non-zero scales have no torch equivalent.
        """

        def __init__(self, *, align_corners=False, scales=(0.0, 0.0), **kwargs):
            scales_pair = _normalize_scales(scales)
            if scales_pair[0] > 0 or scales_pair[1] > 0:
                raise NotImplementedError(
                    "torch third_party for resize_bicubic_v2_grad only supports "
                    "default scales (shape-derived); explicit scales use the aten IR golden"
                )
            self.align_corners = bool(align_corners)

        def __call__(self, grads, original_image, **kwargs):
            x = original_image.to(torch.float32).detach().clone().requires_grad_(True)
            out = torch.nn.functional.interpolate(
                x,
                size=tuple(grads.shape[2:]),
                mode="bicubic",
                align_corners=self.align_corners,
            )
            out.backward(grads.to(torch.float32))
            return [x.grad.to(original_image.dtype)]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        # fp16/bf16 follow the TTK official dtype table (2^-10 / 2^-7),
        # verified on NPU (Ascend950PR) with multiple seeds.
        "float32": {"standard": "stat_rel_err", "threshold": 1e-4},
        "float16": {"standard": "stat_rel_err", "threshold": 0.0009765625},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 0.0078125},
    }
