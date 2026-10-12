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

"""Golden for resize_bicubic_v2 — torch ATen IR implementation.

The golden calls ``torch.ops.aten.upsample_bicubic2d`` directly (per PR !1808
review: the Ascend IR op is benchmarked against the ATen IR op).

Equivalence (Ascend IR ResizeBicubicV2 <-> aten::upsample_bicubic2d), verified
point-wise in float64 against an independent high-precision reference
(rel <= 4.2e-12 across half_pixel / align_corners / out==1 / explicit-scales /
mixed-scales / identity shapes):

Parameter mapping:
* x (tensor input)               -> self
* size (int32[2] tensor input)   -> output_size [oh, ow] (tensor value read)
* align_corners (attr, bool)     -> align_corners
* scales (attr ListFloat,        -> scales_h / scales_w. Both carry the same
  OUT/IN ratio: ATen computes the src/dst scale as 1/value when value > 0
  (compute_scales_value), matching the Ascend tiling rule (ComputeScale:
  scale > 0 -> 1/scale). Non-positive values are passed as None so ATen
  falls back to in_size/out_size, identical to the IR default {0.0, 0.0}.
* align_corners=True and out==1: ATen area_pixel_compute_scale returns 0
  while the Ascend forward tiling keeps in/out — mathematically identical
  because dst=0 -> src=0*scale=0 either way (verified).
* Layout: ATen is NCHW; NHWC inputs are transposed around the aten call.
* dtype fp16/bf16: lifted to fp32 before the call. The NPU kernel casts
  reduced floats to float for the inner math (static_cast<float> in
  resize_bicubic_v2_simt_base.h) and the aten fp32 path is verified against
  the fp64 reference; a DIRECT aten Half/BFloat16 call computes in
  scalar_t, which does NOT match that contract and is intentionally not
  used. The result is quantised back to the input dtype.
* golden_mode=Promote (cross_check) may deliver fp32->fp64 lifted inputs;
  aten::upsample_bicubic2d dispatches double natively, no handling needed.
"""

import numpy as np
import torch

__spec__ = {"resize_bicubic_v2": "ResizeBicubicV2TestSpec"}


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


class ResizeBicubicV2TestSpec:
    """TestSpec for resize_bicubic_v2 (kernel path, aten IR golden)."""

    def golden(x, size, *, align_corners=False, scales=(0.0, 0.0), **kwargs):
        """Golden for resize_bicubic_v2 via torch.ops.aten.upsample_bicubic2d.

        Parameters follow resize_bicubic_v2_def.cpp without outputs.
        All inputs are numpy.ndarray (size is the int32 [outH, outW] tensor).

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name, golden_mode
        """
        x = np.asarray(x)
        size_arr = np.asarray(size).flatten()
        if size_arr.size < 2:
            raise ValueError(f"size tensor must hold 2 elements, got {size_arr.size}")
        out_h, out_w = int(size_arr[0]), int(size_arr[1])

        orig_dtype = str(x.dtype)
        x_t = _to_torch(x)
        if x_t.dtype in (torch.float16, torch.bfloat16):
            x_t = x_t.to(torch.float32)

        nhwc = _is_nhwc(kwargs.get("input_formats"))
        if nhwc:
            if x_t.dim() != 4:
                raise ValueError(f"x must be 4-D, got shape {tuple(x_t.shape)}")
            x_t = x_t.permute(0, 3, 1, 2).contiguous()

        scales_pair = _normalize_scales(scales)
        s_h = scales_pair[0] if scales_pair[0] > 0 else None
        s_w = scales_pair[1] if scales_pair[1] > 0 else None

        y_t = torch.ops.aten.upsample_bicubic2d(
            x_t, [out_h, out_w], bool(align_corners), s_h, s_w
        )

        if nhwc:
            y_t = y_t.permute(0, 2, 3, 1).contiguous()
        if orig_dtype == "float16":
            y_t = y_t.to(torch.float16)
        elif orig_dtype == "bfloat16":
            y_t = y_t.to(torch.bfloat16)
        return [_to_numpy(y_t, orig_dtype)]

    class ThirdPartyImpl:
        """torch.nn.functional.interpolate(bicubic) reference.

        Only supports the default scales ({0,0} / shape-derived) semantics:
        torch derives scale from input/output sizes when ``size`` is given.
        Explicit non-zero scales have no direct torch equivalent and raise.
        """

        def __init__(self, *, align_corners=False, scales=(0.0, 0.0), **kwargs):
            scales_pair = _normalize_scales(scales)
            if scales_pair[0] > 0 or scales_pair[1] > 0:
                raise NotImplementedError(
                    "torch third_party for resize_bicubic_v2 only supports default "
                    "scales (shape-derived); explicit scales use the aten IR golden"
                )
            self.align_corners = bool(align_corners)

        def __call__(self, x, size, **kwargs):
            size_arr = np.asarray(size).flatten()
            out = torch.nn.functional.interpolate(
                x,
                size=(int(size_arr[0]), int(size_arr[1])),
                mode="bicubic",
                align_corners=self.align_corners,
            )
            return [out]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        # fp32 3e-4 (evidence, TTK@202675b + Ascend950PR, seed=7 repro):
        # stat_rel_err judges mere < th AND mare < 10*th. Down-sampling inputs
        # occasionally hit cancellation points (measured: |y|=4.5e-5 from
        # 16 terms with sum(|terms|)=2.77, i.e. 61404x cancellation); the NPU
        # fp32 ordered accumulation then differs from the golden by an
        # absolute amount well inside the deterministic fp32 rounding bound
        # 16*eps*sum(|terms|) (all 288 points inside, 0 systematic bias) —
        # but the tiny denominator inflates mare to 1.48e-3. Both 1e-4 and
        # the TTK default 2^-13 (1.22e-4, mare bound 1.22e-3) fail
        # intermittently (~6% of random seeds) on such points; mere stays
        # ~5e-6 so semantic regressions (typically >1%) remain detectable.
        # fp16/bf16 follow the TTK official dtype table (2^-10 / 2^-7),
        # verified on NPU with multiple seeds.
        "float32": {"standard": "stat_rel_err", "threshold": 3e-4},
        "float16": {"standard": "stat_rel_err", "threshold": 0.0009765625},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 0.0078125},
    }
