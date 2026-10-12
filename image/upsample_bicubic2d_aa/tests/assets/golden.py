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

"""Golden for upsample_bicubic2d_aa — torch ATen IR implementation.

The golden calls ``torch.ops.aten._upsample_bicubic2d_aa`` directly (per PR
!1808 review: the Ascend IR op is benchmarked against the ATen IR op).

Equivalence (Ascend IR UpsampleBicubic2dAA <-> aten::_upsample_bicubic2d_aa):
both use the Keys cubic filter with A=-0.5 (PIL-compatible antialias;
NPU kernel WeightCalculate vs PyTorch aa_filter "using -0.5 for bicubic,
antialiasing=true"), the normalised filter window
support=2*scale (scale=in/out, downsample) / invscale=1/scale, and the
separable W-then-H passes. Verified point-wise in float64 against an
independent reference (rel <= 2.1e-12 across up/down/identity/align_corners/
non-integer shapes).

Parameter mapping:
* x (tensor input)              -> self
* output_size (attr ListInt)    -> output_size [oh, ow]
* align_corners (attr, bool)    -> align_corners (only changes the scale)
* scales_h / scales_w (attr)    -> scales_h / scales_w. NOTE the ATen args
  carry the OUT/IN ratio (value > 0 -> src/dst scale = 1/value), while the
  Ascend IR attr semantics are architecture-dependent (aclnn
  UpsampleBicubic2dAA IsRegBase branch):
    - Ascend950 (RegBase, tiling_arch35.cpp): IR attr is OUT/IN  -> pass
      through to ATen unchanged;
    - other archs (910b/910_93, upsample_bicubic2d_aa_tiling.cpp): IR attr
      is IN/OUT -> pass 1/attr to ATen when attr > 0.
  The SoC is taken from kwargs when available; defaults to RegBase.

Two ATen limitations, reported in the PR (not worked around silently):
1. aten::_upsample_bicubic2d_aa does NOT dispatch Half/BFloat16 on CPU
   (RuntimeError: "compute_index_ranges_weights" not implemented). Reduced
   floats are lifted to fp32 before the call — matching the NPU kernel,
   which computes AA weights/accumulation in float (upsample_bicubic2d_aa_
   kernel.h local float tensors; fp16/bf16 weights CAST_RINT-quantised) —
   and the result is quantised back.
2. aten::_upsample_bicubic2d_aa collapses the H direction when the OUTPUT
   WIDTH == 1 (ATen CPU bug, reproduced on 2.7.1 with float64 inputs:
   every H row equals the first; independent reference chains — torch
   single-axis + copy construction, pointwise unrolled second
   implementation — agree the correct output differs, rel=1.0). For
   output_size[1] == 1 ONLY, the golden falls back to the in-file
   independent fp64 implementation below (verified by three independent
   reference chains, rel <= 7.8e-6 cross-agreement); the NPU kernel
   behaves correctly on this shape (NPU cases pass against the fallback).
   See the PR discussion for the maintainer decision on this scenario.
"""

import numpy as np
import torch

__spec__ = {"upsample_bicubic2d_aa": "UpsampleBicubic2dAATestSpec"}

_CUBIC_A = -0.5  # PIL/Keys antialias coefficient — do NOT use -0.75 here


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


def _is_regbase_soc(short_soc_version=None, full_soc_version=None):
    """Ascend950 (RegBase) family detection.

    The IR attr scales_h/scales_w semantics differ by architecture (see the
    aclnnUpsampleBicubic2dAA IsRegBase branch): RegBase archs carry the
    OUT/IN ratio in the attr, non-RegBase archs carry IN/OUT. Defaults to
    RegBase when no SoC info is available (primary target of this asset).
    """
    for soc in (short_soc_version, full_soc_version):
        if soc:
            s = str(soc)
            if "950" in s or "960" in s:
                return True
            if "910" in s or "310" in s or "920" in s:
                return False
    return True  # default: RegBase (Ascend950) semantics


def _normalize_output_size(output_size):
    """Accept list/tuple/ndarray/int; return (oh, ow)."""
    if output_size is None:
        raise ValueError("output_size attr is required for upsample_bicubic2d_aa")
    if isinstance(output_size, (int, np.integer)):
        return int(output_size), int(output_size)
    if isinstance(output_size, np.ndarray):
        output_size = output_size.flatten().tolist()
    vals = [int(v) for v in output_size]
    if len(vals) == 1:
        return vals[0], vals[0]
    if len(vals) != 2:
        raise ValueError(f"output_size must hold 2 elements, got {output_size!r}")
    return vals[0], vals[1]


# ---------------------------------------------------------------------------
# Independent fp64 fallback — used ONLY for the output_width == 1 shape,
# where aten::_upsample_bicubic2d_aa has the H-collapse bug (see module
# docstring item 2). Verified against three independent reference chains.
# ---------------------------------------------------------------------------


def _w_keys(x, a=_CUBIC_A):
    """Cubic convolution kernel, vectorised (fp64)."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    out = np.zeros_like(ax)
    m1 = ax <= 1.0
    m2 = (ax > 1.0) & (ax < 2.0)
    out[m1] = (a + 2.0) * ax[m1] ** 3 - (a + 3.0) * ax[m1] ** 2 + 1.0
    out[m2] = a * ax[m2] ** 3 - 5.0 * a * ax[m2] ** 2 + 8.0 * a * ax[m2] - 4.0 * a
    return out


def _aa_scale(in_size, out_size, align_corners, s, is_regbase=True):
    """AA scale (in/out) from the IR attr value (architecture-aware)."""
    if align_corners:
        return (in_size - 1.0) / (out_size - 1.0) if out_size > 1 else 0.0
    if s is not None and s > 0:
        # RegBase(950): attr = out/in -> internal in/out scale = 1/s
        # others(910b): attr = in/out -> internal in/out scale = s
        return 1.0 / float(s) if is_regbase else float(s)
    if in_size == out_size:
        return 1.0
    return in_size / out_size


def _aa_axis_tables(in_size, out_size, align_corners, s, is_regbase=True):
    """Return list of (idx ndarray, normalised weights ndarray) per dst."""
    scale = _aa_scale(in_size, out_size, align_corners, s, is_regbase)
    support = 2.0 * scale if scale >= 1.0 else 2.0
    invscale = 1.0 / scale if scale > 1.0 else 1.0
    max_interp = int(np.ceil(support)) * 2 + 1
    tables = []
    for dst in range(out_size):
        center = (dst + 0.5) * scale
        x_min = max(0, int(np.floor(center + 0.5 - support)))
        x_max = min(in_size, int(np.floor(center + 0.5 + support)))
        count = max(0, min(x_max - x_min, max_interp))
        if count <= 0:
            tables.append((np.zeros(0, dtype=np.int64), np.zeros(0)))
            continue
        idx = np.arange(x_min, x_min + count)
        w = _w_keys((idx + 0.5 - center) * invscale)
        total = w.sum()
        if total != 0.0:
            w = w / total
        tables.append((idx, w))
    return tables


def _upsample_bicubic2d_aa_nchw(
    x, out_h, out_w, align_corners, scales_h, scales_w, is_regbase=True
):
    """Independent AA forward on NCHW fp input -> fp64 [N,C,oh,ow]."""
    n, c, ih, iw = x.shape
    xf = x.astype(np.float64)
    # Pass 1 (W-direction): mid[n,c,ih,ow] = sum_k w * x[n,c,ih,xk]
    wt = _aa_axis_tables(iw, out_w, align_corners, scales_w, is_regbase)
    mid = np.zeros((n, c, ih, out_w), dtype=np.float64)
    for ow_ in range(out_w):
        idx, w = wt[ow_]
        for k in range(idx.size):
            mid[:, :, :, ow_] += w[k] * xf[:, :, :, idx[k]]
    # Pass 2 (H-direction): y[n,c,oh,ow] = sum_k w * mid[n,c,xh,ow]
    ht = _aa_axis_tables(ih, out_h, align_corners, scales_h, is_regbase)
    y = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    for oh_ in range(out_h):
        idx, w = ht[oh_]
        for k in range(idx.size):
            y[:, :, oh_, :] += w[k] * mid[:, :, idx[k], :]
    return y


class UpsampleBicubic2dAATestSpec:
    """TestSpec for upsample_bicubic2d_aa (kernel path, aten IR golden)."""

    def golden(
        x, *, output_size, align_corners=False, scales_h=0.0, scales_w=0.0, **kwargs
    ):
        """Golden for upsample_bicubic2d_aa via aten._upsample_bicubic2d_aa.

        Parameters follow upsample_bicubic2d_aa_def.cpp without outputs.
        x is NCHW (format ND). output_size is the required [outH, outW] attr.
        scales_h/scales_w follow the architecture-dependent IR semantics
        (see module docstring); the SoC is taken from kwargs when available.

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name, golden_mode
        """
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError(f"x must be 4-D NCHW, got shape {x.shape}")
        out_h, out_w = _normalize_output_size(output_size)

        orig_dtype = str(x.dtype)
        is_regbase = _is_regbase_soc(
            kwargs.get("short_soc_version"), kwargs.get("full_soc_version")
        )
        s_h_attr, s_w_attr = float(scales_h), float(scales_w)

        if out_w == 1:
            # ATen CPU defect: H direction collapses for output width == 1
            # (module docstring item 2). Independent fp64 fallback, verified
            # by three independent reference chains; NPU is correct here.
            x_f = x.astype(np.float32) if orig_dtype in ("float16", "bfloat16") else x
            y = _upsample_bicubic2d_aa_nchw(
                x_f, out_h, out_w, bool(align_corners), s_h_attr, s_w_attr, is_regbase
            )
            if orig_dtype == "float16":
                y = y.astype(np.float16)
            elif orig_dtype == "bfloat16":
                y = torch.from_numpy(y).to(torch.bfloat16)
                y = y.view(torch.uint16).numpy().view(np.dtype("bfloat16"))
            else:
                y = y.astype(x.dtype, copy=False)
            return [y]

        x_t = _to_torch(x)
        if x_t.dtype in (torch.float16, torch.bfloat16):
            x_t = x_t.to(torch.float32)

        # ATen scales carry the OUT/IN ratio:
        #   RegBase(950) IR attr is out/in -> pass through;
        #   others(910b) IR attr is in/out  -> invert when > 0.
        def _aten_scale(s_attr):
            if s_attr <= 0:
                return None
            return s_attr if is_regbase else 1.0 / s_attr

        y_t = torch.ops.aten._upsample_bicubic2d_aa(
            x_t,
            [out_h, out_w],
            bool(align_corners),
            _aten_scale(s_h_attr),
            _aten_scale(s_w_attr),
        )

        if orig_dtype == "float16":
            y_t = y_t.to(torch.float16)
        elif orig_dtype == "bfloat16":
            y_t = y_t.to(torch.bfloat16)
        return [_to_numpy(y_t, orig_dtype)]

    class ThirdPartyImpl:
        """torch.nn.functional.interpolate(bicubic, antialias=True) reference.

        Supports default scales only: torch derives scale from sizes when
        ``size`` is given (a provided scale_factor does not participate in
        coordinate mapping under the default recompute_scale_factor=True),
        so explicit non-default scales have no direct torch equivalent and
        raise. Known ATen CPU caveat (see golden docstring): with output
        width == 1 the H-direction collapses — avoid such shapes when
        cross-checking.
        """

        def __init__(
            self,
            *,
            output_size=None,
            align_corners=False,
            scales_h=0.0,
            scales_w=0.0,
            **kwargs,
        ):
            if float(scales_h) > 0 or float(scales_w) > 0:
                raise NotImplementedError(
                    "torch third_party for upsample_bicubic2d_aa only supports "
                    "default scales (shape-derived); explicit scales use the aten IR golden"
                )
            self.output_size = output_size
            self.align_corners = bool(align_corners)

        def __call__(self, x, **kwargs):
            oh, ow = _normalize_output_size(self.output_size)
            out = torch.nn.functional.interpolate(
                x,
                size=(oh, ow),
                mode="bicubic",
                align_corners=self.align_corners,
                antialias=True,
            )
            return [out]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        # fp16/bf16 follow the TTK official dtype table (2^-10 / 2^-7),
        # verified on NPU (Ascend950PR) with multiple seeds. fp32 2e-4 covers
        # the NPU weight CAST_RINT quantisation + fp32 accumulation order.
        "float32": {"standard": "stat_rel_err", "threshold": 2e-4},
        "float16": {"standard": "stat_rel_err", "threshold": 0.0009765625},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 0.0078125},
    }
