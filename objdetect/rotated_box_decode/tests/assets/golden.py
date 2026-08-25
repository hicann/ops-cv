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
import math

import numpy
import torch


__spec__ = {
    "rotated_box_decode": "RotatedBoxDecodeTestSpec",
}

_DEFAULT_WEIGHT = (1.0, 1.0, 1.0, 1.0, 1.0)
_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


def _resolve_weight(weight_optional):
    if weight_optional is None:
        return [float(w) for w in _DEFAULT_WEIGHT]
    return [float(w) for w in weight_optional]


def _decode_core(anchor_t, deltas_t, weight_list):
    lx = anchor_t[:, 0, :]
    ly = anchor_t[:, 1, :]
    rx = anchor_t[:, 2, :]
    ry = anchor_t[:, 3, :]
    angle_a = anchor_t[:, 4, :]
    a_w = torch.clamp(rx - lx, min=1.0)
    a_h = torch.clamp(ry - ly, min=1.0)
    a_cx = lx + a_w / 2.0
    a_cy = ly + a_h / 2.0

    w = torch.tensor(weight_list, dtype=anchor_t.dtype, device=anchor_t.device).reshape(
        1, 5, 1
    )
    dp = deltas_t / w
    dpx = dp[:, 0, :]
    dpy = dp[:, 1, :]
    dpw = dp[:, 2, :]
    dph = dp[:, 3, :]
    dpt = dp[:, 4, :]

    t_cx = a_cx + dpx * a_w
    t_cy = a_cy + dpy * a_h

    t_w = torch.exp(dpw) * a_w
    t_h = torch.exp(dph) * a_h

    theta_a_rad = angle_a * _DEG2RAD
    theta_t_rad = torch.arctan(torch.tan(theta_a_rad) + dpt)
    theta_t = theta_t_rad * _RAD2DEG

    theta_t = torch.where(
        torch.abs(angle_a) > 1.0e6, torch.full_like(theta_t, float("nan")), theta_t
    )

    out_lx = t_cx - t_w / 2.0
    out_ly = t_cy - t_h / 2.0
    out_rx = t_cx + t_w / 2.0
    out_ry = t_cy + t_h / 2.0

    return torch.stack([out_lx, out_ly, out_rx, out_ry, theta_t], dim=1)


def _np_to_torch_fp32(np_arr):
    dtype_name = np_arr.dtype.name
    arr = numpy.ascontiguousarray(np_arr)
    if "bfloat16" in dtype_name:
        t = torch.from_numpy(arr.view(numpy.int16)).view(torch.bfloat16)
    else:
        t = torch.from_numpy(arr)
    return t.to(torch.float32)


def _torch_to_np(y_t, target_dtype_name):
    if "bfloat16" in target_dtype_name:
        from ml_dtypes import bfloat16 as np_bf16

        t_bf16 = y_t.to(torch.bfloat16).contiguous()
        np_int16 = t_bf16.view(torch.int16).numpy()
        return numpy.ascontiguousarray(np_int16).view(np_bf16)
    elif "float16" in target_dtype_name:
        return y_t.to(torch.float16).contiguous().numpy()
    else:
        return y_t.to(torch.float32).contiguous().numpy()


class RotatedBoxDecodeTestSpec:
    def golden(anchor_box, deltas, weightOptional=None, **kwargs):
        resolved = (
            weightOptional if weightOptional is not None else kwargs.get("weight")
        )
        weight = _resolve_weight(resolved)

        ori_dtype_name = anchor_box.dtype.name
        anchor_t = _np_to_torch_fp32(anchor_box)
        deltas_t = _np_to_torch_fp32(deltas)
        y_t = _decode_core(anchor_t, deltas_t, weight)
        return [_torch_to_np(y_t, ori_dtype_name)]

    class ThirdPartyImpl:
        def __init__(self, weightOptional=None, *, weight=None, **kwargs):
            resolved = weightOptional if weightOptional is not None else weight
            self.weight = _resolve_weight(resolved)

        def __call__(self, anchor_box, deltas, **kwargs):
            return [_decode_core(anchor_box, deltas, self.weight)]

    third_party = {
        "torch": ThirdPartyImpl,
    }

    tolerance = {
        "float32": {"standard": "stat_rel_err", "threshold": 1.0e-1, "floor": 1.0e-5},
        "float16": {"standard": "stat_rel_err", "threshold": 1.0e-1, "floor": 1.0e-3},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 1.0e-1, "floor": 4.0e-3},
    }
