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


class RotatedBoxDecodeTestSpec:
    def golden(anchor_box, deltas, weightOptional=None, **kwargs):
        resolved = (
            weightOptional if weightOptional is not None else kwargs.get("weight")
        )
        weight = list(_DEFAULT_WEIGHT) if resolved is None else list(resolved)

        anchor_np = numpy.ascontiguousarray(anchor_box)
        if "bfloat16" in anchor_np.dtype.name:
            anchor_t = torch.from_numpy(anchor_np.view(numpy.int16)).view(
                torch.bfloat16
            )
        else:
            anchor_t = torch.from_numpy(anchor_np)
        out_dtype = anchor_t.dtype
        if anchor_t.dtype in (torch.float16, torch.bfloat16):
            anchor_t = anchor_t.to(torch.float32)

        deltas_np = numpy.ascontiguousarray(deltas)
        if "bfloat16" in deltas_np.dtype.name:
            deltas_t = torch.from_numpy(deltas_np.view(numpy.int16)).view(
                torch.bfloat16
            )
        else:
            deltas_t = torch.from_numpy(deltas_np)
        if deltas_t.dtype in (torch.float16, torch.bfloat16):
            deltas_t = deltas_t.to(torch.float32)

        lx = anchor_t[:, 0, :]
        ly = anchor_t[:, 1, :]
        rx = anchor_t[:, 2, :]
        ry = anchor_t[:, 3, :]
        angle_a = anchor_t[:, 4, :]
        a_w = torch.clamp(rx - lx, min=1.0)
        a_h = torch.clamp(ry - ly, min=1.0)
        a_cx = lx + a_w / 2.0
        a_cy = ly + a_h / 2.0

        w = torch.tensor(weight, dtype=anchor_t.dtype, device=anchor_t.device).reshape(
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

        y_t = torch.stack([out_lx, out_ly, out_rx, out_ry, theta_t], dim=1)
        if out_dtype in (torch.float16, torch.bfloat16):
            y_t = y_t.to(out_dtype)
        return [y_t.contiguous().numpy()]

    class ThirdPartyImpl:
        def __init__(self, weightOptional=None, *, weight=None, **kwargs):
            resolved = weightOptional if weightOptional is not None else weight
            self.weight = list(_DEFAULT_WEIGHT) if resolved is None else list(resolved)

        def __call__(self, anchor_box, deltas, **kwargs):
            out_dtype = anchor_box.dtype
            anchor_t = (
                anchor_box.to(torch.float32)
                if anchor_box.dtype in (torch.float16, torch.bfloat16)
                else anchor_box
            )
            deltas_t = (
                deltas.to(torch.float32)
                if deltas.dtype in (torch.float16, torch.bfloat16)
                else deltas
            )

            lx = anchor_t[:, 0, :]
            ly = anchor_t[:, 1, :]
            rx = anchor_t[:, 2, :]
            ry = anchor_t[:, 3, :]
            angle_a = anchor_t[:, 4, :]
            a_w = torch.clamp(rx - lx, min=1.0)
            a_h = torch.clamp(ry - ly, min=1.0)
            a_cx = lx + a_w / 2.0
            a_cy = ly + a_h / 2.0

            w = torch.tensor(
                self.weight, dtype=anchor_t.dtype, device=anchor_t.device
            ).reshape(1, 5, 1)
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
                torch.abs(angle_a) > 1.0e6,
                torch.full_like(theta_t, float("nan")),
                theta_t,
            )

            out_lx = t_cx - t_w / 2.0
            out_ly = t_cy - t_h / 2.0
            out_rx = t_cx + t_w / 2.0
            out_ry = t_cy + t_h / 2.0

            y_t = torch.stack([out_lx, out_ly, out_rx, out_ry, theta_t], dim=1)
            if out_dtype in (torch.float16, torch.bfloat16):
                y_t = y_t.to(out_dtype)
            return [y_t]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
