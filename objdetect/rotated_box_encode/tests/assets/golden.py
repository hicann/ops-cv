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

import numpy as np
import torch

__spec__ = {"rotated_box_encode": "RotatedBoxEncodeTestSpec"}


def _compute(a, g, weight):
    ax0, ay0, ax1, ay1, aang = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    gx0, gy0, gx1, gy1, gang = g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4]

    wa = torch.maximum(ax1 - ax0, torch.ones_like(ax1))
    ha = torch.maximum(ay1 - ay0, torch.ones_like(ay1))
    cxa = ax0 + wa * 0.5
    cya = ay0 + ha * 0.5

    wg = torch.maximum(gx1 - gx0, torch.ones_like(gx1))
    hg = torch.maximum(gy1 - gy0, torch.ones_like(gy1))
    cxg = gx0 + wg * 0.5
    cyg = gy0 + hg * 0.5

    wx, wy, ww, wh, wa_w = weight

    dx = (cxg - cxa) / wa * wx
    dy = (cyg - cya) / ha * wy
    dw = (torch.log(wg) - torch.log(wa)) * ww
    dh = (torch.log(hg) - torch.log(ha)) * wh

    aang_rad = aang * (np.pi / 180.0)
    gang_rad = gang * (np.pi / 180.0)
    dtheta = (torch.tan(gang_rad) - torch.tan(aang_rad)) * wa_w

    return torch.stack([dx, dy, dw, dh, dtheta], dim=1)


class RotatedBoxEncodeTestSpec:
    """rotated_box_encode 算子测试规范

    将 anchor_box 与 gt_box 之间的几何偏差编码为 5 通道回归 delta (dx, dy, dw, dh, dθ)。
    输入输出均为 3D 张量 (B, 5, N)，5 通道按 corner form (x0, y0, x1, y1, θ_deg) 组织。
    golden 和 third_party 均用 torch 算子拼接实现，供 cross_check 交叉比对。
    """

    def golden(anchor_box, gt_box, *, weight=(1.0, 1.0, 1.0, 1.0, 1.0), **kwargs):
        x_dtype = anchor_box.dtype
        if anchor_box.size == 0:
            return [np.empty_like(anchor_box)]

        B, C, N = anchor_box.shape
        a = torch.from_numpy(
            anchor_box.astype(np.float32).permute(0, 2, 1).reshape(B * N, C)
        )
        g = torch.from_numpy(
            gt_box.astype(np.float32).permute(0, 2, 1).reshape(B * N, C)
        )
        w = [float(v) for v in weight]

        out = _compute(a, g, w)
        out = out.reshape(B, N, C).permute(0, 2, 1)
        return [out.numpy().astype(x_dtype)]

    class TorchImpl:
        """torch 算子拼接实现：用 torch.maximum + torch.log + torch.tan 拼接编码公式。"""

        def __init__(self, *, weight=(1.0, 1.0, 1.0, 1.0, 1.0), **kwargs):
            self.weight = [float(v) for v in weight]

        def __call__(self, anchor_box, gt_box, **kwargs):
            x_dtype = anchor_box.dtype
            B, C, N = anchor_box.shape
            a = anchor_box.to(torch.float32).permute(0, 2, 1).reshape(B * N, C)
            g = gt_box.to(torch.float32).permute(0, 2, 1).reshape(B * N, C)

            out = _compute(a, g, self.weight)
            out = out.reshape(B, N, C).permute(0, 2, 1)
            return [out.to(x_dtype)]

    third_party = {
        "torch": TorchImpl,
    }

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
    }
