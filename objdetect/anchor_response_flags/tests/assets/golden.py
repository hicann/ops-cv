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

"""anchor_response_flags 在 kernel/geir 测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，转 torch tensor 后用 torch 小算子拼接计算，结果转回 numpy；
third_party 的 torch vendor 收到 torch.Tensor（框架按 vendor 自动转换），直接用小算子拼接，无需手动转换。
无现成 torch 接口，以 torch 小算子拼接实现 golden 与三方标杆。
"""

__spec__ = {"anchor_response_flags": "AnchorResponseFlagsKernelSpec"}

import torch


class AnchorResponseFlagsKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 torch.Tensor"""

    def golden(gt_bboxes, *, featmap_size, strides, num_base_anchors, **kwargs):
        """
        Golden function for anchor_response_flags.
        All the parameters (names and order) follow @anchor_response_flags_def.cpp without outputs.
        All the input Tensors are numpy.ndarray.

        Args:
            gt_bboxes: numpy array, shape (N, 4), dtype float16/float32
                       Ground truth bounding box coordinates in [x1, y1, x2, y2] format.
            featmap_size: list of 2 ints, [feat_h, feat_w]
            strides: list of 2 ints, [stride_h, stride_w]
            num_base_anchors: int, number of base anchors per grid position
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            List containing one output tensor, shape (feat_h * feat_w * num_base_anchors,), dtype uint8
        """
        feat_h, feat_w = featmap_size
        stride_h, stride_w = strides

        if stride_h == 0 or stride_w == 0:
            raise ZeroDivisionError("strides must not contain 0")

        N = gt_bboxes.shape[0]
        total_anchors = feat_h * feat_w * num_base_anchors

        if N == 0:
            return [torch.zeros(total_anchors, dtype=torch.uint8)]

        # Cast to float32 for computation (avoid fp16 precision issues)
        gt = torch.from_numpy(gt_bboxes).to(torch.float32)

        cx = (gt[:, 0] + gt[:, 2]) * 0.5
        cy = (gt[:, 1] + gt[:, 3]) * 0.5

        cx = torch.nan_to_num(cx, nan=0.0, posinf=0.0, neginf=0.0)
        cy = torch.nan_to_num(cy, nan=0.0, posinf=0.0, neginf=0.0)

        raw_gx = torch.floor(cx / float(stride_h))
        raw_gy = torch.floor(cy / float(stride_w))

        grid_x = torch.clamp(raw_gx, min=0.0, max=float(feat_w - 1)).to(torch.int64)
        grid_y = torch.clamp(raw_gy, min=0.0, max=float(feat_h - 1)).to(torch.int64)

        grid_idx = grid_y * feat_w + grid_x

        responsible_grid = torch.zeros(feat_h * feat_w, dtype=torch.uint8)
        responsible_grid[grid_idx] = 1

        flags = torch.repeat_interleave(responsible_grid, num_base_anchors)

        return [flags.numpy()]

    class ThirdPartyImpl:
        """torch vendor 三方标杆 — 入参 torch.Tensor，用小算子拼接实现"""

        def __init__(
            self, gt_bboxes, *, featmap_size, strides, num_base_anchors, **kwargs
        ):
            self.feat_h, self.feat_w = featmap_size
            self.stride_h, self.stride_w = strides
            self.num_base_anchors = num_base_anchors
            self.total_anchors = self.feat_h * self.feat_w * self.num_base_anchors
            self.N = gt_bboxes.shape[0]
            if self.stride_h == 0 or self.stride_w == 0:
                raise ZeroDivisionError("strides must not contain 0")

        def __call__(self, gt_bboxes, **kwargs):
            if self.N == 0:
                return [torch.zeros(self.total_anchors, dtype=torch.uint8)]

            gt = gt_bboxes.to(torch.float32)

            cx = (gt[:, 0] + gt[:, 2]) * 0.5
            cy = (gt[:, 1] + gt[:, 3]) * 0.5

            cx = torch.nan_to_num(cx, nan=0.0, posinf=0.0, neginf=0.0)
            cy = torch.nan_to_num(cy, nan=0.0, posinf=0.0, neginf=0.0)

            raw_gx = torch.floor(cx / float(self.stride_h))
            raw_gy = torch.floor(cy / float(self.stride_w))

            grid_x = torch.clamp(raw_gx, min=0.0, max=float(self.feat_w - 1)).to(
                torch.int64
            )
            grid_y = torch.clamp(raw_gy, min=0.0, max=float(self.feat_h - 1)).to(
                torch.int64
            )

            grid_idx = grid_y * self.feat_w + grid_x

            responsible_grid = torch.zeros(self.feat_h * self.feat_w, dtype=torch.uint8)
            responsible_grid[grid_idx] = 1

            flags = torch.repeat_interleave(responsible_grid, self.num_base_anchors)

            return [flags]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {"uint8": {"standard": "binary_equal"}}
