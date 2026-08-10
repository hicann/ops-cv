#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np


def _dtype(x):
    return str(x.dtype).split(".")[-1]


def _validate(dy, bboxes, gtboxes, is_cross, mode):
    if is_cross or mode != "iou":
        raise ValueError("GIoUGrad supports only is_cross=False and mode='iou'")
    if dy.ndim != 1:
        raise ValueError(f"dy must be 1D, got {tuple(dy.shape)}")
    if (
        bboxes.ndim != 2
        or bboxes.shape[0] != 4
        or gtboxes.ndim != 2
        or gtboxes.shape[0] != 4
    ):
        raise ValueError(
            f"bboxes and gtboxes must have shape (4, N), got {tuple(bboxes.shape)} and {tuple(gtboxes.shape)}"
        )
    if bboxes.shape[1] != dy.shape[0] or gtboxes.shape[1] != dy.shape[0]:
        raise ValueError(
            "is_cross=False requires dy, bboxes and gtboxes to have the same N"
        )
    if len({_dtype(dy), _dtype(bboxes), _dtype(gtboxes)}) != 1 or _dtype(dy) not in (
        "float16",
        "float32",
        "float64",
    ):
        raise TypeError(
            "dy, bboxes and gtboxes must share a supported or promoted floating dtype"
        )


def _to_xyxy(boxes, trans):
    if not trans:
        return boxes
    import torch

    xy, wh = boxes[:, :2], boxes[:, 2:]
    return torch.cat((xy - wh / 2, xy + wh / 2), dim=-1)


class GIoUGradSpec:
    def golden(
        dy, bboxes, gtboxes, *, trans=False, is_cross=False, mode="iou", **kwargs
    ):
        import torch
        from torchvision.ops import generalized_box_iou_loss

        _validate(dy, bboxes, gtboxes, is_cross, mode)
        if dy.size == 0:
            return [np.empty_like(bboxes), np.empty_like(gtboxes)]
        bboxes_t = torch.from_numpy(bboxes.T.copy()).requires_grad_(True)
        gtboxes_t = torch.from_numpy(gtboxes.T.copy()).requires_grad_(True)
        loss = generalized_box_iou_loss(
            _to_xyxy(bboxes_t, trans),
            _to_xyxy(gtboxes_t, trans),
            reduction="none",
            eps=1e-9,
        )
        loss.backward(-torch.from_numpy(dy.copy()))
        return [bboxes_t.grad.T.numpy(), gtboxes_t.grad.T.numpy()]

    tolerance = {
        dtype: {"standard": "stat_rel_err"} for dtype in ("float16", "float32")
    }


__spec__ = {"g_io_u_grad": "GIoUGradSpec"}
