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

__golden__ = {"kernel": {"yolo": "yolo_golden"}}


def yolo_golden(
    x,
    *,
    boxes=3,
    coords=4,
    classes=80,
    yolo_version="V3",
    softmax=False,
    background=False,
    softmaxtree=False,
    **kwargs,
):
    """
    Golden function for yolo.
    All the parameters (names and order) follow SE doc prototype definition without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: numpy.ndarray, shape (N, boxes*(coords+1+classes), H, W), dtype float16/float32
        boxes: int, anchor box count
        coords: int, coordinate parameter count (fixed 4: x, y, w, h)
        classes: int, predicted class count
        yolo_version: str, "V2" or "V3"
        softmax: bool, whether to apply softmax (V2 only)
        background: bool, obj and classes operation type (V2 only)
        softmaxtree: bool, fixed False (unused)
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of (coord_data, obj_prob, classes_prob), each numpy.ndarray
    """
    x_dtype = x.dtype
    # @constraint: float16 input promoted to float32 for computation (SE 5.5)
    if x_dtype.name in ("bfloat16", "float16"):
        x = x.astype(np.float32)

    x_torch = torch.from_numpy(x)

    N, C, H, W = x_torch.shape
    HW = H * W
    B = boxes  # anchor box count

    # Promote to float32 for intermediate computation
    x_fp32 = x_torch.to(torch.float32)

    # Channel layout: [x0..xB, y0..yB, w0..wB, h0..hB, obj0..objB, cls0_0..cls0_B, ..., clsK_0..clsK_B]
    # Each group has B channels

    # Step 1: Coordinate processing
    # @constraint: x, y coordinates do sigmoid; w, h do move (SE 5.2)
    # @constraint: output coord_data channel order is (x, y, h, w) - w/h position swapped (SE 5.2)
    coord_x = torch.sigmoid(x_fp32[:, 0:B, :, :])  # (N, B, H, W)
    coord_y = torch.sigmoid(x_fp32[:, B : 2 * B, :, :])  # (N, B, H, W)
    coord_w = x_fp32[:, 2 * B : 3 * B, :, :]  # move (N, B, H, W)
    coord_h = x_fp32[:, 3 * B : 4 * B, :, :]  # move (N, B, H, W)

    # Output order: (x, y, h, w) - note h comes before w
    coord_data = torch.cat(
        [
            coord_x.reshape(N, B, HW),
            coord_y.reshape(N, B, HW),
            coord_h.reshape(N, B, HW),  # h before w
            coord_w.reshape(N, B, HW),  # w after h
        ],
        dim=1,
    )  # (N, 4*B, HW) = (N, boxes*coords, H*W)

    # Step 2: Determine yolo_mode
    # @constraint: 4 modes determined by yolo_version + softmax + background (SE 1.1)
    if yolo_version.upper() == "V3":
        yolo_mode = "YOLO_MODE_1"
    else:  # V2
        if not softmax and not background:
            yolo_mode = "YOLO_MODE_1"
        elif softmax and not background:
            yolo_mode = "YOLO_MODE_2"
        elif not softmax and background:
            yolo_mode = "YOLO_MODE_3"
        else:
            yolo_mode = "YOLO_MODE_4"

    # Step 3: Object confidence and class probability processing
    obj_raw = x_fp32[:, 4 * B : 5 * B, :, :]  # (N, B, H, W)
    cls_raw = x_fp32[:, 5 * B : 5 * B + classes * B, :, :]  # (N, K*B, H, W)
    # Reshape cls_raw to (N, K, B, H, W)
    cls_raw = cls_raw.reshape(N, classes, B, H, W)

    if yolo_mode == "YOLO_MODE_1":
        # @constraint: obj=sigmoid, classes=sigmoid (SE 1.1)
        obj_prob = torch.sigmoid(obj_raw).reshape(N, B * HW)
        classes_prob = torch.sigmoid(cls_raw).reshape(N, classes, B * HW)

    elif yolo_mode == "YOLO_MODE_2":
        # @constraint: obj=sigmoid, classes=softmax along K dimension (SE 1.1)
        obj_prob = torch.sigmoid(obj_raw).reshape(N, B * HW)
        classes_prob = torch.softmax(cls_raw, dim=1).reshape(N, classes, B * HW)

    elif yolo_mode == "YOLO_MODE_3":
        # @constraint: obj=move, classes=sigmoid (SE 1.1)
        obj_prob = obj_raw.reshape(N, B * HW)
        classes_prob = torch.sigmoid(cls_raw).reshape(N, classes, B * HW)

    else:  # YOLO_MODE_4
        # @constraint: obj+classes together softmax along K+1 dimension (SE 1.1)
        all_raw = torch.cat(
            [
                obj_raw.unsqueeze(1),  # (N, 1, B, H, W)
                cls_raw,  # (N, K, B, H, W)
            ],
            dim=1,
        )  # (N, K+1, B, H, W)
        all_softmax = torch.softmax(all_raw, dim=1)  # (N, K+1, B, H, W)
        obj_prob = all_softmax[:, 0, :, :, :].reshape(N, B * HW)
        classes_prob = all_softmax[:, 1:, :, :, :].reshape(N, classes, B * HW)

    # Convert back to original dtype
    coord_data_np = coord_data.numpy()
    obj_prob_np = obj_prob.numpy()
    classes_prob_np = classes_prob.numpy()

    if x_dtype.name in ("bfloat16", "float16"):
        coord_data_np = coord_data_np.astype(x_dtype, copy=False)
        obj_prob_np = obj_prob_np.astype(x_dtype, copy=False)
        classes_prob_np = classes_prob_np.astype(x_dtype, copy=False)

    # CeilX alignment: pad output last dim to aligned size (zero padding)
    # Aligns with infershape CeilX(size*2 + 32, 32) / 2
    hw_aligned = (HW * 2 + 32 + 31) // 32 * 32 // 2
    boxes_hw_aligned = (B * HW * 2 + 32 + 31) // 32 * 32 // 2

    coord_pad = hw_aligned - HW
    if coord_pad > 0:
        coord_data_np = np.pad(
            coord_data_np,
            ((0, 0), (0, 0), (0, coord_pad)),
            mode="constant",
            constant_values=0,
        )

    obj_pad = boxes_hw_aligned - (B * HW)
    if obj_pad > 0:
        obj_prob_np = np.pad(
            obj_prob_np, ((0, 0), (0, obj_pad)), mode="constant", constant_values=0
        )

    classes_pad = boxes_hw_aligned - (B * HW)
    if classes_pad > 0:
        classes_prob_np = np.pad(
            classes_prob_np,
            ((0, 0), (0, 0), (0, classes_pad)),
            mode="constant",
            constant_values=0,
        )

    return coord_data_np, obj_prob_np, classes_prob_np
