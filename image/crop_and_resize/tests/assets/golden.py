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

"""Golden for crop_and_resize, directly calling tf.image.crop_and_resize."""

import numpy as np
import tensorflow as tf


def _tf_crop_and_resize_safe(
    x_np, boxes_f32, box_index, crop_size, extrapolation_value, out_dtype
):
    """Call tf.image.crop_and_resize directly."""
    y_tf = tf.image.crop_and_resize(
        tf.constant(x_np),
        tf.constant(boxes_f32),
        tf.constant(np.asarray(box_index, dtype=np.int32)),
        [int(crop_size[0]), int(crop_size[1])],
        method="bilinear",
        extrapolation_value=float(extrapolation_value),
    )
    # TF 输出始终为 float32，需转为与 kernel 输出一致的 dtype（out_dtype 取自 boxes dtype）
    # 当 boxes=FP16 时 kernel 输出 FP16，需做 float32→FP16 截断以对齐精度对比
    return y_tf.numpy().astype(out_dtype)


class CropAndResizeComposeSpec:
    """Class form — composition for crop_and_resize"""

    # -- golden — function form --
    @staticmethod
    def golden(
        x,
        boxes,
        box_index,
        crop_size,
        *,
        extrapolation_value=0.0,
        method="bilinear",
        **kwargs,
    ):
        """
        Golden function for crop_and_resize.
        All the parameters (names and order) follow SE doc prototype definition without outputs.
        All the input Tensors are numpy.ndarray.

        Args:
            x: input image, shape (N, H, W, C)
            boxes: crop boxes, shape (num_boxes, 4)
            box_index: box index per batch, shape (num_boxes,)
            crop_size: [crop_height, crop_width]
            extrapolation_value: float, value for out-of-bounds
            method: "bilinear" only
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            Output tensor, shape (num_boxes, crop_height, crop_width, C)
        """
        if method != "bilinear":
            raise ValueError(
                f"Unsupported method: {method}, only bilinear is supported"
            )

        boxes_np = np.asarray(boxes)
        if boxes_np.dtype in (np.float16, np.float32):
            out_dtype = boxes_np.dtype
        else:
            out_dtype = np.float32

        x_np = np.asarray(x).astype(np.float32)
        boxes_f32 = boxes_np.astype(np.float32)
        result = _tf_crop_and_resize_safe(
            x_np,
            boxes_f32,
            box_index,
            crop_size,
            extrapolation_value,
            out_dtype,
        )
        return result

    # -- third_party — dict multi-vendor --
    class ThirdPartyImpl:
        def __init__(
            self,
            *,
            extrapolation_value=0.0,
            method="bilinear",
            x=None,
            boxes=None,
            box_index=None,
            crop_size=None,
            **kwargs,
        ):
            if method != "bilinear":
                raise ValueError(
                    f"Unsupported method: {method}, only bilinear is supported"
                )
            self.extrapolation_value = extrapolation_value
            self.method = method

            # dtype 判断和数据转换前移到 __init__（TTK _bind 会把输入 tensor 同时喂给 __init__ 和 __call__）
            boxes_np = np.asarray(boxes)
            if boxes_np.dtype in (np.float16, np.float32):
                self.out_dtype = boxes_np.dtype
            else:
                self.out_dtype = np.float32

            self.x_np = np.asarray(x).astype(np.float32)
            self.boxes_f32 = boxes_np.astype(np.float32)
            self.box_index = box_index
            self.crop_size = crop_size

        def __call__(
            self, x=None, boxes=None, box_index=None, crop_size=None, **kwargs
        ):
            return _tf_crop_and_resize_safe(
                self.x_np,
                self.boxes_f32,
                self.box_index,
                self.crop_size,
                self.extrapolation_value,
                self.out_dtype,
            )

    third_party = {
        "tf": ThirdPartyImpl,
    }

    # -- tolerance --
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "crop_and_resize": "CropAndResizeComposeSpec",
}
