#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Golden for crop_and_resize, directly calling tf.image.crop_and_resize (NCHW-aware).

When x format is NCHW: transpose x to NHWC, compute, transpose result back to
NCHW (NCHW in, NCHW out, aligning with canndev TBE semantics).
When x format is ND/NHWC (or input_formats is absent): legacy NHWC path.
"""

import tensorflow as tf


def _is_nchw_input(input_formats):
    """Return True iff the x input format (first entry) is NCHW."""
    if not input_formats:
        return False
    if isinstance(input_formats, str):
        x_format = input_formats
    else:
        try:
            x_format = input_formats[0]
        except (TypeError, IndexError, KeyError):
            return False
    return str(x_format).strip().upper() == "NCHW"


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
            x: input image, (N, H, W, C) for ND/NHWC, (N, C, H, W) for NCHW
            boxes: crop boxes, shape (num_boxes, 4)
            box_index: box index per batch, shape (num_boxes,)
            crop_size: [crop_height, crop_width]
            extrapolation_value: float, value for out-of-bounds
            method: "bilinear" only; others rejected
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            Output tensor. (num_boxes, crop_h, crop_w, C) for ND/NHWC input;
            (num_boxes, C, crop_h, crop_w) for NCHW input.
        """
        if method != "bilinear":
            raise ValueError(
                f"Unsupported method: {method}, only bilinear is supported by tf.image.crop_and_resize"
            )

        is_nchw = _is_nchw_input(kwargs.get("input_formats"))

        boxes_tf_dtype = tf.dtypes.as_dtype(boxes.dtype)
        out_dtype = (
            boxes_tf_dtype if boxes_tf_dtype in (tf.float16, tf.float32) else tf.float32
        )

        if is_nchw:
            x_tf = tf.cast(tf.transpose(x, (0, 2, 3, 1)), tf.float32)
        else:
            x_tf = tf.cast(x, tf.float32)
        boxes_tf = tf.cast(boxes, tf.float32)
        box_index_tf = tf.cast(box_index, tf.int32)
        crop_size_list = [int(crop_size[0]), int(crop_size[1])]
        extrapolation_value_f = float(extrapolation_value)

        y_tf = tf.image.crop_and_resize(
            x_tf,
            boxes_tf,
            box_index_tf,
            crop_size_list,
            method="bilinear",
            extrapolation_value=extrapolation_value_f,
        )
        if is_nchw:
            y_tf = tf.transpose(y_tf, (0, 3, 1, 2))
        # TF 输出始终为 float32，需转为与 kernel 输出一致的 dtype（out_dtype 取自 boxes dtype）
        # 当 boxes=FP16 时 kernel 输出 FP16，需做 float32→FP16 截断以对齐精度对比
        return tf.cast(y_tf, out_dtype).numpy()

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
                    f"Unsupported method: {method}, only bilinear is supported by tf.image.crop_and_resize"
                )
            self.is_nchw = _is_nchw_input(kwargs.get("input_formats"))

            boxes_tf_dtype = tf.dtypes.as_dtype(boxes.dtype)
            self.out_dtype = (
                boxes_tf_dtype
                if boxes_tf_dtype in (tf.float16, tf.float32)
                else tf.float32
            )

            if self.is_nchw:
                self.x_tf = tf.cast(tf.transpose(x, (0, 2, 3, 1)), tf.float32)
            else:
                self.x_tf = tf.cast(x, tf.float32)
            self.boxes_tf = tf.cast(boxes, tf.float32)
            self.box_index_tf = box_index
            self.crop_size_list = [int(crop_size[0]), int(crop_size[1])]
            self.extrapolation_value = float(extrapolation_value)

        def __call__(
            self, x=None, boxes=None, box_index=None, crop_size=None, **kwargs
        ):
            y_tf = tf.image.crop_and_resize(
                self.x_tf,
                self.boxes_tf,
                self.box_index_tf,
                self.crop_size_list,
                method="bilinear",
                extrapolation_value=self.extrapolation_value,
            )
            if self.is_nchw:
                y_tf = tf.transpose(y_tf, (0, 3, 1, 2))
            # TF 输出始终为 float32，需转为与 kernel 输出一致的 dtype（out_dtype 取自 boxes dtype）
            # 当 boxes=FP16 时 kernel 输出 FP16，需做 float32→FP16 截断以对齐精度对比
            return tf.cast(y_tf, self.out_dtype).numpy()

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
