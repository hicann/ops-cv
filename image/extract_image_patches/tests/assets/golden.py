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


def _dtype(x):
    return getattr(x.dtype, "name", None) or str(x.dtype).split(".")[-1]


def _format(kwargs, attrs):
    formats = kwargs.get("input_ori_formats") or kwargs.get("input_formats")
    if formats:
        value = str(formats[0]).upper()
        if "NCHW" in value:
            return "NCHW"
        if "NHWC" in value:
            return "NHWC"
        raise ValueError(f"x format must be NHWC or NCHW, got {formats[0]}")
    nchw, nhwc = (
        all(values[1] == 1 for values in attrs),
        all(values[3] == 1 for values in attrs),
    )
    return "NCHW" if nchw and not nhwc else "NHWC"


def _attrs(name, values):
    if len(values) != 4:
        raise ValueError(f"{name} must contain 4 integers, got {values}")
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
        for value in values
    ):
        raise TypeError(f"{name} must contain integers, got {values}")
    return tuple(int(value) for value in values)


def _validate(x, ksizes, strides, rates, padding, kwargs):
    if len(x.shape) != 4:
        raise ValueError(f"x must be 4D, got {x.shape}")
    if _dtype(x) not in ("float16", "float32", "bfloat16", "int8", "uint8"):
        raise TypeError(f"unsupported ExtractImagePatches dtype: {_dtype(x)}")
    ksizes, strides, rates = (
        _attrs("ksizes", ksizes),
        _attrs("strides", strides),
        _attrs("rates", rates),
    )
    fmt = _format(kwargs, (ksizes, strides, rates))
    if padding not in ("SAME", "VALID"):
        raise ValueError(f"padding must be SAME or VALID, got {padding}")
    h_idx, w_idx, c_idx = (1, 2, 3) if fmt == "NHWC" else (2, 3, 1)
    if any(values[0] != 1 or values[c_idx] != 1 for values in (ksizes, strides, rates)):
        raise ValueError("ksizes/strides/rates values for N and C dimensions must be 1")
    k_h, k_w = ksizes[h_idx], ksizes[w_idx]
    s_h, s_w = strides[h_idx], strides[w_idx]
    r_h, r_w = rates[h_idx], rates[w_idx]
    if k_h <= 0 or k_w <= 0 or s_h <= 0 or s_w <= 0 or r_h < 1 or r_w < 1:
        raise ValueError(
            "spatial ksizes/strides must be positive and spatial rates must be at least 1"
        )
    return fmt, [1, k_h, k_w, 1], [1, s_h, s_w, 1], [1, r_h, r_w, 1]


def _extract_image_patches(x, sizes, strides, rates, padding, fmt):
    import tensorflow as tf

    x = x if fmt == "NHWC" else tf.transpose(x, (0, 2, 3, 1))
    if hasattr(tf, "extract_image_patches"):
        output = tf.extract_image_patches(
            x, ksizes=sizes, strides=strides, rates=rates, padding=padding
        )
    else:
        output = tf.image.extract_patches(
            x, sizes=sizes, strides=strides, rates=rates, padding=padding
        )
    return output if fmt == "NHWC" else tf.transpose(output, (0, 3, 1, 2))


class ExtractImagePatchesSpec:
    def golden(x, *, ksizes, strides, rates, padding, **kwargs):
        fmt, sizes, strides, rates = _validate(
            x, ksizes, strides, rates, padding, kwargs
        )
        return [_extract_image_patches(x, sizes, strides, rates, padding, fmt).numpy()]

    class TfThirdPartyImpl:
        def __init__(self, *, ksizes, strides, rates, padding, **kwargs):
            self.ksizes, self.strides, self.rates, self.padding = (
                ksizes,
                strides,
                rates,
                padding,
            )

        def __call__(self, x, **kwargs):
            fmt, sizes, strides, rates = _validate(
                x, self.ksizes, self.strides, self.rates, self.padding, kwargs
            )
            return [_extract_image_patches(x, sizes, strides, rates, self.padding, fmt)]

    third_party = {"tf": TfThirdPartyImpl}
    tolerance = {
        dtype: {"standard": "cross_check", "level": "L1"}
        for dtype in ("float16", "float32", "bfloat16")
    }
    tolerance.update(
        {dtype: {"standard": "binary_equal"} for dtype in ("int8", "uint8")}
    )


__spec__ = {"extract_image_patches": "ExtractImagePatchesSpec"}
