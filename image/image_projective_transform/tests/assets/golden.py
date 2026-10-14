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

"""TensorFlow V2 TestSpec for image_projective_transform Kernel/GEIR tests.

Golden main path: ``tf.raw_ops.ImageProjectiveTransformV2`` (3 inputs, no fill_value),
which natively covers the v2.0 enhancement scope (SE §7):
- fill_mode four modes CONSTANT/REFLECT/WRAP/NEAREST (NEAREST is undocumented in TF
  but supported by the TF 2.21.0 kernel, matching SE §5.2-A formulas point-by-point);
- one_trans broadcast: transforms ``[1, 8]`` is broadcast to every batch natively
  (bit-equivalent to explicitly tiling it to ``[N, 8]``, SE §5.2-B.4).
No numpy computation is performed; numpy is only the container boundary.
"""

import tensorflow as tf

__spec__ = {
    "image_projective_transform": "ImageProjectiveTransformTestSpec",
}


def _image_projective_transform_v2(
    images, transforms, output_shape, *, interpolation, fill_mode
):
    """Call the TensorFlow V2 reference API with the actual OpDef signature."""
    return tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.convert_to_tensor(images),
        transforms=tf.cast(tf.convert_to_tensor(transforms), tf.float32),
        output_shape=tf.cast(tf.convert_to_tensor(output_shape), tf.int32)[:2],
        interpolation=interpolation,
        fill_mode=fill_mode,
    )


class ImageProjectiveTransformTestSpec:
    """Golden receives numpy arrays; the tf provider receives tf.Tensor inputs."""

    def golden(
        images,
        transforms,
        output_shape,
        *,
        interpolation,
        fill_mode="CONSTANT",
        **kwargs,
    ):
        result = _image_projective_transform_v2(
            images,
            transforms,
            output_shape,
            interpolation=interpolation,
            fill_mode=fill_mode,
        )
        return [result.numpy()]

    class TfThirdPartyImpl:
        def __init__(self, *, interpolation, fill_mode="CONSTANT", **kwargs):
            self.interpolation = interpolation
            self.fill_mode = fill_mode

        def __call__(self, images, transforms, output_shape, **kwargs):
            result = _image_projective_transform_v2(
                images,
                transforms,
                output_shape,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
            )
            return [result]

    third_party = {"tf": TfThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "uint8": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
    }
