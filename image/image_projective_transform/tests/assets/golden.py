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

"""
Golden for image_projective_transform (kernel level).

计算逻辑: 使用 TensorFlow ``tf.raw_ops.ImageProjectiveTransformV2`` 作为参考实现,
辅以 numpy 早退判定 (transforms 含 inf/nan/超大系数时按 BILINEAR 填 nan、否则填 0)。

执行方式说明 (重要):
    TTK worker 进程在 profiling 阶段会通过 ``ctypes.CDLL(libruntime.so, RTLD_GLOBAL)``
    加载 CANN 运行时库。此后在 *同一进程* 内 ``import tensorflow`` 会触发 SIGSEGV
    (libruntime 与 TF 的共享库符号冲突)。经实验确认:
      - 同进程 libruntime -> import TF : SIGSEGV (exit 139)
      - 同进程 libruntime + subprocess.Popen 子进程 import TF : OK (exit 0)
    原因: ``subprocess.Popen`` 使用 fork+exec, exec 会替换进程镜像并卸载父进程通过
    ctypes 加载的共享库, 因此子进程获得干净的 TF 运行环境。
    故本 golden 将 TF 计算放入独立子进程执行, 计算逻辑保持不变。
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

__golden__ = {
    "kernel": {"image_projective_transform": "image_projective_transform_golden"}
}


# ---------------------------------------------------------------------------
# TF 计算核心: 在独立子进程中执行, 避免 libruntime 与 TF 同进程冲突
# ---------------------------------------------------------------------------
def _tf_compute(images, transforms_f32, output_shape, interpolation, fill_mode, out_dtype):
    """在独立子进程中调用 tf.raw_ops.ImageProjectiveTransformV2。

    通过临时文件传递输入/输出 numpy 数组与标量参数; 子进程由
    ``subprocess.Popen`` (fork+exec) 启动, 不继承父进程 ctypes 加载的 libruntime。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "images.npy"), images)
        np.save(os.path.join(tmpdir, "transforms.npy"), transforms_f32)
        np.save(os.path.join(tmpdir, "output_shape.npy"), np.asarray(output_shape))
        with open(os.path.join(tmpdir, "params.json"), "w") as f:
            json.dump(
                {
                    "interpolation": interpolation,
                    "fill_mode": fill_mode,
                    "out_dtype": np.dtype(out_dtype).name,
                },
                f,
            )
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--tf-subprocess", tmpdir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ),
        )
        _out, err = proc.communicate(timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(
                "TF golden subprocess exited %d:\n%s"
                % (proc.returncode, err.decode(errors="replace")[-2000:])
            )
        return np.load(os.path.join(tmpdir, "result.npy"))


# ---------------------------------------------------------------------------
# Golden 主函数
# ---------------------------------------------------------------------------
def image_projective_transform_golden(images, transforms, output_shape,
                                       *, interpolation, fill_mode="CONSTANT",
                                       **kwargs):
    out_dtype = images.dtype
    n = images.shape[0]
    c = images.shape[3]
    transforms_f32 = np.asarray(transforms, dtype=np.float32).reshape(n, 8)

    out_shape_flat = np.asarray(output_shape).reshape(-1)
    h_out = int(out_shape_flat[0])
    w_out = int(out_shape_flat[1])

    result_np = _tf_compute(
        images, transforms_f32,
        output_shape, interpolation, fill_mode, out_dtype)
    return result_np.reshape(n, h_out, w_out, c)


# ---------------------------------------------------------------------------
# 子进程入口: 当以 ``python golden.py --tf-subprocess <tmpdir>`` 运行时,
# 读取临时文件中的输入, 执行 TF 计算, 写回结果。此分支在 TTK 以 importlib
# 加载本模块时不会触发 (__name__ != "__main__")。
# ---------------------------------------------------------------------------
if __name__ == "__main__" and "--tf-subprocess" in sys.argv:
    tmpdir = sys.argv[sys.argv.index("--tf-subprocess") + 1]
    images = np.load(os.path.join(tmpdir, "images.npy"))
    transforms_f32 = np.load(os.path.join(tmpdir, "transforms.npy"))
    output_shape = np.load(os.path.join(tmpdir, "output_shape.npy"))
    with open(os.path.join(tmpdir, "params.json")) as _f:
        _params = json.load(_f)
    _out_dtype = np.dtype(_params["out_dtype"])

    import tensorflow as tf

    tf_images = tf.constant(images)
    tf_transforms = tf.constant(transforms_f32, dtype=tf.float32)
    tf_output_shape = tf.constant(output_shape, dtype=tf.int32)

    _result = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf_images,
        transforms=tf_transforms,
        output_shape=tf_output_shape,
        interpolation=_params["interpolation"],
        fill_mode=_params["fill_mode"],
    )
    _result_np = _result.numpy()
    if _result_np.dtype != _out_dtype:
        _result_np = _result_np.astype(_out_dtype)
    np.save(os.path.join(tmpdir, "result.npy"), _result_np)
    sys.exit(0)
