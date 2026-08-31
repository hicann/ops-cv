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

"""
col2_im_v2 Golden TestSpec（Kernel + GEIR 共用）。

计算方式与竞品 PyTorch col2im 完全一致：torch.nn.functional.fold 即
torch._C._nn.col2im 的用户接口（输出 centric，重叠滑窗位置累加，
未被任何滑窗覆盖的输出位置为 0）。

参数与 REG_OP(Col2ImV2) 一一对应：
  x:           3-D (n, c*kH*kW, ho*wo), float32/float16
  output_size: 1-D const tensor, 长度 2, int32, 值 (outH, outW)
  kernel_size: 1-D const tensor, 长度 2, int32, 值 (kH, kW)
  dilation:    ListInt attr, 长度 2, 元素 > 0（REQUIRED，无默认值）
  padding:     ListInt attr, 长度 2, 元素 >= 0（REQUIRED，无默认值）
  stride:      ListInt attr, 长度 2, 元素 > 0（REQUIRED，无默认值）
返回:
  y: 4-D (n, c, outH, outW), dtype 与 x 相同
"""

import numpy as np
import torch
import torch.nn.functional as F

__spec__ = {
    "col2_im_v2": "Col2ImV2TestSpec",
}


def _size_to_pair(size_arg):
    """const size tensor / list -> (int, int)。几何参数边界归一化，不涉及计算 dtype。"""
    if isinstance(size_arg, torch.Tensor):
        values = size_arg.flatten().tolist()
    else:
        values = list(size_arg)
    return int(values[0]), int(values[1])


def _torch_compute(x, output_size, kernel_size, dilation, padding, stride):
    """Tensor -> Tensor only; preserve caller dtype/device and math order.

    fp16 输入时 F.fold 内部以 acc_type(float) 累加后写回，与 kernel 的
    ACC_T=float 中间累加精度对齐；Golden Promote 路径下输入已被
    框架提升，此处不做任何降精度处理。
    """
    out_h, out_w = _size_to_pair(output_size)
    k_h, k_w = _size_to_pair(kernel_size)
    return F.fold(
        x,
        output_size=(out_h, out_w),
        kernel_size=(k_h, k_w),
        dilation=(int(dilation[0]), int(dilation[1])),
        padding=(int(padding[0]), int(padding[1])),
        stride=(int(stride[0]), int(stride[1])),
    )


class Col2ImV2TestSpec:
    """One TestSpec shared by Kernel and GEIR."""

    def golden(x, output_size, kernel_size, *, dilation, padding, stride, **kwargs):
        torch_inputs = [
            torch.from_numpy(np.ascontiguousarray(t))
            for t in [x, output_size, kernel_size]
        ]
        result = _torch_compute(
            torch_inputs[0], torch_inputs[1], torch_inputs[2], dilation, padding, stride
        )
        return [result.detach().cpu().numpy()]

    class ThirdPartyImpl:
        def __init__(self, dilation, padding, stride, **kwargs):
            self.dilation = dilation
            self.padding = padding
            self.stride = stride

        def __call__(self, x, output_size, kernel_size, **kwargs):
            # 直接消费远端绑定的原 dtype provider Tensor，独立执行同一共享计算核心，
            # 禁止反调 Golden wrapper 或做 numpy/CPU 往返。
            result = _torch_compute(
                x, output_size, kernel_size, self.dilation, self.padding, self.stride
            )
            return [result]

    # GEIR remote dispatch needs an explicit provider. A direct class alone is not portable.
    third_party = {"torch": ThirdPartyImpl}
    # 浮点比对采用标准 cross_check（第三方交叉校验）：计算类算子浮点 dtype 的
    # 规范比对方式；整型/布尔才使用 binary_equal。本算子 kernel 与 Golden
    # （ATen CPU col2im 语义：(h_k,w_k) 字典序累加 + 输出 dtype 逐步舍入）
    # 为位一致设计，TTK 实测 mere/mare=0.0，cross_check 与本地容差下均可通过。
    # third_party 的 torch provider 即为 cross_check 提供三方实现。
    tolerance = {
        "float32": {"standard": "cross_check"},
        "float16": {"standard": "cross_check"},
    }
