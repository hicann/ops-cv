#!/usr/bin/env python3
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

__spec__ = {"check_valid": "CheckValidTestSpec"}


class CheckValidTestSpec:
    """check_valid 算子测试规范（kernel 流程）

    golden 用 torch 拼接实现，供 cross_check 交叉比对
    """

    def golden(bbox_tensor, img_metas, **kwargs):
        metas = torch.from_numpy(img_metas.astype(np.float32))
        H = metas[0]
        W = metas[1]
        r = metas[2]
        img_width_x = W * r - 1.0
        img_height_y = H * r - 1.0

        bbox = torch.from_numpy(bbox_tensor.astype(np.float32))
        x0 = bbox[:, 0]
        y0 = bbox[:, 1]
        x1 = bbox[:, 2]
        y1 = bbox[:, 3]

        c1 = x0 >= 0
        c2 = y0 >= 0
        c3 = x1 <= img_width_x
        c4 = y1 <= img_height_y

        valid = c1 & c2 & c3 & c4
        valid_tensor = valid.to(torch.int8).reshape(-1, 1)
        return [valid_tensor.numpy()]

    class TorchImpl:
        """torch 算子拼接实现：用 torch 比较运算拼接 check_valid 公式。"""

        def __init__(self, **kwargs):
            pass

        def __call__(self, bbox_tensor, img_metas, **kwargs):
            metas = img_metas.to(torch.float32)
            H = metas[0]
            W = metas[1]
            r = metas[2]
            img_width_x = W * r - 1.0
            img_height_y = H * r - 1.0

            bbox = bbox_tensor.to(torch.float32)
            x0 = bbox[:, 0]
            y0 = bbox[:, 1]
            x1 = bbox[:, 2]
            y1 = bbox[:, 3]

            c1 = x0 >= 0
            c2 = y0 >= 0
            c3 = x1 <= img_width_x
            c4 = y1 <= img_height_y

            valid = c1 & c2 & c3 & c4
            valid_tensor = valid.to(torch.int8).reshape(-1, 1)
            return [valid_tensor]

    third_party = {
        "torch": TorchImpl,
    }

    tolerance = {
        "int8": {"standard": "binary_equal"},
    }
