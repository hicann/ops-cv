#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import ast
from pathlib import Path
import numpy as np


def gen_data_and_golden(shape_str, d_type="float32", align_corners="False"):
    np_type = {"float32": np.float32, "float16": np.float16}[d_type]
    parsed_shape = ast.literal_eval(shape_str)
    shape = (parsed_shape,) if isinstance(parsed_shape, int) else tuple(parsed_shape)
    al = str(align_corners).strip().lower() in ("true", "1", "yes")

    rng = np.random.default_rng(20260728)
    grid = rng.uniform(-1.0, 1.0, size=shape).astype(np_type)
    assist = rng.uniform(1.0, 64.0, size=shape).astype(np_type)
    if grid.size >= 6:
        grid_flat = grid.reshape(-1)
        assist_flat = assist.reshape(-1)
        grid_flat[:6] = np.array([-1.0, -0.984375, 0.0, 1.0, 0.0, 1.0], dtype=np_type)
        assist_flat[:6] = np.array([64.0, 64.0, 3.0, 1.0, 1.0, 2.0], dtype=np_type)

    # golden：按算子规格公式独立生成，统一 fp32 中间计算
    g = grid.astype(np.float32)
    a = assist.astype(np.float32)
    t = (g + 1.0) * 0.5
    pos_base = t * (a - 1.0) if al else t * a - 0.5
    floor = np.floor(pos_base)
    position = floor.astype(np.int32)
    diff = (pos_base - floor).astype(np_type)

    grid.tofile(f"{d_type}_grid_grid_unnormal.bin")
    assist.tofile(f"{d_type}_assist_grid_unnormal.bin")
    diff.tofile(f"{d_type}_golden_diff_grid_unnormal.bin")
    position.tofile(f"{d_type}_golden_position_grid_unnormal.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: gen_data.py <shape> <dtype> <align_corners>")
        exit(1)
    for path in Path(".").glob("*.bin"):
        if path.is_file():
            path.unlink()
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])
