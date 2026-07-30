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
import os
import glob
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))


def _one(pattern):
    files = glob.glob(os.path.join(curr_dir, pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return files[0]


def process(d_type):
    np_type = {"float32": np.float32, "float16": np.float16}[d_type]
    # diff/position 双输出各自 dtype 与容差
    rtol = 1e-5 if d_type == "float32" else 1e-4
    atol = 1e-4 if d_type == "float32" else 1e-3

    diff_out = np.fromfile(_one(f"{d_type}_output_diff*.bin"), np_type).astype(
        np.float32
    )
    diff_gold = np.fromfile(_one(f"{d_type}_golden_diff*.bin"), np_type).astype(
        np.float32
    )
    pos_out = np.fromfile(_one(f"{d_type}_output_position*.bin"), np.int32)
    pos_gold = np.fromfile(_one(f"{d_type}_golden_position*.bin"), np.int32)

    ok = True
    if not np.allclose(diff_out, diff_gold, rtol=rtol, atol=atol):
        bad = np.where(~np.isclose(diff_out, diff_gold, rtol=rtol, atol=atol))[0]
        print(f"DIFF FAILED! mismatched={len(bad)}")
        for i in bad[:5]:
            print(f"  idx {i}: out={diff_out[i]}, gold={diff_gold[i]}")
        ok = False
    if not np.array_equal(pos_out, pos_gold):
        bad = np.where(pos_out != pos_gold)[0]
        print(f"POSITION FAILED! mismatched={len(bad)}")
        for i in bad[:5]:
            print(f"  idx {i}: out={pos_out[i]}, gold={pos_gold[i]}")
        ok = False

    print("PASSED!" if ok else "FAILED!")
    return ok


if __name__ == "__main__":
    ret = process(sys.argv[1])
    exit(0 if ret else 1)
