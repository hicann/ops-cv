#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import sys

test_case_0 = [
    False, False, 1, 8, 0, 1, 39, 32,
    8, 1, 8, 8, 8, 8, 8, 6,
    0.5, 1, 2, 2, 196352
]

test_case_1 = [
    False, False, 1, 8, 0, 1, 39, 32,
    8, 2, 8, 8, 8, 8, 8, 5,
    0.5, 1, 2, 2, 196352
]

test_case_2 = [
    False, False, 1, 8, 0, 1, 39, 32,
    8, 3, 8, 8, 8, 8, 8, 0,
    0.5, 1, 2, 2, 196352
]

params_info = {
    "test_case_0": test_case_0,
    "test_case_1": test_case_1,
    "test_case_2": test_case_2
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list[:], dtype=np.int32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)

if __name__ == '__main__':
    main()