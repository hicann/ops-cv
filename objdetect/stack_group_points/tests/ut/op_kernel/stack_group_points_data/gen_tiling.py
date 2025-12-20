#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import numpy as np


def gen_tiling():
    variables_dict = {
        "b": 2,
        "m": 10,
        "c": 12,
        "nsample": 6,
        "res": 0,
        "featuresSize": 256,
        "indicesSize": 256,
        "fbcSize": 32,
        "ibcSize": 32,
        "reminder": 10,
        "outLength": 2880,
        "n": 5,
        "standard": 720,
        "actCore": 48,
    }

    variables_array = [variables_dict[key] for key in variables_dict]
    print("tiling_data:", variables_array)
    return variables_array


def main():
    params_list = gen_tiling()
    base_params = np.array(params_list, dtype=np.uint32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == "__main__":
    main()
