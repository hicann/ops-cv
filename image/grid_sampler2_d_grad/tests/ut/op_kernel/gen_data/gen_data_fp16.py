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

import os
import numpy as np
import stat
import torch

OPEN_FILE_MODES_640 = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
WRITE_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

def write_file(shape, input):
    if "x" in input:
        x = np.random.randint(1, 255, shape).astype(np.float16)
        x.tofile("./x_16.bin")
    if "grid" in input:
        grid = np.random.random(shape).astype(np.float16)
        grid.tofile("./grid_16.bin")
    if "grad" in input:
        grad = np.random.random(shape).astype(np.float16)
        grad.tofile("./grad_16.bin")




def gen_tiling():
    batch = 8
    batchPerCore = 1
    tailBatch = 0
    channel = 8
    height = 8
    width = 8
    blockNum = 1
    ubSize = (1024 * 192 - 2 * 1024) // 2  // 32 * 32
    
    perUb = ubSize // 45 // 32 * 32
    dataTypeSize = 2
    ubFactorElement = perUb // 4
    interpolation = 1
    padding = 0
    alignCorners = 1
    gridH = 8
    gridW = 8
    tiling = (np.array(i, dtype=np.uint32) for i in (batch, batchPerCore, tailBatch, channel,
                                                    height, width, blockNum, ubFactorElement, 
                                                    interpolation, padding, alignCorners, gridH, gridW
                                                    ))
    tiling_data = b''.join(x.tobytes() for x in tiling)

    with os.fdopen(os.open('./tiling_16.bin', WRITE_FILE_FLAGS, OPEN_FILE_MODES_640), 'wb') as f:
        f.write(tiling_data)

if __name__ == "__main__":
    x_shape = [8, 8, 8, 8] # NHWC
    grid_shape = [8, 8, 8, 2]
    grad_shape = [8, 8, 8, 8]
    write_file(x_shape, "x_shape")
    write_file(grid_shape, "grid_shape")
    write_file(grad_shape, "grad_shape")
    gen_tiling()