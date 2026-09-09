#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Reproduce and bit-check the kernel's binary64-to-two-binary32 coefficients."""

import math
from pathlib import Path
import re
import struct


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def main():
    source = (
        Path(__file__).resolve().parents[2]
        / "op_kernel/arch35/rotated_overlaps_kernel.h"
    ).read_text()
    for name, offset in (("sin", 1), ("cos", 0)):
        table = re.search(name + r"Coefficients\[8\] = \{(.*?)\};", source, re.S)
        pairs = re.findall(r"\{([^,{}]+)F,\s*([^,{}]+)F\}", table.group(1))
        assert len(pairs) == 8
        for k, (high_text, low_text) in enumerate(pairs, 1):
            coefficient = (-1) ** k / math.factorial(2 * k + offset)
            high = float32(coefficient)
            low = float32(coefficient - high)
            assert struct.pack("ff", high, low) == struct.pack(
                "ff", float(high_text), float(low_text)
            ), (name, k)
            print(f"{name}[{k}]: {{{high:.17g}F, {low:.17g}F}}")
        # Alternating decreasing terms on |x| <= pi/4: first omitted term.
        power = 18 + offset
        print(
            f"{name} truncation <= {(math.pi / 4) ** power / math.factorial(power):.9e}"
        )
    print("PASS: all 16 coefficient pairs match bit-for-bit")


if __name__ == "__main__":
    main()
