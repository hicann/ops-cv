# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-side property checks of the actual compensated geometry header."""

import ctypes
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

source = (
    Path(__file__).resolve().parents[2] / "op_kernel/arch35/rotated_overlaps_kernel.h"
).read_text()
body = source[
    source.index("struct PairBox") : source.index(
        "template <bool Trans, typename IndexT>"
    )
]
constants = source[
    source.index("constexpr float kDegreesToRadians =") : source.index(
        "enum FloatVectorSlot"
    )
]
prefix = """#include <cmath>
#include <cstdint>
#define __simt_callee__
#define __gm__
using std::isfinite;
constexpr uint32_t kCoordinateCount=5;
"""
suffix = """
extern "C" void trig(float x, double* out) {
  FloatExpansion s,c; ExpandedSinCos(x,s,c);
  out[0]=double(s.high)+s.low; out[1]=double(c.high)+c.low;
}
extern "C" float area(float* a, float* b, int trans, int optimized) {
  PairBox x,y;
  if(trans) {LoadPairBox<true>(a,1,0,0,x);LoadPairBox<true>(b,1,0,0,y);}
  else {LoadPairBox<false>(a,1,0,0,x);LoadPairBox<false>(b,1,0,0,y);}
  if(!x.valid || !y.valid) return 0;
  if(optimized && DefinitelyDisjoint(x,y)) return 0;
  if(x.width*x.height>y.width*y.height) {auto tmp=x;x=y;y=tmp;}
  return trans ? PreciseIntersectionArea<true>(x,y) : PreciseIntersectionArea<false>(x,y);
}
"""
prefix += constants
with tempfile.TemporaryDirectory(prefix="rotated-perf-host-") as temp:
    temp = Path(temp)
    (temp / "test.cpp").write_text(prefix + body + suffix)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(temp / "test.cpp"),
            "-o",
            str(temp / "test.so"),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(temp / "test.so"))
    lib.trig.argtypes = [ctypes.c_float, ctypes.POINTER(ctypes.c_double)]
    lib.area.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [ctypes.c_int] * 2
    lib.area.restype = ctypes.c_float
    rng = np.random.default_rng(20260908)
    angles = np.r_[
        rng.uniform(-1e6, 1e6, 10000).astype(np.float32),
        np.arange(-360, 361, 45, dtype=np.float32),
    ]
    worst = 0.0
    for x in angles:
        out = (ctypes.c_double * 2)()
        lib.trig(float(x), out)
        r = math.remainder(float(x), 360) * math.pi / 180
        worst = max(worst, abs(out[0] - math.sin(r)), abs(out[1] - math.cos(r)))
    assert worst < 2e-14, worst
    from shapely.geometry import Polygon

    def polygon(a):
        x, y, w, h, t = map(float, a)
        t = math.radians(t)
        s = math.sin(t)
        c = math.cos(t)
        return Polygon(
            [
                (x + dx * c - dy * s, y + dx * s + dy * c)
                for dx, dy in [
                    (-w / 2, -h / 2),
                    (w / 2, -h / 2),
                    (w / 2, h / 2),
                    (-w / 2, h / 2),
                ]
            ]
        )

    def run(a, b, optimized, trans=0):
        return lib.area(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            trans,
            optimized,
        )

    for i in range(4000):
        scale = 10.0 ** rng.uniform(-9, 9)
        a = np.r_[
            rng.uniform(-3, 3, 2) * scale,
            rng.uniform(0.1, 5, 2) * scale,
            rng.uniform(-360, 360),
        ].astype(np.float32)
        b = np.r_[
            rng.uniform(-3, 3, 2) * scale,
            rng.uniform(0.1, 5, 2) * scale,
            rng.uniform(-360, 360),
        ].astype(np.float32)
        if i % 4 == 0:
            b[:2] = a[:2]
            b[2:4] = a[2:4] * 0.001
            b[4] = a[4]
        result = run(a, b, 1)
        baseline = run(a, b, 0)
        assert result == baseline, (i, result, baseline)
        expected = np.float32(polygon(a).intersection(polygon(b)).area)
        assert np.isclose(result, expected, rtol=2e-6, atol=0), (i, result, expected)
    # Tiny positive overlaps, exact contact, separated boxes; bound must not
    # discard a positive sliver. Exercise the smallest normal float32 scales.
    for scale in (1e-19, 1e-9, 1.0, 1e9):
        for shift in (0.9999999, 1.0, 1.0000001):
            a = np.array([0, 0, scale, scale, 45], np.float32)
            b = a.copy()
            b[0] = scale * shift / math.sqrt(2)
            b[1] = b[0]
            assert run(a, b, 1) == run(a, b, 0)
    for i in range(20000):
        scale = 10.0 ** rng.uniform(-18, 18)
        a = np.r_[rng.uniform(-5, 5, 4) * scale, rng.uniform(-4000, 4000)].astype(
            np.float32
        )
        b = np.r_[rng.uniform(-5, 5, 4) * scale, rng.uniform(-4000, 4000)].astype(
            np.float32
        )
        for x in (a, b):
            x[0], x[2] = min(x[0], x[2]), max(x[0], x[2])
            x[1], x[3] = min(x[1], x[3]), max(x[1], x[3])
        assert run(a, b, 1, 1) == run(a, b, 0, 1), (i, a, b)
    print(
        "PASS: 10017 trig inputs, 4000 geometry/Shapely checks, 12 contact checks, 20000 endpoint-bound checks; max trig abs error",
        worst,
    )
