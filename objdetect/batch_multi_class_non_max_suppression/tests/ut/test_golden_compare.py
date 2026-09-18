#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import importlib.util
import unittest
from pathlib import Path

import numpy


def _load_golden_module():
    golden_path = Path(__file__).parents[1] / "assets" / "golden.py"
    spec = importlib.util.spec_from_file_location(
        "batch_nms_assets_golden", golden_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOLDEN_MODULE = _load_golden_module()
COMPARE = GOLDEN_MODULE.BatchMultiClassNonMaxSuppressionTestSpec.compare


def _reference_outputs():
    boxes = numpy.zeros((1, 10, 4), dtype=numpy.float32)
    for index in range(10):
        boxes[0, index] = [3 * index, 0, 3 * index + 1, 1]
    scores = numpy.linspace(0.9, 0.1, 10, dtype=numpy.float32).reshape(1, 10)
    classes = numpy.zeros((1, 10), dtype=numpy.float32)
    count = numpy.array([10], dtype=numpy.int32)
    return boxes, scores, classes, count


def _compare(npu_outputs, golden_outputs):
    return COMPARE(*npu_outputs, *golden_outputs)


class TestGoldenCompare(unittest.TestCase):
    def test_accepts_equivalent_detection_permutation(self):
        golden = _reference_outputs()
        npu = tuple(value.copy() for value in golden)
        for value in npu[:3]:
            value[0] = value[0, ::-1]
        self.assertTrue(all(result["pass"] for result in _compare(npu, golden)))

    def test_rejects_missing_detections(self):
        golden = _reference_outputs()
        npu = tuple(value.copy() for value in golden)
        for value in npu[:3]:
            value[0, 7:] = 0
        npu[3][0] = 7
        results = _compare(npu, golden)
        self.assertFalse(results[0]["pass"])
        self.assertFalse(results[3]["pass"])

    def test_rejects_wrong_classes(self):
        golden = _reference_outputs()
        npu = tuple(value.copy() for value in golden)
        npu[2][0, :2] = 1
        self.assertFalse(_compare(npu, golden)[2]["pass"])

    def test_rejects_four_percent_score_error(self):
        golden = _reference_outputs()
        npu = tuple(value.copy() for value in golden)
        npu[1][...] *= 1.04
        self.assertFalse(_compare(npu, golden)[1]["pass"])


if __name__ == "__main__":
    unittest.main()
