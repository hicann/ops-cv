#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Route-specific goldens and competitors for NonMaxSuppressionV7.

CPU golden and GPU third-party both use ONNX Runtime's native
``NonMaxSuppression`` node, selecting the CPU and CUDA execution providers
respectively.  They share the public V7 output adaptation so their semantics
differ only by execution device and remain independent from the NPU Kernel.
NumPy is used only at the TTK output boundary.
"""

import importlib.util
from pathlib import Path

import torch


def _load_inputs_plugin():
    input_path = Path(__file__).with_name("inputs.py")
    spec = importlib.util.spec_from_file_location(
        "non_max_suppression_v7_test_inputs", input_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.non_max_suppression_v7_inputs


_non_max_suppression_v7_inputs_impl = None


def non_max_suppression_v7_inputs(*args, **kwargs):
    """Expose a source-defined input hook so TTK plugin discovery can find it."""
    global _non_max_suppression_v7_inputs_impl
    if _non_max_suppression_v7_inputs_impl is None:
        _non_max_suppression_v7_inputs_impl = _load_inputs_plugin()
    return _non_max_suppression_v7_inputs_impl(*args, **kwargs)


__spec__ = {
    # Kernel and GEIR share the raw op_name and the same numpy-facing golden.
    "non_max_suppression_v7": "NonMaxSuppressionV7KernelSpec",
}
__golden__ = {
    "kernel": {"non_max_suppression_v7": "non_max_suppression_v7_kernel_golden"},
    "geir": {"non_max_suppression_v7": "non_max_suppression_v7_geir_golden"},
}
__input__ = {
    "kernel": {"non_max_suppression_v7": "non_max_suppression_v7_inputs"},
    "geir": {"non_max_suppression_v7": "non_max_suppression_v7_inputs"},
}


KERNEL_OUTPUT_TOLERANCE = {
    # boxes/scores support float16 and float32 independently, but the only
    # output is selected_indices:int32 for every dtype combination.
    "int32": {"standard": "binary_equal"},
}
GEIR_OUTPUT_TOLERANCE = {
    "int32": {"standard": "binary_equal"},
}


def _scalar(value, default):
    if value is None:
        return default
    return torch.as_tensor(value).reshape(-1)[0].item()


_ONNX_SESSIONS = {}


def _onnx_dependencies(require_cuda=False):
    """Load the ONNX reference dependencies for the requested provider."""
    try:
        onnx = importlib.import_module("onnx")
        ort = importlib.import_module("onnxruntime")
        ort_capi = importlib.import_module("onnxruntime.capi._pybind_state")
    except ImportError as error:
        raise RuntimeError(
            "NonMaxSuppressionV7 reference requires onnx and onnxruntime"
        ) from error
    if require_cuda and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("onnxruntime-gpu has no CUDAExecutionProvider")
    return onnx, ort, ort_capi


def _onnx_nms_model(center_point_box):
    """Build the native ONNX NonMaxSuppression model in memory."""
    onnx, _, _ = _onnx_dependencies()
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    inputs = [
        helper.make_tensor_value_info("boxes", tensor_proto.FLOAT, [None, None, 4]),
        helper.make_tensor_value_info("scores", tensor_proto.FLOAT, [None, None, None]),
        helper.make_tensor_value_info("max_output", tensor_proto.INT64, []),
        helper.make_tensor_value_info("iou_threshold", tensor_proto.FLOAT, []),
        helper.make_tensor_value_info("score_threshold", tensor_proto.FLOAT, []),
    ]
    output = helper.make_tensor_value_info(
        "selected_indices", tensor_proto.INT64, [None, 3]
    )
    node = helper.make_node(
        "NonMaxSuppression",
        [value.name for value in inputs],
        [output.name],
        center_point_box=int(center_point_box),
        name="direct_onnx_non_max_suppression",
    )
    graph = helper.make_graph(
        [node], "direct_onnx_non_max_suppression", inputs, [output]
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        producer_name="non-max-suppression-v7-direct-competitor",
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _onnx_session(center_point_box, device_type, device_id, stream):
    """Reuse an ONNX Runtime session bound to the requested device."""
    key = (int(center_point_box), device_type, int(device_id), int(stream))
    cached = _ONNX_SESSIONS.get(key)
    if cached is not None:
        return cached

    require_cuda = device_type == "cuda"
    _, ort, _ = _onnx_dependencies(require_cuda=require_cuda)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if require_cuda:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": int(device_id),
                    "user_compute_stream": str(stream),
                    "do_copy_in_default_stream": True,
                },
            )
        ]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(
        _onnx_nms_model(center_point_box),
        sess_options=options,
        providers=providers,
    )
    session.disable_fallback()
    _ONNX_SESSIONS[key] = session
    return session


def _ortvalue_from_torch(tensor):
    """Create a zero-copy OrtValue for a torch tensor."""
    _, ort, ort_capi = _onnx_dependencies()
    if hasattr(ort.OrtValue, "from_dlpack"):
        return ort.OrtValue.from_dlpack(tensor)
    capsule = torch.utils.dlpack.to_dlpack(tensor)
    return ort.OrtValue(ort_capi.OrtValue.from_dlpack(capsule))


def _torch_from_ortvalue(value):
    """Create a zero-copy torch view of an ONNX Runtime output."""
    if hasattr(value, "__dlpack__"):
        return torch.from_dlpack(value)
    return torch.utils.dlpack.from_dlpack(value._ortvalue.to_dlpack())


def _reference_index_map(index_id, batch_count, class_count, box_count, device):
    """Build the V7 mapping using the execution device's native conversion."""
    if index_id is None:
        batch_ids = torch.arange(batch_count, dtype=torch.int32, device=device)
        class_ids = torch.arange(class_count, dtype=torch.int32, device=device)
        box_ids = torch.arange(box_count, dtype=torch.int32, device=device)
        batch_grid, class_grid, box_grid = torch.meshgrid(
            batch_ids, class_ids, box_ids, indexing="ij"
        )
        return torch.stack((batch_grid, class_grid, box_grid), dim=-1)

    values = torch.as_tensor(index_id, device=device).to(dtype=torch.int32)
    if values.shape[-1] == 4:
        return torch.stack(
            (values[..., 0], values[..., 1], values[..., 2] * 1000 + values[..., 3]),
            dim=-1,
        )
    return values[..., :3]


def _class_limit(max_boxes_size, batch_count, class_count, max_output_size):
    task_count = batch_count * class_count
    limit = (max_boxes_size + task_count - 1) // task_count if task_count else 0
    if max_output_size is not None:
        limit = min(limit, max(int(_scalar(max_output_size, 0)), 0))
    return limit


def _golden_impl(
    boxes,
    scores,
    max_output_size=None,
    iou_threshold=None,
    score_threshold=None,
    index_id=None,
    *,
    center_point_box=0,
    max_boxes_size=0,
    **kwargs,
):
    """CPU golden using ONNX Runtime's native NonMaxSuppression."""
    reference = _OnnxNmsV7Reference(
        torch.as_tensor(boxes),
        torch.as_tensor(scores),
        max_output_size,
        iou_threshold,
        score_threshold,
        None if index_id is None else torch.as_tensor(index_id),
        center_point_box=center_point_box,
        max_boxes_size=max_boxes_size,
        **kwargs,
    )
    return [reference()[0].cpu().numpy()]


def non_max_suppression_v7_kernel_golden(*args, **kwargs):
    return _golden_impl(*args, **kwargs)


def non_max_suppression_v7_geir_golden(*args, **kwargs):
    return _golden_impl(*args, **kwargs)


class _OnnxNmsV7Reference:
    """Device-neutral ONNX Runtime NMS plus the public V7 output adapter.

    Input conversion, scalar preparation, index-map construction, Session
    creation, and I/O binding are intentionally outside ``__call__``.  The
    timed GPU call therefore contains no ``.to`` or host/device conversion.
    """

    def __init__(
        self,
        boxes,
        scores,
        max_output_size=None,
        iou_threshold=None,
        score_threshold=None,
        index_id=None,
        *,
        center_point_box=0,
        max_boxes_size=0,
        **kwargs,
    ):
        del kwargs
        if boxes.device != scores.device:
            raise RuntimeError("boxes and scores must be on the same device")
        if boxes.device.type not in ("cpu", "cuda"):
            raise RuntimeError("ONNX reference supports only CPU and CUDA tensors")

        self.output_capacity = int(max_boxes_size)
        self.boxes = boxes.to(dtype=torch.float32).contiguous()
        self.scores = scores.to(dtype=torch.float32).contiguous()
        batch_count, class_count, box_count = self.scores.shape
        limit = _class_limit(
            self.output_capacity,
            batch_count,
            class_count,
            max_output_size,
        )
        self.max_output = torch.tensor(
            limit, dtype=torch.int64, device=self.boxes.device
        )
        self.iou_threshold = torch.tensor(
            float(_scalar(iou_threshold, 0.0)),
            dtype=torch.float32,
            device=self.boxes.device,
        )
        self.score_threshold = torch.tensor(
            float(_scalar(score_threshold, 0.0)),
            dtype=torch.float32,
            device=self.boxes.device,
        )
        self.indices = _reference_index_map(
            index_id,
            batch_count,
            class_count,
            box_count,
            self.boxes.device,
        )
        self.output = torch.full(
            (self.output_capacity, 3),
            -1,
            dtype=torch.int32,
            device=self.boxes.device,
        )
        self.binding = None
        if self.output_capacity == 0:
            return

        device_type = self.boxes.device.type
        device_id = self.boxes.device.index or 0
        stream = (
            torch.cuda.current_stream(device_id).cuda_stream
            if device_type == "cuda"
            else 0
        )
        self.session = _onnx_session(center_point_box, device_type, device_id, stream)
        self.binding = self.session.io_binding()
        for name, tensor in (
            ("boxes", self.boxes),
            ("scores", self.scores),
            ("max_output", self.max_output),
            ("iou_threshold", self.iou_threshold),
            ("score_threshold", self.score_threshold),
        ):
            self.binding.bind_ortvalue_input(name, _ortvalue_from_torch(tensor))
        self.binding.bind_output("selected_indices", device_type, device_id)

    def __call__(self):
        if self.output_capacity == 0:
            return [self.output]

        self.session.run_with_iobinding(self.binding)
        selected = _torch_from_ortvalue(self.binding.get_outputs()[0])
        count = min(selected.shape[0], self.output_capacity)
        self.output.fill_(-1)
        if count:
            chosen = selected[:count]
            mapped = self.indices[chosen[:, 0], chosen[:, 1], chosen[:, 2]]
            self.output[:count].copy_(mapped)
        return [self.output]


class NonMaxSuppressionV7KernelThirdParty(_OnnxNmsV7Reference):
    """ONNX Runtime CUDA competitor exposed through TTK's torch provider."""

    def __init__(self, boxes, scores, *args, **kwargs):
        if not boxes.is_cuda or not scores.is_cuda:
            raise RuntimeError(
                "NonMaxSuppressionV7 third-party benchmark requires CUDA"
            )
        super().__init__(boxes, scores, *args, **kwargs)


class NonMaxSuppressionV7GeirThirdParty(NonMaxSuppressionV7KernelThirdParty):
    """The same ONNX Runtime CUDA competitor for the GEIR route."""


class NonMaxSuppressionV7KernelSpec:
    """Kernel route specification used by the TTK Kernel/GEIR lookup key."""

    golden = staticmethod(non_max_suppression_v7_kernel_golden)
    # The key stays "torch" because TTK uses that provider for CUDA tensor
    # transport/profiling; the implementation above executes ONNX Runtime CUDA.
    third_party = {"torch": NonMaxSuppressionV7KernelThirdParty}
    tolerance = KERNEL_OUTPUT_TOLERANCE


class NonMaxSuppressionV7GeirSpec:
    """Standalone GEIR route specification for route-level review."""

    golden = staticmethod(non_max_suppression_v7_geir_golden)
    third_party = {"torch": NonMaxSuppressionV7GeirThirdParty}
    tolerance = GEIR_OUTPUT_TOLERANCE


def _self_test():
    mapped = _reference_index_map(
        torch.tensor(
            [[[[float("nan"), float("inf"), float("-inf"), float("nan")]]]],
            dtype=torch.float16,
        ),
        1,
        1,
        1,
        torch.device("cpu"),
    )
    torch.testing.assert_close(
        mapped,
        torch.full((1, 1, 1, 3), torch.iinfo(torch.int32).min, dtype=torch.int32),
    )


# Not registered because these pathways are not delivered by this operator:
# - ACLNN: CMakeLists.txt declares ACLNNTYPE aclnn_exclude.
# - torch/TF E2E: no framework API binding is provided in this change.
# - ONNX: TTK has no standalone ONNX execution path.  The delivered Kernel and
#   GEIR references nevertheless use ONNX Runtime's native NMS through TTK's
#   CPU golden and torch/CUDA provider boundaries.


if __name__ == "__main__":
    _self_test()
