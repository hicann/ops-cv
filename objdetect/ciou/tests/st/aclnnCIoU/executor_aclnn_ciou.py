# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import math
import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

@register("executor_aclnn_ciou")
class aclnnCIoUExecutor(BaseApi):
    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        np.random.seed(42)
        bBoxes = input_data.kwargs["bBoxes"]
        gtBoxes = input_data.kwargs["gtBoxes"]
        dtype = input_data.kwargs["bBoxes"].dtype
        size = input_data.kwargs["bBoxes"].shape[1]
        bBoxes[0] = torch.tensor(np.random.uniform(0, 640, size), dtype=dtype)
        bBoxes[1] = torch.tensor(np.random.uniform(0, 640, size), dtype=dtype)
        bBoxes[2] = torch.tensor(np.random.uniform(640, 1280, size), dtype=dtype)
        bBoxes[3] = torch.tensor(np.random.uniform(640, 1280, size), dtype=dtype)
        gtBoxes[0] = torch.tensor(np.random.uniform(0, 640, size), dtype=dtype)
        gtBoxes[1] = torch.tensor(np.random.uniform(0, 640, size), dtype=dtype)
        gtBoxes[2] = torch.tensor(np.random.uniform(640, 1280, size), dtype=dtype)
        gtBoxes[3] = torch.tensor(np.random.uniform(640, 1280, size), dtype=dtype)
        if self.device == "pyaclnn":
            input_data.kwargs["bBoxes"] = bBoxes.npu()
            input_data.kwargs["gtBoxes"] = gtBoxes.npu()
        else:
            input_data.kwargs["bBoxes"] = bBoxes.cpu()
            input_data.kwargs["gtBoxes"] = gtBoxes.cpu()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        bBoxes = input_data.kwargs["bBoxes"]
        gtBoxes = input_data.kwargs["gtBoxes"]
        trans = input_data.kwargs["trans"]
        isCross = input_data.kwargs["isCross"]
        mode = input_data.kwargs["mode"]
        return self.ciou(bBoxes, gtBoxes, trans, isCross, mode)

    def ciou(self, bBoxes, gtBoxes, trans, isCross, mode, eps=1e-9):
        box1 = bBoxes.T
        box2 = gtBoxes.T
        if trans:
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
        # Union Area
        if mode == "iou":
            union = w1 * h1 + w2 * h2 - inter + eps
        else:
            union = w2 * h2 + eps
        iou = inter / union
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        atanSub = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / math.pi ** 2) * (atanSub).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        overlap = iou - (rho2 / c2 + v * alpha)
        return overlap.T, atanSub.T