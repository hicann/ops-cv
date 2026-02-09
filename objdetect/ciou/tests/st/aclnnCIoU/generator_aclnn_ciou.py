# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import random
from typing import Union, List
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig,CaseConfig

BBOXES_INDEX = 0
GTBOXES_INDEX = 1
TRANS_INDEX = 2
ISCROSS_INDEX= 3
MODE_INDEX = 4
ATANSUBFLAG_INDEX = 5


@GENERATOR_REGISTRY.register("gen_aclnn_ciou")
class DefaultGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.shape = []
        self.dtype = None
        self.dim = None
        

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        case_config.inputs[GTBOXES_INDEX].dtype = case_config.inputs[BBOXES_INDEX].dtype
        case_config.inputs[BBOXES_INDEX].shape[0] = 4
        case_config.inputs[BBOXES_INDEX].shape[1] = random.randint(1, 4096)
        case_config.inputs[GTBOXES_INDEX].shape = case_config.inputs[BBOXES_INDEX].shape
        return case_config