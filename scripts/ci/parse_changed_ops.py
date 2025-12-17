# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------


import os
import sys
import re
import logging

NEW_OPS_PATH = [
    "image",
    "objdetect"
    # 添加更多算子路径
]


class OperatorChangeInfo:
    def __init__(self, changed_operators=None, operator_file_map=None):
        self.changed_operators = [] if changed_operators is None else changed_operators
        self.operator_file_map = {} if operator_file_map is None else operator_file_map


def extract_operator_name(file_path):
    clean_path = file_path.lstrip('/')
    path_parts = clean_path.split('/')
    default_name = ''

    if len(path_parts) >= 2:
        domain = path_parts[0]
        operator_name = path_parts[1]

        if domain in NEW_OPS_PATH:
            return operator_name
    return default_name


def get_operator_info_from_ci(changed_file_info_from_ci):
    """
    get operator change info from ci, ci will write `git diff > /or_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or OperatorChangeInf
    """
    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        logging.error("[ERROR] change file is not exist, can not get file change info in this pull request.")
        return None
    with open(or_file_path) as or_f:
        lines = or_f.readlines()
        changed_operators = set()
        operator_file_map = {}

        get_change_operators(changed_operators, lines, operator_file_map)
    return OperatorChangeInfo(changed_operators=list(changed_operators), operator_file_map=operator_file_map)


def get_change_operators(changed_operators, lines, operator_file_map):
    for line in lines:
        line = line.strip()
        ext = os.path.splitext(line)[-1].lower()
        if ext in (".md",):
            continue
        operator_name = extract_operator_name(line)
        if operator_name:
            changed_operators.add(operator_name)
            if operator_name not in operator_file_map:
                operator_file_map[operator_name] = []
            operator_file_map[operator_name].append(line)


def get_change_ops_list(changed_file_info_from_ci):
    ops_change_info = get_operator_info_from_ci(changed_file_info_from_ci)
    if not ops_change_info:
        logging.info("[INFO] not found ops change info, run all c++.")
        return None

    return ";".join(ops_change_info.changed_operators)


if __name__ == '__main__':
    logging.info(get_change_ops_list(sys.argv[1]))
