#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import json
import os
import stat
import sys
from operator import truediv

required_op_input_info_keys = ["paramType", "name"]
required_op_output_info_keys = ["paramType", "name"]
param_type_valid_value = ["dynamic", "optional", "required"]
required_attr_key = ["type", "value", "paramType"]
aicpu_infershape_subtype_valid_value = ["1", "2", "3", "4"]
aicpu_ops_flag_valid_value = ["OPS_FLAG_OPEN", "OPS_FLAG_CLOSE"]
# workspaceSize is valid when it is within the inclusive range [100, 500]
AICPU_WORKSPACE_SIZE_MIN = 100
AICPU_WORKSPACE_SIZE_MAX = 500
aicpu_kernel_so_valid_value = [
    "libmath_aicpu_kernels.so",
    "libcv_aicpu_kernels.so",
    "libnn_aicpu_kernels.so",
    "libtransformer_aicpu_kernels.so",
]


def parse_ini_files(ini_files):
    """
    parse ini files to json
    Parameters:
    ----------------
    ini_files:input file list
    return:ops_info
    ----------------
    """
    tbe_ops_info = {}
    for ini_file in ini_files:
        if not os.path.exists(ini_file):
            print("ini file {} not exists!".format(ini_file))
            continue
        parse_ini_to_obj(ini_file, tbe_ops_info)
    return tbe_ops_info


def parse_ini_to_obj(ini_file, tbe_ops_info):
    """
    parse ini file to json obj
    Parameters:
    ----------------
    ini_file:ini file path
    tbe_ops_info:ops_info
    ----------------
    """
    with open(ini_file) as ini_file:
        lines = ini_file.readlines()
        op = {}
        op_name = ""
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("["):
                op_name = line[1:-1]
                op = {}
                tbe_ops_info[op_name] = op
            else:
                key1 = line[: line.index("=")].strip()
                key2 = line[line.index("=") + 1:].strip()
                key1_0, key1_1 = key1.split(".")
                if not key1_0 in op:
                    op[key1_0] = {}
                if key1_1 in op[key1_0]:
                    raise RuntimeError(
                        "Op:" + op_name + " " + key1_0 + " " + key1_1 + " is repeated!"
                )
                op[key1_0][key1_1] = key2


def is_cpu_engine(op_engine):
    """判断是否为 AICPU 引擎"""
    return op_engine == "DNN_VM_AICPU"


def check_required_keys(op_info, required_keys, op_engine):
    """检查必需字段是否存在，AICPU 算子跳过 paramType 检查"""
    missing_keys = []
    for required_key in required_keys:
        # AICPU 算子不需要 paramType 字段
        if is_cpu_engine(op_engine) and required_key == "paramType":
            continue
        if required_key not in op_info:
            missing_keys.append(required_key)
    return missing_keys


def check_param_type(op_info, op_engine, error_message):
    """检查 paramType 是否合法，AICPU 算子跳过检查"""
    if is_cpu_engine(op_engine):
        return True
    if op_info.get("paramType") not in param_type_valid_value:
        print(error_message)
        return False
    return True


def check_infershape_subtype(op_info):
    """检查 AICPU 的 subTypeOfInferShape 字段是否合法"""
    if op_info.get("subTypeOfInferShape") not in aicpu_infershape_subtype_valid_value:
        return False
    return True


def check_ops_flag(op_info):
    """检查 AICPU 的 opsFlag 字段是否合法"""
    if op_info.get("opsFlag") not in aicpu_ops_flag_valid_value:
        return False
    return True


def check_workspace_size(op_info):
    """检查 AICPU 的 workspaceSize 是否在 [100, 500] 范围内"""
    workspace_size = op_info.get("workspaceSize")
    try:
        workspace_size = int(workspace_size)
    except (TypeError, ValueError):
        return False
    return AICPU_WORKSPACE_SIZE_MIN <= workspace_size <= AICPU_WORKSPACE_SIZE_MAX


def check_opkernel_so(op_info):
    """检查 kernelSo 是否在白名单中"""
    if op_info.get("kernelSo") not in aicpu_kernel_so_valid_value:
        return False
    return True


def validate_io_info(op_key, op_info_key, op_info, op_engine, required_keys):
    """校验输入或输出信息"""
    is_valid = True
    missing_keys = check_required_keys(op_info, required_keys, op_engine)
    if missing_keys:
        print(f"op: {op_key} {op_info_key} missing: {','.join(missing_keys)}")
        is_valid = False

    error_message = f"op: {op_key} {op_info_key} paramType not valid, valid key:{param_type_valid_value}"
    if not check_param_type(op_info, op_engine, error_message):
        is_valid = False

    return is_valid


def validate_optional_info(op_key, op_info_key, op_info):
    """校验 AICPU 的可选字段"""
    is_valid = True
    if "subTypeOfInferShape" in op_info and not check_infershape_subtype(op_info):
        print(f"op: {op_key} {op_info_key} infershape_subtype not valid"
              f" valid key: {aicpu_infershape_subtype_valid_value}")
        is_valid = False
    if "opsFlag" in op_info and not check_ops_flag(op_info):
        print(f"op: {op_key} {op_info_key} opsFlag not valid"
              f" valid key: {aicpu_ops_flag_valid_value}")
        is_valid = False
    if "workspaceSize" in op_info and not check_workspace_size(op_info):
        print(f"op: {op_key} {op_info_key} workspaceSize not valid"
              f" valid range: [{AICPU_WORKSPACE_SIZE_MIN}, {AICPU_WORKSPACE_SIZE_MAX}]")
        is_valid = False
    if "kernelSo" in op_info and not check_opkernel_so(op_info):
        print(f"op: {op_key} {op_info_key} kernelSo not valid"
              f" valid key: {aicpu_kernel_so_valid_value}")
        is_valid = False
    return is_valid


def check_op_info(tbe_ops):
    """校验算子信息"""
    print("\n\n==============check valid for ops info start==============")

    io_required_keys_map = {
        "input": required_op_input_info_keys,
        "output": required_op_output_info_keys,
    }
    is_valid = True
    for op_key in tbe_ops:
        op = tbe_ops[op_key]
        op_engine = op.get("opInfo", {}).get("engine", "") if isinstance(op, dict) else ""

        for op_info_key in op:
            op_info_flag = False
            io_prefix = ""
            if op_info_key.startswith("input"):
                io_prefix = "input"
            elif op_info_key.startswith("output"):
                io_prefix = "output"
            elif op_info_key.startswith("opInfo"):
                op_info_flag = True
            else:
                continue

            op_info = op[op_info_key]
            if io_prefix != "" and not validate_io_info(
                op_key, op_info_key, op_info, op_engine, io_required_keys_map.get(io_prefix)
            ):
                is_valid = False
            if is_cpu_engine(op_engine) and op_info_flag and not validate_optional_info(op_key, op_info_key, op_info):
                is_valid = False

    print("==============check valid for ops info end================\n\n")
    return is_valid


def write_json_file(tbe_ops_info, json_file_path):
    """
    Save info to json file
    Parameters:
    ----------------
    tbe_ops_info: ops_info
    json_file_path: json file path
    ----------------
    """
    json_file_real_path = os.path.realpath(json_file_path)
    with open(json_file_real_path, "w") as f:
        # Only the owner and group have rights
        os.chmod(json_file_real_path, stat.S_IWGRP + stat.S_IWUSR + stat.S_IRGRP + stat.S_IRUSR)
        json.dump(tbe_ops_info, f, sort_keys=True, indent=4, separators=(',', ':'))
    print("Compile op info cfg successfully.")


def parse_ini_to_json(ini_file_paths, outfile_path):
    """
    parse ini files to json file
    Parameters:
    ----------------
    ini_file_paths: list of ini file path
    outfile_path: output file path
    ----------------
    """
    tbe_ops_info = parse_ini_files(ini_file_paths)
    if not check_op_info(tbe_ops_info):
        print("Compile op info cfg failed.")
        return False
    write_json_file(tbe_ops_info, outfile_path)
    return True


if __name__ == '__main__':
    args = sys.argv

    output_file_path = "tbe_ops_info.json"
    ini_file_path_list = []

    for arg in args:
        if arg.endswith("ini"):
            ini_file_path_list.append(arg)
            output_file_path = arg.replace(".ini", ".json")
        if arg.endswith("json"):
            output_file_path = arg

    if len(ini_file_path_list) == 0:
        ini_file_path_list.append("tbe_ops_info.ini")

    if not parse_ini_to_json(ini_file_path_list, output_file_path):
        sys.exit(1)
    sys.exit(0)


