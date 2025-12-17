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

OP_API_UT = "OP_API_UT"
OP_HOST_UT = "OP_HOST_UT"
OP_GRAPH_UT = "OP_GRAPH_UT"
OP_KERNEL_UT = "OP_KERNEL_UT"
ALL_UT = "ALL_UT"

NEW_OPS_PATH = ["image", "objdetect"]   # 添加更多算子路径
COMM_FILES = ["tests", "common"]        # 添加更多算子路径


class FileChangeInfo:
    def __init__(self, op_api_changed_files=None, op_host_changed_files=None, op_graph_changed_files=None,
                 op_kernel_changed_files=None, comm_changed_files=None):
        self.op_api_changed_files = [] if op_api_changed_files is None else op_api_changed_files
        self.op_host_changed_files = [] if op_host_changed_files is None else op_host_changed_files
        self.op_graph_changed_files = [] if op_graph_changed_files is None else op_graph_changed_files
        self.op_kernel_changed_files = [] if op_kernel_changed_files is None else op_kernel_changed_files
        self.comm_changed_files = [] if comm_changed_files is None else comm_changed_files


def get_file_change_info_from_ci(changed_file_info_from_ci):
    """
    get file change info from ci, ci will write `git diff > /or_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or FileChangeInfo
    """
    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        logging.error("[ERROR] change file is not exist, can not get file change info in this pull request.")
        return None
    with open(or_file_path) as or_f:
        lines = or_f.readlines()
        op_api_changed_files = []
        op_host_changed_files = []
        op_graph_changed_files = []
        op_kernel_changed_files = []
        comm_changed_files = []

        host_pattern = re.compile(rf"({'|'.join(NEW_OPS_PATH)})/.*/op_host/.*\.(cc|cpp|h)$")
        api_pattern = re.compile(rf"({'|'.join(NEW_OPS_PATH)})/.*/op_api/.*\.(cc|cpp|h)$")
        kernel_pattern = re.compile(rf"({'|'.join(NEW_OPS_PATH)})/.*/op_kernel/.*\.(cc|cpp|h)$")
        graph_pattern = re.compile(rf"({'|'.join(NEW_OPS_PATH)})/.*/op_graph/.*\.(cc|cpp|h)$")
        host_test_pattern = re.compile(rf"({'|'.join(NEW_OPS_PATH)})/.*/tests/ut/op_host/.*\.(cc|cpp|txt)$")
        api_test_pattern = re.compile(rf"({'|'.join(NEW_OPS_PATH)})/.*/tests/ut/op_api/.*\.(cc|cpp|txt|py)$")
        graph_test_pattern = re.compile(rf"({'|'.join(NEW_OPS_PATH)})/.*/tests/ut/op_graph/.*\.(cc|cpp|txt)$")
        kernel_test_pattern = re.compile(rf"({'|'.join(NEW_OPS_PATH)})/.*/tests/ut/op_kernel/.*\.(cc|cpp|txt)$")
        comm_files_pattern = re.compile(rf"^({'|'.join(COMM_FILES)})")

        for line in lines:
            line = line.strip()
            ext = os.path.splitext(line)[-1].lower()
            if ext in (".md",):
                continue
            if api_pattern.match(line) or api_test_pattern.match(line):
                op_api_changed_files.append(line)
            elif host_pattern.match(line) or host_test_pattern.match(line):
                op_host_changed_files.append(line)
            elif kernel_pattern.match(line) or kernel_test_pattern.match(line):
                op_kernel_changed_files.append(line)
            elif graph_pattern.match(line) or graph_test_pattern.match(line):
                op_graph_changed_files.append(line)
            elif comm_files_pattern.match(line):
                comm_changed_files.append(line)
    return FileChangeInfo(op_host_changed_files=op_host_changed_files,
                          op_api_changed_files=op_api_changed_files,
                          op_graph_changed_files=op_graph_changed_files, 
                          op_kernel_changed_files=op_kernel_changed_files,
                          comm_changed_files=comm_changed_files)


def get_change_relate_ut_dir_list(changed_file_info_from_ci):
    file_change_info = get_file_change_info_from_ci(changed_file_info_from_ci)
    if not file_change_info:
        logging.info("[INFO] not found file change info, run all c++.")
        return None

    def _get_relate_ut_list_by_file_change():
        relate_ut = set()
        if len(file_change_info.op_host_changed_files) > 0:
            relate_ut.add(OP_HOST_UT)
        if len(file_change_info.op_api_changed_files) > 0:
            relate_ut.add(OP_API_UT)
        if len(file_change_info.op_graph_changed_files) > 0:
            relate_ut.add(OP_GRAPH_UT)
        if len(file_change_info.op_kernel_changed_files) > 0:
            relate_ut.add(OP_KERNEL_UT)
        if len(file_change_info.comm_changed_files) > 0:
            relate_ut.add(ALL_UT)
        return relate_ut

    try:
        relate_uts = _get_relate_ut_list_by_file_change()
    except BaseException as e:
        logging.error(e.args)
        return None
    return str(relate_uts)


if __name__ == '__main__':
    logging.info(get_change_relate_ut_dir_list(sys.argv[1]))
