#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

logging() {
  echo "[INFO] $@"
}

mk_dir() {
  local create_dir="$1"
  mkdir -pv "${create_dir}"
  logging "Created ${create_dir}"
}

# 获取 lcov 版本号
get_lcov_version() {
  lcov --version 2>/dev/null | grep -oP 'LCOV version \K[0-9]+\.[0-9]+' | head -1
}

# 根据 lcov 版本构建 --ignore-errors 参数
build_ignore_errors_opt() {
  local version="$1"
  # lcov 2.0+ supports: mismatch, gcov, source, empty, etc.
  # lcov 1.x supports: gcov, inconsistent, etc.
  
  if [[ "${version}" =~ ^2\. ]]; then
    echo "--ignore-errors mismatch,gcov,source,empty"
  else
    # lcov 1.14 and earlier
    echo "--ignore-errors gcov"
  fi
}

# using lcov to generate coverage for cpp files
generate_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  if [[ -z "${_source_dir}" ]]; then
    logging "directory required to find the .da files"
    exit 1
  fi

  if [[ ! -d "${_source_dir}" ]]; then
    logging "directory does not exist, please check ${_source_dir}"
    exit 1
  fi

  if [[ -z "${_coverage_file}" ]]; then
    _coverage_file="coverage.info"
    logging "using default file name to generate coverage"
  fi

  which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mk_dir "${_path_to_gen}"
  fi

  # 根据 lcov 版本选择合适的忽略错误选项
  local _lcov_version=$(get_lcov_version)
  local _ignore_opt=$(build_ignore_errors_opt "${_lcov_version}")

  # 生成基础覆盖率文件
  lcov --capture \
    --directory "${_source_dir}" \
    --output-file "${_coverage_file}" \
    --rc geninfo_unexecuted_blocks=1 \
    ${_ignore_opt}

  logging "generated coverage file ${_coverage_file}"
}

# filter out some unused directories or files
filter_coverage() {
  local _coverage_file="$1"
  local _filtered_file="$2"

  if [[ ! -f "${_coverage_file}" ]]; then
    logging "coverage data file required"
    exit 1
  fi

  # 根据 lcov 版本选择合适的忽略错误选项
  local _lcov_version=$(get_lcov_version)
  local _ignore_opt=""
  if [[ "${_lcov_version}" =~ ^2\. ]]; then
    _ignore_opt="--ignore-errors unused,inconsistent,empty"
  else
    # lcov 1.14 and earlier
    _ignore_opt="--ignore-errors inconsistent"
  fi

  # 移除不需要的库文件 (Ascend, /usr/include 等)
  lcov --remove "${_coverage_file}" "${ASCEND_PARENT_PATH}/*" \
    '/usr/include/*' \
    '*/third_party/*' \
    '*/common/*' \
    '*/tests/*' -o "${_filtered_file}" ${_ignore_opt}
}

# generate html report
generate_html() {
  local _filtered_file="$1"
  local _out_path="$2"

  which genhtml >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "genhtml is required to generate coverage html report, please install"
    exit 1
  fi

  if [[ ! -d "${_out_path}" ]]; then
    mk_dir "${_out_path}"
  fi

  # 根据 lcov 版本选择合适的忽略错误选项
  local _lcov_version=$(get_lcov_version)
  local _ignore_opt=""
  if [[ "${_lcov_version}" =~ ^2\. ]]; then
    _ignore_opt="--ignore-errors empty"
  fi

  genhtml "${_filtered_file}" -o "${_out_path}" ${_ignore_opt}
}

if [[ $# -ne 3 ]]; then
  logging "Usage: $0 DIR COV_FILE OUT_PATH"
  exit 1
fi

_src="$1"
_cov_file="$2"
_out="$3"

if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  logging "ASCEND_HOME_PATH is not set"
  exit 1
fi

ASCEND_PARENT_PATH=$(dirname "${ASCEND_HOME_PATH}")
if [[ -z "${ASCEND_PARENT_PATH}" ]]; then
  logging "ASCEND_HOME_PATH is not set"
  exit 1
fi

generate_coverage "${_src}" "${_cov_file}"
filter_coverage "${_cov_file}" "${_cov_file}_filtered"
generate_html "${_cov_file}_filtered" "${_out}"
