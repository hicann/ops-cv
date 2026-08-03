#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Build clean functions.

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    rm -rf "${BUILD_PATH}"/*
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    rm -rf "${BUILD_OUT_PATH}"/*
  fi
}

clean_build_binary() {
  if [ -d "${BUILD_PATH}/tbe" ]; then
    rm -rf "${BUILD_PATH}/tbe/"
  fi
  if [ -d "${BUILD_PATH}/autogen" ]; then
    rm -rf "${BUILD_PATH}/autogen/"
  fi
  if [ -d "${BUILD_PATH}/binary" ]; then
    rm -rf "${BUILD_PATH}/binary/"
  fi
  if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
    if [ -d "${BUILD_PATH}/static_library_files" ]; then
      rm -rf "${BUILD_PATH}/static_library_files/"
    fi
  fi
}
