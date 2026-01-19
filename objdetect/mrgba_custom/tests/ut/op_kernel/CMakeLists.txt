# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_library(mrgba_custom SHARED
           ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/mrgba_custom.cpp
            ./test_mrgba_custom.cpp
)

target_compile_options(mrgba_custom PRIVATE -g -include ${CMAKE_CURRENT_SOURCE_DIR}/test_mrgba_custom.h)

target_link_libraries(mrgba_custom PRIVATE
                      $<BUILD_INTERFACE:intf_llt_pub_asan>
                      -Wl,--whole-archive
                      -Wl,--no-as-needed
                      ${PRIVATE_L}
                      -Wl,--as-needed
                      -Wl,--no-whole-archive
)


