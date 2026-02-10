# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_LIBRARY_OUTPUT_DIRECTOR ${PROJECT_SOURCE_DIR}/build)

file(GLOB_RECURSE OP_API_FILES "*.cpp")

add_library(opapi_math SHARED ${OP_API_FILES})

target_include_directories(opapi_math
  PRIVATE
  ${OPAPI_INCLUDE}
)

target_link_libraries(opapi_math
  PRIVATE
  $<BUILD_INTERFACE:intf_pub_cxx17>
)

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(SUB_DIR ${CURRENT_DIRS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
            add_subdirectory(${SUB_DIR})
        endif()
    endforeach()
endif()