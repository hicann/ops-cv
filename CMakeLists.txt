# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(PKG_NAME cv)
project(${PKG_NAME} VERSION 1.0.0)

set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")

if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(STATUS "compile project with library")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops-cv with cann pkg" ON)
else()
  message(STATUS "compile project with src")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops-cv with cann source" OFF)
endif()

if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()
set(OPS_CV_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(OPS_CV_VERSION_FILE ${CMAKE_CURRENT_BINARY_DIR}/version/version.info)
add_custom_command(
    OUTPUT ${OPS_CV_VERSION_FILE}
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/update_version_info/update_version_info.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/version.info
    ${OPS_CV_VERSION_FILE}
    DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/version.info
    VERBATIM
)

add_custom_target(generate_ops_cv_version_file ALL
    DEPENDS ${OPS_CV_VERSION_FILE}
)

# 外部传参
option(ENABLE_ASAN "Enable asan" OFF)
option(ENABLE_VALGRIND "Enable valgrind" OFF)
option(ENABLE_TEST "Enable test" OFF)
option(ENABLE_UT_EXEC "Enable exec ut" OFF)
option(ENABLE_BINARY "Enable build binary" OFF)
option(ENABLE_CUSTOM "Enable build custom" OFF)
option(ENABLE_PACKAGE "Enable build package" OFF)
option(ENABLE_EXPERIMENTAL "Enable experimental module" OFF)
option(ENABLE_COVERAGE "Enable coverage" OFF)
option(ENABLE_MSSANITIZER "Enable mssanitizer mode" OFF)
option(ENABLE_OOM "Enable oom mode" OFF)
option(DISABLE_AICPU "Disable aicpu" OFF)

option(OP_HOST_UT "Enable ophost ut" OFF)
option(OP_API_UT "Enable opapi ut" OFF)
option(OP_GRAPH_UT "Enable graph ut" OFF)
option(OP_KERNEL_UT "Enable kernel ut" OFF)
option(OP_KERNEL_AICPU_UT "Enable aicpu kernel ut" OFF)
option(UT_TEST_ALL "Enable all ut" OFF)
set(BUILD_MODE "" CACHE STRING "build mode -O0/O1/O2/O3")
set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "soc that need to be compiled")
set(ASCEND_ALL_COMPUTE_UNIT "ascend031;ascend035;ascend310b;ascend310p;ascend610lite;ascend630;ascend910;ascend910b;ascend910_93;ascend950;mc62cm12a;kirinx90;kirin9030" CACHE STRING "all soc list")
set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/build/third_party CACHE STRING "cann third party lib path")
set(ASCEND_OP_NAME "" CACHE STRING "operators that need to be compiled")
set(VENDOR_NAME "custom" CACHE STRING "vendor name")

if(ENABLE_TEST)
  set(DEFAULT_BUILD_TYPE "Debug")
else()
  set(DEFAULT_BUILD_TYPE "Release")
endif ()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the build type: Release/Debug" FORCE)
endif ()

if(ENABLE_TEST)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the build type: Release/Debug" FORCE)
endif ()

if("${BUILD_MODE}" STREQUAL "-O0" OR "${BUILD_MODE}" STREQUAL "-O1" OR
   "${BUILD_MODE}" STREQUAL "-O2" OR "${BUILD_MODE}" STREQUAL "-O3"
  )
  set(COMPILE_OP_MODE ${BUILD_MODE})
else()
  if(ENABLE_TEST)
    set(COMPILE_OP_MODE "-O0")
  else()
    set(COMPILE_OP_MODE "-O2")
  endif()
endif()

if(ENABLE_TEST)
  set(COMPILE_OP_MODE "${COMPILE_OP_MODE} -g")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "cmake build type")
else ()
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "cmake build type")
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Cmake Compile配置
set(CMAKE_CXX_STANDARD 17 CACHE STRING "c++17 is needed for this project")
include(cmake/dependencies.cmake)
include(cmake/variables.cmake)
include(cmake/opbuild.cmake)
include(cmake/third_party/eigen.cmake)
include(cmake/third_party/abseil-cpp.cmake)
include(cmake/third_party/ascend_protobuf.cmake)
include(cmake/third_party/json.cmake)
include(cmake/func.cmake)
include(cmake/intf_pub_linux.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ut.cmake)

add_subdirectory(common)
if(ENABLE_EXPERIMENTAL)
    message(STATUS "Build experimental module")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/experimental/image")
      add_subdirectory(experimental/image)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/experimental/objdetect")
      add_subdirectory(experimental/objdetect)
    endif()
else()
  add_subdirectory(image)
  add_subdirectory(objdetect)
  add_subdirectory(common/src/framework)
endif()

if(ENABLE_TEST)
  add_subdirectory(tests/ut)
endif()

if("${ASCEND_OP_NAME}" STREQUAL "add_example"
   OR "${ASCEND_OP_NAME}" STREQUAL "add_example_aicpu"
   OR "${ASCEND_OP_NAME}" STREQUAL "add_example;add_example_aicpu"
   OR "${ASCEND_OP_NAME}" STREQUAL "add_example_aicpu;add_example")
  add_subdirectory(examples)
endif()

check_compiled_ops()

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  include(cmake/gen_ops_info.cmake)
  gen_ops_info_and_python()

  include(cmake/symbol.cmake)
  if(ENABLE_CUSTOM)
    # custom package library setting and install
    gen_cust_symbol()
  else()
    gen_norm_symbol()
  endif()
endif()

if(ENABLE_PACKAGE)
  include(cmake/package.cmake)

  if(ENABLE_CUSTOM)
    pack_custom()
  else()
    pack_built_in()
  endif()
  if(ENABLE_STATIC)
    include(cmake/static.cmake)
  endif()
endif()