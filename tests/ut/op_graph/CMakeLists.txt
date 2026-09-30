# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

if(UT_TEST_ALL OR OP_GRAPH_UT)
  set(OP_GRAPH_UT_EXE ${PKG_NAME}_op_graph_ut)

  add_op_graph_ut_modules(${OP_GRAPH_MODULE_NAME})

  add_executable(${OP_GRAPH_UT_EXE}
      test_op_graph_main.cpp
  )
  target_compile_definitions(${OP_GRAPH_UT_EXE} PRIVATE LOG_CPP)
  target_compile_options(${OP_GRAPH_UT_EXE} PUBLIC -fPIE -fno-access-control)
  target_include_directories(${OP_GRAPH_UT_EXE} PRIVATE
      ${ASCEND_DIR}/pkg_inc
      ${ASCEND_DIR}/include
      ${ASCEND_DIR}/include/ge
  )
  target_link_directories(${OP_GRAPH_UT_EXE} PRIVATE
      ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
      ${ASCEND_DIR}/lib64
  )
  target_link_libraries(${OP_GRAPH_UT_EXE} PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      -Wl,--whole-archive
          ${OP_GRAPH_MODULE_NAME}_static_lib
      -Wl,--no-whole-archive
      -Wl,--no-as-needed
          metadef
          unified_dlog
          ascendalog
          eager_style_graph_builder_base
          es_cv
          es_math
      -Wl,--as-needed
      register
      opp_registry
      graph
      platform
      GTest::gtest
      graph_base
      exe_graph
      ge_compiler
      ge_common
      error_manager
      rt2_registry
      c_sec
      json
  )

  if(ENABLE_UT_EXEC)
    if(ENABLE_ASAN)
      execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
        OUTPUT_VARIABLE LIBASAN_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE result
      )
      if(NOT result EQUAL 0)
        message(FATAL_ERROR "compiler not support asan, please disable asan")
      endif()
      execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE LIBSTDC_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE stdc_result
      )
      if(NOT stdc_result EQUAL 0)
        message(FATAL_ERROR "not find libstdc++.so, please confirm.")
      endif()
      add_custom_command(
        TARGET ${OP_GRAPH_UT_EXE}
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
          LD_PRELOAD=${LIBASAN_PATH}:${LIBSTDC_PATH}
          ASAN_OPTIONS=detect_leaks=0 $<TARGET_FILE:${OP_GRAPH_UT_EXE}>
        COMMENT "Run ops op_graph utest with asan"
      )
    else()
      add_custom_command(
        TARGET ${OP_GRAPH_UT_EXE}
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} $<TARGET_FILE:${OP_GRAPH_UT_EXE}>
        COMMENT "Run ops op_graph utest"
      )
    endif()
  endif()
endif()
