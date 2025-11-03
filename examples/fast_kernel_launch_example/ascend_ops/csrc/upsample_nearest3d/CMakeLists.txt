# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

message(STATUS "BUILD_TORCH_OPS ON in upsample_nearest3d")
# UPSAMPLE_NEAREST3D operation sources
file(GLOB UPSAMPLE_NEAREST3D_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# set(UPSAMPLE_NEAREST3D_SOURCES ${UPSAMPLE_NEAREST3D_CPP_HEADER} ${UPSAMPLE_NEAREST3D_CPP_SOURCES} ${UPSAMPLE_NEAREST3D_NPU_SOURCES})
set(UPSAMPLE_NEAREST3D_SOURCES ${UPSAMPLE_NEAREST3D_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${UPSAMPLE_NEAREST3D_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(upsample_nearest3d_objects OBJECT ${UPSAMPLE_NEAREST3D_SOURCES})

target_compile_options(upsample_nearest3d_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(upsample_nearest3d_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()
