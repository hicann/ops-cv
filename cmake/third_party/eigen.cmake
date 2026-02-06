# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(EIGEN_DOWNLOAD_PATH ${CANN_3RD_LIB_PATH}/pkg)
set(EIGEN_VERSION_PKG eigen-3.4.0.tar.gz)

if (EXISTS "${CANN_3RD_LIB_PATH}/eigen/CMakeLists.txt" AND NOT FORCE_REBUILD_CANN_3RD)
  message("[ThirdPartyLib][eigen] eigen found, and not force rebuild cann third_party")
  set(SOURCE_DIR "${CANN_3RD_LIB_PATH}/eigen")
else()
  set(REQ_URL "https://gitcode.com/cann-src-third-party/eigen/releases/download/3.4.0/${EIGEN_VERSION_PKG}")
  set(EIGEN_ARCHIVE ${CANN_3RD_LIB_PATH}/pkg/${EIGEN_VERSION_PKG})
  file(MAKE_DIRECTORY ${EIGEN_DOWNLOAD_PATH})

  # Search in CANN_3RD_LIB_PATH and move to pkg if found
  if(EXISTS ${CANN_3RD_LIB_PATH}/${EIGEN_VERSION_PKG} AND NOT EXISTS ${EIGEN_ARCHIVE})
      message(STATUS "[ThirdPartyLib][eigen] Found egien archive in ${CANN_3RD_LIB_PATH}, moving to pkg")
      file(RENAME ${CANN_3RD_LIB_PATH}/${EIGEN_VERSION_PKG} ${EIGEN_ARCHIVE})
  endif()

  if(EXISTS ${EIGEN_ARCHIVE})
      message(STATUS "[ThirdPartyLib][eigen] Found egien archive at ${EIGEN_ARCHIVE}")
      set(EIGEN_URL "file://${EIGEN_ARCHIVE}")
  else()
      set(EIGEN_URL ${REQ_URL})
      message(STATUS "[ThirdPartyLib][eigen] not found eigen ,need to download ${MAKESELF_NAME} from ${REQ_URL}")
  endif()

  include(ExternalProject)
  ExternalProject_Add(external_eigen_cv
    TLS_VERIFY        OFF
    URL               ${EIGEN_URL}
    DOWNLOAD_DIR      ${EIGEN_DOWNLOAD_PATH}
    SOURCE_DIR        ${CANN_3RD_LIB_PATH}/eigen
    PREFIX            third_party
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
  )
  ExternalProject_Get_Property(external_eigen_cv SOURCE_DIR)
endif()


add_library(EigenCv INTERFACE)
target_compile_options(EigenCv INTERFACE -w)

set_target_properties(EigenCv PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${SOURCE_DIR}"
)
add_dependencies(EigenCv external_eigen_cv)

add_library(Eigen3::EigenCv ALIAS EigenCv)