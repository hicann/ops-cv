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
# Option parsing, validation, and help display.

# print usage message
usage() {
  local specific_help="$1"

  if [[ -n "$specific_help" ]]; then
    case "$specific_help" in
      package)
        echo "Package Build Options:"
        echo $dotted_line
        echo "    --pkg                  Build run package with kernel bin"
        echo "    --pkg-type=<TYPE>      Specify package type(TYPE options: run/rpm/deb/all), Default: run"
        echo "    --static               Build static library package"
        echo "    --jit                  Build run package without kernel bin"
        echo "    --soc=soc_version      Compile for specified Ascend SoC"
        echo "    --vendor_name=name     Specify custom operator package vendor name"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --asan                 Enable ASAN (Address Sanitizer) on the host side"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo "    --build-type=<TYPE>"
        echo "                           Specify build type(TYPE options: Release/Debug), Default:Release"
        echo "    --experimental         Build experimental version"
        echo "    --cann_3rd_lib_path=<PATH>"
        echo "                           Set ascend third_party package install path, default ./third_party"
        echo "    --mssanitizer          Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
        echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
        echo "    --dump_cce             Dump kernel precompiled files"
        echo "    --bisheng_flags=flag1,flag2"
        echo "                           Specify bisheng compiler flags (comma-separated for multiple)"
        echo "    --kernel_template_input=args0,args1"
        echo "                           Specify kernel template input arguments(comma-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --pkg --soc=ascend910b --vendor_name=customize -j16 -O3"
        echo "    bash build.sh --pkg --pkg-type=deb --soc=ascend910b"
        echo "    bash build.sh --pkg --pkg-type=rpm --soc=ascend910b"
        echo "    bash build.sh --pkg --ops=grid_sample,crop_and_resize --build-type=Debug"
        echo "    bash build.sh --pkg --static --soc=ascend910b"
        echo "    bash build.sh --pkg --ops=grid_sample,crop_and_resize --build-type=Debug"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=grid_sample --mssanitizer"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=grid_sample --oom"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=grid_sample --dump_cce"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=grid_sample --bisheng_flags=ccec_g,oom"
        echo "    bash build.sh --pkg --experimental --soc=ascend950 --ops=grid_sample --kernel_template_input=0,1"
        return
        ;;
      opkernel)
        echo "Opkernel Build Options:"
        echo $dotted_line
        echo "    --opkernel             Build binary kernel"
        echo "    --soc=soc_version      Compile for specified Ascend SoC"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo "    --mssanitizer          Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
        echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
        echo "    --dump_cce             Dump kernel precompiled files"
        echo "    --bisheng_flags=flag1,flag2"
        echo "                           Specify bisheng compiler config (comma-separated for multiple)"
        echo "    --kernel_template_input=args0,args1"
        echo "                           Specify kernel template input arguments(comma-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=grid_sample"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=grid_sample --build-type=Debug"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=grid_sample --mssanitizer"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=grid_sample --oom"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=grid_sample --dump_cce"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=grid_sample --bisheng_flags=ccec_g,oom"
        echo "    bash build.sh --opkernel --soc=ascend950 --ops=grid_sample --kernel_template_input=0,1"
        return
        ;;
      opkernel_aicpu)
        echo "AICPU Opkernel Build Options:"
        echo $dotted_line
        echo "    --opkernel_aicpu       Build AICPU kernel"
        echo "    --soc=soc_version      Compile for specified Ascend SoC"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo "    --mssanitizer          Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
        echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
        echo "    --dump_cce             Dump kernel precompiled files"
        echo "    --bisheng_flags=flag1,flag2"
        echo "                           Specify bisheng compiler flags (comma-separated for multiple)"
        echo "    --kernel_template_input=args0,args1"
        echo "                           Specify kernel template input arguments(comma-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=crop_and_resize"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=crop_and_resize --build-type=Debug"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=crop_and_resize --mssanitizer"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=crop_and_resize --oom"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=crop_and_resize --dump_cce"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=crop_and_resize --bisheng_flags=ccec_g,oom"
        return
        ;;
      test)
        echo "Test Options:"
        echo $dotted_line
        echo "    -u                     Build and run all unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --cov                  Enable code coverage for unit tests"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
        echo "    --ophost_test          Build and run ophost unit tests"
        echo "    --opapi_test           Build and run opapi unit tests"
        echo "    --opgraph_test         Build and run opgraph unit tests"
        echo "    --ophost -u            Same as --ophost_test"
        echo "    --opapi -u             Same as --opapi_test"
        echo "    --opgraph -u           Same as --opgraph_test"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh -u --noexec --cov"
        echo "    bash build.sh -u --ophost --soc=ascend910b --ops=grid_sample"
        echo "    bash build.sh --ophost_test --opapi_test --noexec"
        echo "    bash build.sh --ophost --opapi --opgraph -u --cov"
        return
        ;;
      clean)
        echo "Clean Options:"
        echo $dotted_line
        echo "    --make_clean           Clean build artifacts"
        echo $dotted_line
        return
        ;;
      valgrind)
        echo "Valgrind Options:"
        echo $dotted_line
        echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
        echo $dotted_line
        return
        ;;
      ophost)
        echo "Ophost Build Options:"
        echo $dotted_line
        echo "    --ophost               Build ophost library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --build-type=<TYPE>"
        echo "                           Specify build type(TYPE options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --ophost -j16 -O3"
        echo "    bash build.sh --ophost --build-type=Debug"
        return
        ;;
      opapi)
        echo "Opapi Build Options:"
        echo $dotted_line
        echo "    --opapi                Build opapi library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --build-type=<TYPE>"
        echo "                           Specify build type(TYPE options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opapi -j16 -O3"
        echo "    bash build.sh --opapi --build-type=Debug"
        return
        ;;
      opgraph)
        echo "Opgraph Build Options:"
        echo $dotted_line
        echo "    --opgraph              Build opgraph library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --build-type=<TYPE>"
        echo "                           Specify build type(TYPE options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opgraph -j16 -O3"
        echo "    bash build.sh --opgraph --build-type=Debug"
        return
        ;;
      onnxplugin)
        echo "ONNXPlugin Build Options:"
        echo $dotted_line
        echo "    --onnxplugin           Build onnxplugin library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --build-type=<TYPE>    Specify build type(TYPE options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --onnxplugin -j16 -O3"
        echo "    bash build.sh --onnxplugin --build-type=Debug"
        return
        ;;
      ophost_test)
        echo "Ophost Test Options:"
        echo $dotted_line
        echo "    --ophost_test          Build and run ophost unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --cov                  Enable code coverage for unit tests"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --ophost_test --noexec --cov"
        return
        ;;
      opapi_test)
        echo "Opapi Test Options:"
        echo $dotted_line
        echo "    --opapi_test           Build and run opapi unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --cov                  Enable code coverage for unit tests"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opapi_test --noexec --cov"
        return
        ;;
      opgraph_test)
        echo "Opgraph Test Options:"
        echo $dotted_line
        echo "    --opgraph_test         Build and run opgraph unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --cov                  Enable code coverage for unit tests"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opgraph_test --noexec --cov"
        return
        ;;
      run_example)
        echo "Run examples Options:"
        echo $dotted_line
        echo "    --run_example op_type  mode[eager:graph] [pkg_mode --vendor_name=name]     Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
        echo "    --simulator   Enable simulator mode when running aclnn examples"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --run_example grid_sample eager"
        echo "    bash build.sh --run_example grid_sample graph"
        echo "    bash build.sh --run_example grid_sample eager cust"
        echo "    bash build.sh --run_example grid_sample eager cust --vendor_name=custom"
        echo "    bash build.sh --run_example grid_sample eager --simulator --soc=ascend950"
        return
        ;;
      genop)
        echo "Gen Op Directory Options:"
        echo $dotted_line
        echo "    --genop=op_class/op_name      Create the initial directory for op_name undef op_class"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --genop=examples/add"
        return
        ;;
      genop_aicpu)
        echo "Gen Op Directory Options:"
        echo $dotted_line
        echo "    --genop_aicpu=op_class/op_name      Create the initial directory for op_name undef op_class"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --genop_aicpu=examples/add"
        return
        ;;
    esac
  fi

  echo "build script for ops-cv repository"
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-O[n]] [-u] "
  echo ""
  echo ""
  echo "Options:"
  echo $dotted_line
  echo "    Build parameters "
  echo $dotted_line
  echo "    -h Print usage"
  echo "    -j[n] Compile thread nums, default is 8, eg: -j8"
  echo "    -v Cmake compile verbose"
  echo "    -O[n] Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
  echo "    -u Compile all ut"
  echo $dotted_line
  echo "    examples, Build ophost_test with O3 level compilation optimization and do not execute."
  echo "    ./build.sh --ophost_test --noexec -O3"
  echo $dotted_line
  echo "    The following are all supported arguments:"
  echo $dotted_line
  echo "    --build-type Specify build type(TYPE options: Release/Debug), Default:Release"
  echo "    --cov When building uTest locally, count the coverage."
  echo "    --noexec Only compile ut, do not execute the compiled executable file"
  echo "    --make_clean make clean"
  echo "    --asan enable asan with pkg on the host side"
  echo "    --valgrind run ut with valgrind. This option will disable asan, noexec and run utest by valgrind"
  echo "    --ccache=<VALUE> Enable or disable ccache compilation acceleration"
  echo "                     VALUE options: on/off/true/false/disable, Default: on"
  echo "                     Example: --ccache=off to disable ccache"
  echo "    --ops Compile specified operator, use snake name, like: --ops=grid_sample,iou_v2, use ',' to separate different operator"
  echo "    --soc Compile binary with specified Ascend SoC, like: --soc=ascend910b"
  echo "    --soc supported parameters must only in [ascend031 ascend035 ascend310b ascend310p ascend610lite ascend630 ascend910_93 ascend950 ascend910b ascend910 mc62 kirinx90 kirin9030], A3(--soc=ascend910_93)"
  echo "    --vendor_name Specify the custom operator package vendor name, like: --vendor_name=customize, default to customize-cv"
  echo "    --aicpu build aicpu task"
  echo "    --opgraph build op_graph_cv.so"
  echo "    --onnxplugin build oponnx_plugin_cv.so"
  echo "    --opapi build opapi_cv.so"
  echo "    --ophost build ophost_cv.so"
  echo "    --opkernel build binary kernel"
  echo "    --opkernel_aicpu build aicpu kernel"
  echo "    --jit build run package without kernel bin"
  echo "    --pkg build run package with kernel bin"
  echo "    --pkg-type=<TYPE> Specify package type(TYPE options: run/rpm/deb/all), Default: run"
  echo "    --static build static library package"
  echo "    --experimental Build experimental version"
  echo "    --opapi_test build and run opapi unit tests"
  echo "    --ophost_test build and run ophost unit tests"
  echo "    --opgraph_test build and run opgraph unit tests"
  echo "    --opkernel_test build and run opkernel unit tests"
  echo "    --opkernel_aicpu_test build and run aicpu opkernel unit tests"
  echo "    --run_example Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
  echo "    --simulator Enable simulator mode for run_example (requires --soc parameter)"
  echo "    --genop Create the initial directory for op"
  echo "    --genop_aicpu Create the initial directory for AI CPU op"
  echo "    --mssanitizer Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
  echo "    --oom Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
  echo "    --dump_cce Dump kernel precompiled files"
  echo "    --bisheng_flags Specify bisheng compiler config, like: --bisheng_flags=ccec_g,oom, use ',' to separate different compiler flags"
  echo "    --kernel_template_input Specify kernel template input arguments, like: --kernel_template_input=0,1, use ',' to separate different kernel template args"
  echo "to be continued ..."
}

check_help_combinations() {
  local args=("$@")
  local has_u=false
  local has_test_command=false
  local has_build_command=false
  local has_package=false
  local has_opkernel=false
  local has_opkernel_aicpu=false

  for arg in "${args[@]}"; do
    case "$arg" in
      -u) has_u=true ;;
      --ophost_test | --opapi_test | --opgraph_test | --ophost | --opapi | --opgraph | --onnxplugin)
        has_test_command=true
        has_build_command=true
        ;;
      --pkg) has_package=true ;;
      --opkernel) has_opkernel=true ;;
      --opkernel_aicpu) has_opkernel_aicpu=true ;;
      --help | -h) ;;
    esac
  done

  # Check the invalid command combinations in help
  if [[ "$has_package" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    echo "[ERROR] --pkg cannot be used with test(-u, --ophost_test, etc.), --ophost, --opapi, or --opgraph"
    return 1
  fi

  if [[ "$has_opkernel" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    echo "[ERROR] --opkernel cannot be used with test(-u, --ophost_test, etc.), --ophost, --opapi, or --opgraph"
    return 1
  fi

  if [[ "$has_opkernel_aicpu" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    echo "[ERROR] --opkernel_aicpu cannot be used with test(-u, --ophost_test, etc.), --ophost, --opapi, or --opgraph"
    return 1
  fi

  return 0
}

check_param() {
    if [[ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]]; then
      ENABLE_CUSTOM=FALSE
    fi
  # --ops不能与--ophost，--opapi，--opgraph同时存在，如果带U则可以
  if [[ -n "$COMPILED_OPS" && "$ENABLE_TEST" == "FALSE" ]] && [[ "$OP_HOST" == "TRUE" || "$OP_API" == "TRUE" || "$OP_GRAPH" == "TRUE" ]]; then
    echo "[ERROR] --ops cannot be used with --ophost, --opapi, or --opgraph"
    exit 1
  fi

  # -pkg不能与-u（UT模式，包含_test的参数）或者--ophost，--opapi，--opgraph同时存在
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    if [[ "$ENABLE_TEST" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with test(-u, --ophost_test, etc.)"
      exit 1
    fi

    if [[ "$OP_HOST" == "TRUE" || "$OP_API" == "TRUE" || "$OP_GRAPH" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with --ophost, --opapi, --opgraph"
      exit 1
    fi

    if [[ "$ENABLE_GENOP" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with --genop"
      exit 1
    fi

    if [[ "$ENABLE_GENOP_AICPU" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with --genop_aicpu"
      exit 1
    fi

    if [[ -n "${BUILD_TYPE}" ]]; then
      if [[ "${BUILD_TYPE}" != "Release" && "${BUILD_TYPE}" != "Debug" ]]; then
        echo "[ERROR] --build-type only support Release/Debug Mode"
        exit 1
      fi
    fi
    if [[ "${BUILD_TYPE}" == "Debug" ]]; then
      if [[ "$ENABLE_MSSANITIZER" == "TRUE" || "$ENABLE_OOM" == "TRUE" || "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
        echo "[ERROR] --build-type=Debug cannot be used with --mssanitizer, --oom or --dump_cce"
        exit 1
      fi
    fi

    if [[ "$ENABLE_MSSANITIZER" == "TRUE" && "$ENABLE_OOM" == "TRUE" ]]; then
      echo "[ERROR] --mssanitizer cannot be used with --oom"
      exit 1
    fi

    if [ -n "$BISHENG_FLAGS" ]; then
      if [[ "$ENABLE_MSSANITIZER" == "TRUE" || "$ENABLE_OOM" == "TRUE" || "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
        echo "[ERROR] --bisheng_flags= cannot be used with --mssanitizer, --oom, --dump_cce"
        exit 1
      fi
    fi

    if [ -n "$KERNEL_TEMPLATE_INPUT" ]; then
      if [[ -z "${COMPILED_OPS}" || "$COMPILED_OPS" == *","* ]]; then
        echo "[ERROR] --kernel_template_input must be used with --ops= and can only specify a single operator"
        exit 1
      fi
    fi
  fi

  if [[ "$PACKAGE_TYPE_SET" == "TRUE" && "$ENABLE_PACKAGE" != "TRUE" ]]; then
    echo "[ERROR] --pkg-type can only be used with --pkg"
    exit 1
  fi

  if [[ "$PACKAGE_TYPE" != "run" ]]; then
    if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
      echo "[ERROR] --pkg-type=${PACKAGE_TYPE} cannot be used with --static"
      exit 1
    fi
    if [[ "$ENABLE_JIT" == "TRUE" ]]; then
      echo "[ERROR] --pkg-type=${PACKAGE_TYPE} cannot be used with --jit"
      exit 1
    fi
    if [[ "$ENABLE_CUSTOM" == "TRUE" ]]; then
      echo "[ERROR] --pkg-type=${PACKAGE_TYPE} only supports built-in ops-cv packages; do not use --ops, --vendor_name, or --experimental"
      exit 1
    fi
  fi

  if $(echo ${USE_CMD} | grep -wq "static") && [[ "$ENABLE_PACKAGE" != "TRUE" ]]; then
    echo "[ERROR] --static can only be used with --pkg"
    exit 1
  fi

  if [[ "$ENABLE_STATIC" == "TRUE" && "$ENABLE_JIT" == "TRUE" && "$ENABLE_CUSTOM" == "TRUE" ]]; then
    echo "[ERROR] --static with --jit cannot be used with --ops, --vendor_name, or --experimental"
    exit 1
  fi

  if $(echo ${USE_CMD} | grep -wq "opkernel") && $(echo ${USE_CMD} | grep -wq "jit"); then
    echo "[ERROR] --opkernel cannot be used with --jit"
    exit 1
  fi

  if $(echo ${USE_CMD} | grep -wq "opkernel_aicpu") && $(echo ${USE_CMD} | grep -wq "jit"); then
    echo "[ERROR] --opkernel_aicpu cannot be used with --jit"
    exit 1
  fi

  if [[ "$ENABLE_SIMULATOR" == "TRUE" && -z "$COMPUTE_UNIT" ]]; then
    echo "[ERROR] --simulator requires --soc parameter to be specified"
    exit 1
  fi

  if [[ "$ENABLE_SIMULATOR" == "TRUE" && "$EXAMPLE_MODE" == "graph" ]]; then
    echo "[ERROR] --simulator does not support graph mode. Please use eager mode instead."
    exit 1
  fi
}

set_create_libs() {
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    return
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" && "$ENABLE_CUSTOM" != "TRUE" ]]; then
    BUILD_LIBS=("ophost_${REPOSITORY_NAME}" "opapi_${REPOSITORY_NAME}" "opgraph_${REPOSITORY_NAME}" "oponnx_plugin_${REPOSITORY_NAME}")
    ENABLE_CREATE_LIB=TRUE
  else
    if [[ "$OP_HOST" == "TRUE" ]]; then
      BUILD_LIBS+=("ophost_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$OP_API" == "TRUE" ]]; then
      BUILD_LIBS+=("opapi_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$OP_GRAPH" == "TRUE" ]]; then
      BUILD_LIBS+=("opgraph_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$ONNX_PLUGIN" == "TRUE" ]]; then
      BUILD_LIBS+=("oponnx_plugin_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$OP_KERNEL" == "TRUE" ]]; then
      ENABLE_BINARY=TRUE
    fi
  fi
}

set_ut_mode() {
  if [[ "$ENABLE_TEST" != "TRUE" ]]; then
    return
  fi
  UT_TEST_ALL=TRUE
  if [[ "$OP_HOST" == "TRUE" ]]; then
    OP_HOST_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_API" == "TRUE" ]]; then
    OP_API_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_GRAPH" == "TRUE" ]]; then
    OP_GRAPH_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_KERNEL" == "TRUE" ]]; then
    OP_KERNEL_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_KERNEL_AICPU" == "TRUE" ]]; then
    OP_KERNEL_AICPU_UT=TRUE
    UT_TEST_ALL=FALSE
  fi

  # 检查测试项，至少有一个
  if [[ "$UT_TEST_ALL" == "FALSE" && "$OP_HOST_UT" == "FALSE" && "$OP_API_UT" == "FALSE" && "$OP_GRAPH_UT" == "FALSE" && "$OP_KERNEL_UT" == "FALSE" && "$OP_KERNEL_AICPU_UT" == "FALSE" ]]; then
    echo "[ERROR] At least one test target must be specified (ophost_test, opapi_test, opgraph_test, opkernel_test, opkernel_aicpu_test)"
    usage
    exit 1
  fi

  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_HOST_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_host_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_API_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_api_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_kernel_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_AICPU_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_aicpu_op_kernel_ut")
  fi
}
process_genop() {
  local opt_name=$1
  local genop_value=$2

  if [[ "$opt_name" == "genop" ]]; then
    ENABLE_GENOP=TRUE
  elif [[ "$opt_name" == "genop_aicpu" ]]; then
    ENABLE_GENOP_AICPU=TRUE
  else
    usage "genop"
    exit 1
  fi

  if [[ "$genop_value" != *"/"* ]] || [[ "$genop_value" == *"/" ]]; then
    usage "$opt_name"
    exit 1
  fi

  GENOP_NAME=${genop_value##*/}
  local remaining=${genop_value%/*}

  if [[ "$remaining" != *"/"* ]]; then
    GENOP_TYPE=$remaining
    GENOP_BASE=${BASE_PATH}
  else
    GENOP_TYPE=${remaining##*/}
    GENOP_BASE=${remaining%/*}
    if [[ ! "$GENOP_BASE" =~ ^/ && ! "$GENOP_BASE" =~ ^[a-zA-Z]: ]]; then
      GENOP_BASE="${BASE_PATH}/${GENOP_BASE}"
    fi
  fi
}

checkopts_run_example() {
  ENABLE_RUN_EXAMPLE=TRUE
  EXAMPLE_NAME="${!OPTIND}"
  ((OPTIND++))
  if [[ $OPTIND -le $# ]] && [[ "${!OPTIND}" != --* ]]; then
    EXAMPLE_MODE="${!OPTIND}"
    ((OPTIND++))
  fi

  if [[ $OPTIND -le $# ]] && [[ "${!OPTIND}" != --* ]]; then
    PKG_MODE="${!OPTIND}"
    ((OPTIND++))
    if [[ $OPTIND -le $# ]] && [[ "${!OPTIND}" == --vendor_name* ]]; then
      VENDOR="${!OPTIND}"
      VENDOR="${VENDOR#*=}"
      ((OPTIND++))
    else
      VENDOR="custom"
    fi
  fi
}

checkopts() {
  THREAD_NUM=${CORE_NUMS}
  THREAD_NUM=8
  VERBOSE=""
  BUILD_MODE=""
  COMPILED_OPS=""
  UT_TEST_ALL=FALSE
  CHANGED_FILES=""
  CI_MODE=FALSE
  COMPUTE_UNIT=""
  VENDOR_NAME=""
  SHOW_HELP=""
  EXAMPLE_NAME=""
  EXAMPLE_MODE=""
  USE_CMD="$*"
  BISHENG_FLAGS=""
  KERNEL_TEMPLATE_INPUT=""

  BUILD_TYPE="Release"
  PACKAGE_TYPE="run"
  PACKAGE_TYPE_SET=FALSE
  ENABLE_MSSANITIZER=FALSE
  ENABLE_OOM=FALSE
  ENABLE_DUMP_CCE=FALSE
  ENABLE_COVERAGE=FALSE
  ENABLE_UT_EXEC=TRUE
  ENABLE_ASAN=FALSE
  ENABLE_VALGRIND=FALSE
  ENABLE_BINARY=FALSE
  ENABLE_CUSTOM=FALSE
  ENABLE_STATIC=FALSE
  ENABLE_SIMULATOR=FALSE
  ENABLE_PACKAGE=FALSE
  ENABLE_EXPERIMENTAL=FALSE
  ENABLE_TEST=FALSE
  ENABLE_JIT=FALSE
  AICPU_ONLY=FALSE
  DISABLE_AICPU=FALSE
  OP_API_UT=FALSE
  OP_HOST_UT=FALSE
  OP_GRAPH_UT=FALSE
  OP_KERNEL_UT=FALSE
  OP_KERNEL_AICPU_UT=FALSE
  OP_API=FALSE
  OP_HOST=FALSE
  OP_GRAPH=FALSE
  ONNX_PLUGIN=FALSE
  OP_KERNEL=FALSE
  OP_KERNEL_AICPU=FALSE
  ENABLE_CREATE_LIB=FALSE
  ENABLE_RUN_EXAMPLE=FALSE
  ENABLE_RULE_LAUNCH=""
  ENABLE_CCACHE=TRUE
  BUILD_LIBS=()
  UT_TARGETS=()

  ENABLE_GENOP=FALSE
  ENABLE_GENOP_AICPU=FALSE
  GENOP_TYPE=""
  GENOP_NAME=""
  GENOP_BASE=${BASE_PATH}

  # 首先检查所有参数是否合法
  for arg in "$@"; do
    if [[ "$arg" =~ ^- ]]; then # 只检查以-开头的参数
      if ! check_option_validity "$arg"; then
        echo "Use 'bash build.sh --help' for more information."
        exit 1
      fi
      if [[ "$arg" == "--pkg-type" ]]; then
        echo "[ERROR] --pkg-type requires a value: run/rpm/deb/all"
        exit 1
      fi
      if [[ "$arg" == --pkg-type=* ]]; then
        check_pkg_type "${arg#*=}"
      fi
    fi
  done

  # 检查并处理--help
  for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
      # 检查帮助信息中的组合参数
      check_help_combinations "$@"
      local comb_result=$?
      if [ $comb_result -eq 1 ]; then
        exit 1
      fi
      SHOW_HELP="general"

      # 检查 --help 前面的命令
      for prev_arg in "$@"; do
        case "$prev_arg" in
          --pkg) SHOW_HELP="package" ;;
          --opkernel) SHOW_HELP="opkernel" ;;
          --opkernel_aicpu) SHOW_HELP="opkernel_aicpu" ;;
          -u) SHOW_HELP="test" ;;
          --make_clean) SHOW_HELP="clean" ;;
          --valgrind) SHOW_HELP="valgrind" ;;
          --ophost) SHOW_HELP="ophost" ;;
          --opapi) SHOW_HELP="opapi" ;;
          --opgraph) SHOW_HELP="opgraph" ;;
          --onnxplugin) SHOW_HELP="onnxplugin" ;;
          --ophost_test) SHOW_HELP="ophost_test" ;;
          --opapi_test) SHOW_HELP="opapi_test" ;;
          --opgraph_test) SHOW_HELP="opgraph_test" ;;
          --run_example) SHOW_HELP="run_example" ;;
          --genop) SHOW_HELP="genop" ;;
          --genop_aicpu) SHOW_HELP="genop_aicpu" ;;
        esac
      done

      usage "$SHOW_HELP"
      exit 0
    fi
  done

  # Process the options
  while getopts $SUPPORTED_SHORT_OPTS opt; do
    case "${opt}" in
      h)
        usage
        exit 0
        ;;
      j) THREAD_NUM=$OPTARG ;;
      v) VERBOSE="VERBOSE=1" ;;
      O) BUILD_MODE="-O$OPTARG" ;;
      u) ENABLE_TEST=TRUE ;;
      f)
        CHANGED_FILES=$OPTARG
        CI_MODE=TRUE
        ;;
      -) case $OPTARG in
        help)
          usage
          exit 0
          ;;
        ops=*)
          COMPILED_OPS=${OPTARG#*=}
          ENABLE_CUSTOM=TRUE
          ;;
        genop=*)
          process_genop "genop" "${OPTARG#*=}"
          ;;
        genop_aicpu=*)
          process_genop "genop_aicpu" "${OPTARG#*=}"
          ;;
        soc=*)
          COMPUTE_UNIT=${OPTARG#*=}
          ;;
        vendor_name=*)
          VENDOR_NAME=${OPTARG#*=}
          ENABLE_CUSTOM=TRUE
          ;;
        build-type=*)
          BUILD_TYPE=${OPTARG#*=}
          ;;
        pkg-type=*)
          PACKAGE_TYPE=${OPTARG#*=}
          check_pkg_type "${PACKAGE_TYPE}"
          PACKAGE_TYPE_SET=TRUE
          ;;
        mssanitizer) ENABLE_MSSANITIZER=TRUE ;;
        oom) ENABLE_OOM=TRUE ;;
        dump_cce) ENABLE_DUMP_CCE=TRUE ;;
        bisheng_flags=*)
          BISHENG_FLAGS=${OPTARG#*=}
          ;;
        kernel_template_input=*)
          KERNEL_TEMPLATE_INPUT=${OPTARG#*=}
          ;;
        cov) ENABLE_COVERAGE=TRUE ;;
        noexec) ENABLE_UT_EXEC=FALSE ;;
        aicpu) AICPU_ONLY=TRUE ;;
        noaicpu) DISABLE_AICPU=TRUE ;;
        static)
          ENABLE_STATIC=TRUE
          ENABLE_BINARY=TRUE
          ;;
        pkg)
          ENABLE_BINARY=TRUE
          ENABLE_PACKAGE=TRUE
          ;;
        cann_3rd_lib_path=*)
          CANN_3RD_LIB_PATH="$(realpath ${OPTARG#*=})"
          ;;
        jit)
          ENABLE_BINARY=FALSE
          ENABLE_JIT=TRUE
          ;;
        asan) ENABLE_ASAN=TRUE ;;
        valgrind)
          ENABLE_VALGRIND=TRUE
          ENABLE_UT_EXEC=FALSE
          BUILD_TYPE="Debug"
          ;;
        simulator) ENABLE_SIMULATOR=TRUE ;;
        rule_launch=*)
          ENABLE_RULE_LAUNCH=${OPTARG#*=}
          ;;
        ccache=*)
          local ccache_val=${OPTARG#*=}
          if [[ "$ccache_val" == "off" || "$ccache_val" == "false" || "$ccache_val" == "disable" ]]; then
            ENABLE_CCACHE=FALSE
          fi
          ;;
        run_example)
          checkopts_run_example "$@"
          ;;
        experimental)
          ENABLE_EXPERIMENTAL=TRUE
          ENABLE_CUSTOM=TRUE
          ;;
        make_clean)
          clean_build
          clean_build_out
          exit 0
          ;;
        *)
          ## 如果不在RELEASE_TARGETS 或者 SUPPORTED_UT_TARGETS，不做处理
          if ! in_array "$OPTARG" "${RELEASE_TARGETS[@]}" && ! in_array "$OPTARG" "${SUPPORTED_UT_TARGETS[@]}"; then
            echo "[ERROR] Invalid option: --$OPTARG"
            usage
            exit 1
          fi
          ## 如果_test形式的，那么获取正确的名，并强设UT_MODE为TRUE
          if [[ "$OPTARG" == *"_test" ]]; then
            OPTARG="${OPTARG%_test}"
            ENABLE_TEST=TRUE
          fi

          if [[ "$OPTARG" == "ophost" ]]; then
            OP_HOST=TRUE
          elif [[ "$OPTARG" == "opapi" ]]; then
            OP_API=TRUE
          elif [[ "$OPTARG" == "opgraph" ]]; then
            OP_GRAPH=TRUE
          elif [[ "$OPTARG" == "onnxplugin" ]]; then
            ONNX_PLUGIN=TRUE
          elif [[ "$OPTARG" == "opkernel" ]]; then
            OP_KERNEL=TRUE
          elif [[ "$OPTARG" == "opkernel_aicpu" ]]; then
            OP_KERNEL_AICPU=TRUE
          else
            usage
            exit 1
          fi
          ;;
      esac ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
        ;;
    esac
  done

  if [[ "$ENABLE_JIT" == "TRUE" ]]; then
    ENABLE_BINARY=FALSE
  fi

  check_param
  set_create_libs
  parse_changed_files
  set_ut_mode
}

parse_changed_files() {
  if [[ -z "$CHANGED_FILES" ]]; then
    return
  fi

  if [[ "$CHANGED_FILES" != /* ]]; then
    CHANGED_FILES=$PWD/$CHANGED_FILES
  fi

  echo "changed files is $CHANGED_FILES"
  echo "$dotted_line"
  echo "changed lines:"
  cat "$CHANGED_FILES"
  echo "$dotted_line"

  COMPILED_OPS=$(python3 scripts/ci/parse_changed_ops.py "$CHANGED_FILES" "$ENABLE_EXPERIMENTAL")
  echo "related ops $COMPILED_OPS"

  if [[ -z $COMPILED_OPS ]]; then
    if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
      COMPILED_OPS='nms_with_mask'
    else
      COMPILED_OPS='grid_sample'
    fi
    echo "No ops changed found, set op $COMPILED_OPS as default."
  fi

  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    return
  fi

  local script_ret=$(python3 scripts/ci/parse_changed_files.py "$CHANGED_FILES" "$ENABLE_EXPERIMENTAL")
  IFS='&&' read -r related_ut soc_info <<<"$script_ret"
  echo "related ut $related_ut"
  echo "related soc_info $soc_info"

  COMPUTE_UNIT=$soc_info

  if [[ "$related_ut" == "set()" ]]; then
    ENABLE_TEST=FALSE
    echo "no ut matched! no need to run!"
    echo "---------------- CANN build finished ----------------"
    return
  else
    ENABLE_TEST=TRUE
  fi

  if [[ "$related_ut" =~ "ALL_UT" ]]; then
    echo "ALL UT is triggered!"
    return
  fi
  if [[ ("$related_ut" =~ "OP_HOST_UT" || "$related_ut" =~ "OP_GRAPH_UT") && "$OP_HOST" == "TRUE" ]]; then
    echo "OP_HOST_UT is triggered!"
    OP_HOST_UT=TRUE
    OP_KERNEL_UT=TRUE
    OP_KERNEL=TRUE
    OP_GRAPH=TRUE
    ENABLE_CUSTOM=TRUE
  fi
  if [[ "$related_ut" =~ "OP_API_UT" && "$OP_API" == "TRUE" ]]; then
    echo "OP_API_UT is triggered!"
    OP_API_UT=TRUE
    ENABLE_CUSTOM=TRUE
  fi
  if [[ "$related_ut" =~ "OP_KERNEL_UT" && "$OP_KERNEL" == "TRUE" ]]; then
    echo "OP_KERNEL_UT is triggered!"
    OP_KERNEL_UT=TRUE
    ENABLE_CUSTOM=TRUE
  fi
  if [[ "$related_ut" =~ "OP_KERNEL_AICPU_UT" && "$OP_KERNEL_AICPU" == "TRUE" ]]; then
    echo "OP_KERNEL_AICPU_UT is triggered!"
    OP_KERNEL_AICPU_UT=TRUE
    ENABLE_CUSTOM=TRUE
  fi
}
