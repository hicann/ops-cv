#!/bin/bash
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -e
RELEASE_TARGETS=("ophost" "opapi" "opgraph" "opkernel")
UT_TARGETS=("ophost_test" "opapi_test" "opgraph_test" "opkernel_test")
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910b" "ascend910_93" "ascend310p" "ascend910")

# 所有支持的短选项
SUPPORTED_SHORT_OPTS="hj:vO:uf:-:"

# 所有支持的长选项
SUPPORTED_LONG_OPTS=(
  "help" "ops=" "soc=" "vendor_name=" "debug" "cov" "noexec" "aicpu" "opkernel" "jit"
  "pkg" "disable_asan" "valgrind" "make_clean"
  "ophost" "opapi" "opgraph" "ophost_test" "opapi_test" "opgraph_test" "opkernel_test"
  "run_example" "genop=" "genop_aicpu=" "experimental"
)

in_array() {
  local needle="$1"
  shift
  local haystack=("$@")
  for item in "${haystack[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

check_option_validity() {
  local arg="$1"

  if [[ "$arg" =~ ^-[^-] ]]; then
    local opt_chars=${arg:1}

    local needs_arg_opts=$(echo "$SUPPORTED_SHORT_OPTS" | grep -o "[a-zA-Z]:" | tr -d ':')

    local i=0
    while [ $i -lt ${#opt_chars} ]; do
      local char="${opt_chars:$i:1}"

      if [[ ! "$SUPPORTED_SHORT_OPTS" =~ "$char" ]]; then
        echo "[ERROR] Invalid short option: -$char"
        return 1
      fi

      if [[ "$needs_arg_opts" =~ "$char" ]]; then
        while [ $i -lt ${#opt_chars} ] && [[ "${opt_chars:$i:1}" =~ [0-9a-zA-Z] ]]; do
          i=$((i + 1))
        done
      else
        i=$((i + 1))
      fi
    done
    return 0
  fi

  if [[ "$arg" =~ ^-- ]]; then
    local long_opt="${arg:2}"
    local opt_name="${long_opt%%=*}"

    for supported_opt in "${SUPPORTED_LONG_OPTS[@]}"; do
      # with "=" in long options
      if [[ "$supported_opt" =~ =$ ]]; then
        local base_opt="${supported_opt%=}"
        if [[ "$opt_name" == "$base_opt" ]]; then
          return 0
        fi
      else
        # without "=" in long options
        if [[ "$opt_name" == "$supported_opt" ]]; then
          return 0
        fi
      fi
    done

    echo "[ERROR] Invalid long option: --$opt_name"
    return 1
  fi

  return 0
}

dotted_line="----------------------------------------------------------------"
export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
REPOSITORY_NAME="cv"

CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
ARCH_INFO=$(uname -m)

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
export COMPILER_INCLUDE_PATH="${ASCEND_HOME_PATH}/compiler/include"
export GRAPH_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/graph"
export GE_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/ge"
export INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
export GRAPH_LIBRARY_STUB_PATH="${ASCEND_HOME_PATH}/compiler/lib64/stub"
export GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/compiler/lib64"

# print usage message
usage() {
  local specific_help="$1"

  if [[ -n "$specific_help" ]]; then
    case "$specific_help" in
      package)
        echo "Package Build Options:"
        echo $dotted_line
        echo "    --pkg                  Build run package with kernel bin"
        echo "    --jit                  Build run package without kernel bin"
        echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
        echo "    --vendor_name=name     Specify custom operator package vendor name"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --debug                Build with debug mode"
        echo "    --experimental         Build experimental version"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --pkg --soc=ascend910b --vendor_name=customize -j16 -O3"
        echo "    bash build.sh --pkg --ops=add,sub --debug"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b"
        return
        ;;
      opkernel)
        echo "Opkernel Build Options:"
        echo $dotted_line
        echo "    --opkernel             Build binary kernel"
        echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub"
        return
        ;;
      test)
        echo "Test Options:"
        echo $dotted_line
        echo "    -u                     Build and run all unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --cov                  Enable code coverage for unit tests"
        echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
        echo "    --ophost_test          Build and run ophost unit tests"
        echo "    --opapi_test           Build and run opapi unit tests"
        echo "    --opgraph_test         Build and run opgraph unit tests"
        echo "    --ophost -u            Same as --ophost_test"
        echo "    --opapi -u             Same as --opapi_test"
        echo "    --opgraph -u           Same as --opgraph_test"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh -u --noexec --cov"
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
        echo "    --debug                Build with debug mode"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --ophost -j16 -O3"
        echo "    bash build.sh --ophost --debug"
        return
        ;;
      opapi)
        echo "Opapi Build Options:"
        echo $dotted_line
        echo "    --opapi                Build opapi library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --debug                Build with debug mode"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opapi -j16 -O3"
        echo "    bash build.sh --opapi --debug"
        return
        ;;
      opgraph)
        echo "Opgraph Build Options:"
        echo $dotted_line
        echo "    --opgraph              Build opgraph library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --debug                Build with debug mode"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opgraph -j16 -O3"
        echo "    bash build.sh --opgraph --debug"
        return
        ;;
      ophost_test)
        echo "Ophost Test Options:"
        echo $dotted_line
        echo "    --ophost_test          Build and run ophost unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --cov                  Enable code coverage for unit tests"
        echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
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
        echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
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
        echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opgraph_test --noexec --cov"
        return
        ;;
      run_example)
        echo "Run examples Options:"
        echo $dotted_line
        echo "    --run_example op_type  mode[eager:graph] [pkg_mode --vendor_name=name]     Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --run_example abs eager"
        echo "    bash build.sh --run_example abs graph"
        echo "    bash build.sh --run_example abs eager cust"
        echo "    bash build.sh --run_example abs eager cust --vendor_name=custom"
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
  echo "    bash build.sh [-h] [-j[n]] [-v] [-O[n]] [-g] [-u] "
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
  echo "    --debug build with debug mode"
  echo "    --cov When building uTest locally, count the coverage."
  echo "    --noexec Only compile ut, do not execute the compiled executable file"
  echo "    --make_clean make clean"
  echo "    --disable_asan disable asan"
  echo "    --valgrind run ut with valgrind. This option will disable asan, noexec and run utest by valgrind"
  echo ""
  echo "    --ops Compile specified operator, use snake name, like: --ops=add,add_lora, use ',' to separate different operator"
  echo "    --soc Compile binary with specified Ascend SoC, like: --soc=ascend310p,ascend910b, use ',' to separate different SoC"
  echo "    --vendor_name Specify the custom operator package vendor name, like: --vendor_name=customize, default to customize-cv"
  echo "    --aicpu build aicpu task"
  echo "    --opgraph build op_graph_cv.so"
  echo "    --opapi build opapi_cv.so"
  echo "    --ophost build ophost_cv.so"
  echo "    --opkernel build binary kernel"
  echo "    --jit build run package without kernel bin"
  echo "    --pkg build run package with kernel bin"
  echo "    --experimental Build experimental version"
  echo "    --opapi_test build and run opapi unit tests"
  echo "    --ophost_test build and run ophost unit tests"
  echo "    --opgraph_test build and run opgraph unit tests"
  echo "    --opkernel_test build and run opkernel unit tests"
  echo "    --run_example Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
  echo "    --genop Create the initial directory for op"
  echo "    --genop_aicpu Create the initial directory for AI CPU op"
  echo "to be continued ..."
}

check_help_combinations() {
  local args=("$@")
  local has_u=false
  local has_test_command=false
  local has_build_command=false
  local has_package=false
  local has_opkernel=false

  for arg in "${args[@]}"; do
    case "$arg" in
      -u) has_u=true ;;
      --ophost_test | --opapi_test | --opgraph_test | --ophost | --opapi | --opgraph)
        has_test_command=true
        has_build_command=true
        ;;
      --pkg) has_package=true ;;
      --opkernel) has_opkernel=true ;;
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

  return 0
}

check_param() {
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

    if $(echo ${USE_CMD} | grep -wq "jit"); then
      ENABLE_BINARY=FALSE
    fi
  fi

  if $(echo ${USE_CMD} | grep -wq "opkernel") && $(echo ${USE_CMD} | grep -wq "jit"); then
    echo "[ERROR] --opkernel cannot be used with --jit"
    exit 1
  fi
}

set_create_libs() {
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    return
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" && "$ENABLE_CUSTOM" != "TRUE" ]]; then
    BUILD_LIBS=("ophost_${REPOSITORY_NAME}" "opapi_${REPOSITORY_NAME}" "opgraph_${REPOSITORY_NAME}")
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

  # 检查测试项，至少有一个
  if [[ "$UT_TEST_ALL" == "FALSE" && "$OP_HOST_UT" == "FALSE" && "$OP_API_UT" == "FALSE" && "$OP_GRAPH_UT" == "FALSE" && "$OP_KERNEL_UT" == "FALSE" ]]; then
    echo "[ERROR] At least one test target must be specified (ophost_test, opapi_test, opgraph_test, opkernel_test)"
    usage
    exit 1
  fi

  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_HOST_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_host_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_API_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_api_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_GRAPH_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_graph_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_kernel_ut")
  fi
}

print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_RED}[ERROR] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

checkopts() {
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

  ENABLE_DEBUG=FALSE
  ENABLE_CONVERAGE=FALSE
  ENABLE_UT_EXEC=TRUE
  ENABLE_ASAN=TRUE
  ENABLE_VALGRIND=FALSE
  ENABLE_BINARY=FALSE
  ENABLE_CUSTOM=FALSE
  ENABLE_PACKAGE=FALSE
  ENABLE_EXPERIMENTAL=FALSE
  ENABLE_TEST=FALSE
  AICPU_ONLY=FALSE
  OP_API_UT=FALSE
  OP_HOST_UT=FALSE
  OP_GRAPH_UT=FALSE
  OP_KERNEL_UT=FALSE
  OP_API=FALSE
  OP_HOST=FALSE
  OP_GRAPH=FALSE
  OP_KERNEL=FALSE
  ENABLE_CREATE_LIB=FALSE
  ENABLE_RUN_EXAMPLE=FALSE
  BUILD_LIBS=()
  UT_TARGES=()

  ENABLE_GENOP=FALSE
  ENABLE_GENOP_AICPU=FALSE
  GENOP_TYPE=""
  GENOP_NAME=""

  # 首先检查所有参数是否合法
  for arg in "$@"; do
    if [[ "$arg" =~ ^- ]]; then # 只检查以-开头的参数
      if ! check_option_validity "$arg"; then
        echo "Use 'bash build.sh --help' for more information."
        exit 1
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
          -u) SHOW_HELP="test" ;;
          --make_clean) SHOW_HELP="clean" ;;
          --valgrind) SHOW_HELP="valgrind" ;;
          --ophost) SHOW_HELP="ophost" ;;
          --opapi) SHOW_HELP="opapi" ;;
          --opgraph) SHOW_HELP="opgraph" ;;
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

    if [[ "$genop_value" != *"/"* ]] || [[ "$genop_value" == *"/"*"/"* ]]; then
      usage "$opt_name"
      exit 1
    fi

    GENOP_TYPE=$(echo "$genop_value" | cut -d'/' -f1)
    GENOP_NAME=$(echo "$genop_value" | cut -d'/' -f2)
  }

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
        debug) ENABLE_DEBUG=TRUE ;;
        cov) ENABLE_CONVERAGE=TRUE ;;
        noexec) ENABLE_UT_EXEC=FALSE ;;
        aicpu) AICPU_ONLY=TRUE ;;
        pkg)
          ENABLE_BINARY=TRUE
          ENABLE_PACKAGE=TRUE
          ;;
        jit) ENABLE_BINARY=FALSE ;;
        disable_asan) ENABLE_ASAN=FALSE ;;
        valgrind)
          ENABLE_VALGRIND=TRUE
          ENABLE_UT_EXEC=FALSE
          ENABLE_DEBUG=TRUE
          ENABLE_ASAN=FALSE
          ;;
        run_example) ENABLE_RUN_EXAMPLE=TRUE ;;
        experimental) ENABLE_EXPERIMENTAL=TRUE ;;
        make_clean)
          clean_build
          clean_build_out
          exit 0
          ;;
        *)
          ## 如果不在RELEASE_TARGETS 或者 UT_TARGETS，不做处理
          if ! in_array "$OPTARG" "${RELEASE_TARGETS[@]}" && ! in_array "$OPTARG" "${UT_TARGETS[@]}"; then
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
          elif [[ "$OPTARG" == "opkernel" ]]; then
            OP_KERNEL=TRUE
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

  if [[ "$1" == "--run_example" ]]; then
    EXAMPLE_NAME="$2"
    EXAMPLE_MODE="$3"
    if [[ -n "$4" ]]; then
      PKG_MODE="$4"
      VENDOR="custom"
      if [[ -n "$5" ]]; then
        VENDOR="$5"
        VENDOR="${VENDOR#*=}"
      fi
    fi
  fi

  check_param
  set_create_libs
  set_ut_mode
}

custom_cmake_args() {
  if [[ -n $COMPILED_OPS ]]; then
    COMPILED_OPS="${COMPILED_OPS//,/;}"
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_OP_NAME=$COMPILED_OPS"
  fi
  if [[ -n $VENDOR_NAME ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DVENDOR_NAME=$VENDOR_NAME"
  fi
}

assemble_cmake_args() {
  if [[ "$ENABLE_ASAN" == "TRUE" ]]; then
    set +e
    echo 'int main() {return 0;}' | gcc -x c -fsanitize=address - -o asan_test >/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "This environment does not have the ASAN library, no need enable ASAN"
      ENABLE_ASAN=FALSE
    else
      $(rm -f asan_test)
      CMAKE_ARGS="$CMAKE_ARGS -DENABLE_ASAN=TRUE"
    fi
    set -e
  fi
  if [[ "$ENABLE_VALGRIND" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_VALGRIND=TRUE"
  fi
  if [[ "$ENABLE_DEBUG" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_DEBUG=TRUE"
  fi
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_TEST=TRUE"
  fi
  if [[ "$ENABLE_UT_EXEC" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_UT_EXEC=TRUE"
  fi
  if [[ "$ENABLE_BINARY" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_BINARY=TRUE"
  fi
  if [[ "$ENABLE_CUSTOM" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_CUSTOM=TRUE -DENABLE_BINARY=TRUE"
    custom_cmake_args
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PACKAGE=TRUE"
  fi
  if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_EXPERIMENTAL=TRUE"
  fi
  if [[ "x$BUILD_MODE" != "x" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DBUILD_MODE=${BUILD_MODE}"
  fi
  if [[ "$OP_HOST_UT" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DOP_HOST_UT=TRUE"
  fi
  if [[ "$OP_API_UT" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DOP_API_UT=TRUE"
  fi
  if [[ "$OP_GRAPH_UT" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DOP_GRAPH_UT=TRUE"
  fi
  if [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DOP_KERNEL_UT=TRUE"
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUT_TEST_ALL=TRUE"
  fi
  if [[ -n $COMPUTE_UNIT ]]; then
    IFS=',' read -ra COMPUTE_UNIT <<<"$COMPUTE_UNIT"
    COMPUTE_UNIT_SHORT=""
    for unit in "${COMPUTE_UNIT[@]}"; do
      for support_unit in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
        lowercase_word=$(echo "$unit" | tr '[:upper:]' '[:lower:]')
        if [[ "$lowercase_word" == *"$support_unit"* ]]; then
          COMPUTE_UNIT_SHORT="$COMPUTE_UNIT_SHORT$support_unit;"
          break
        fi
      done
    done
    if [[ -z $COMPUTE_UNIT_SHORT ]]; then
      print_error "The soc [${COMPUTE_UNIT}] is not support."
      usage
      exit 1
    fi
    echo "COMPUTE_UNIT: ${COMPUTE_UNIT_SHORT}"
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT_SHORT"
  fi
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    rm -rf ${BUILD_PATH}/*
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    rm -rf ${BUILD_OUT_PATH}/*
  fi
}

build_lib() {
  echo $dotted_line
  echo "Start to build libs ${BUILD_LIBS[@]}"
  clean_build

  git submodule init && git submodule update
  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..

  for lib in "${BUILD_LIBS[@]}"; do
    echo "Building target ${lib}"
    cmake --build . --target ${lib} -- ${VERBOSE} -j $THREAD_NUM
  done

  if [[ "$ENABLE_CONVERAGE" =~ "TRUE" ]]; then
    cmake --build . --target generate_ops_cpp_cov -- -j $THREAD_NUM
  fi
  echo $dotted_line
  echo "Build libs ${BUILD_LIBS[@]} success"
  echo $dotted_line
}

build_binary() {
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    return
  fi

  echo $dotted_line
  echo "Start to build binary"
  clean_build

  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake .. ${CMAKE_ARGS}

  echo "--------------- prepare build start ---------------"
  local all_targets=$(cmake --build . --target help)
  if echo "${all_targets}" | grep -wq "ascendc_impl_gen"; then
    cmake --build . --target ascendc_impl_gen -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  fi
  if echo "${all_targets}" | grep -wq "gen_bin_scripts"; then
    cmake --build . --target gen_bin_scripts -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  fi
  echo "--------------- prepare build end ---------------"

  echo "--------------- binary build start ---------------"
  local cur_path=$(pwd)
  mkdir -p ${cur_path}/op_impl/ai_core/tbe/op_tiling
  if [ ! -L op_impl/ai_core/tbe/op_tiling/liboptiling.so ]; then
    if [ -e ${cur_path}/libophost_cv.so ]; then
      ln -s ${cur_path}/libophost_cv.so op_impl/ai_core/tbe/op_tiling/liboptiling.so
    else
      cmake --build . --target ophost_cv -- ${VERBOSE} -j $THREAD_NUM
      ln -s ${cur_path}/libophost_cv.so op_impl/ai_core/tbe/op_tiling/liboptiling.so
    fi
  fi
  export ASCEND_CUSTOM_OPP_PATH=${cur_path}
  if echo "${all_targets}" | grep -wq "binary"; then
    cmake --build . --target binary -- ${VERBOSE} -j1 # TODO: fix -j1
    if [ $? -ne 0 ]; then
      echo "[ERROR] Kernel compile failed!" && exit 1
    fi
  fi
  if echo "${all_targets}" | grep -wq "gen_bin_info_config"; then
    cmake --build . --target gen_bin_info_config -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  fi
  echo "--------------- binary build end ---------------"

  echo "Build binary success"
  echo $dotted_line
}

build_package() {
  echo "--------------- build package start ---------------"
  clean_build_out

  if [[ "$ENABLE_BINARY" != "TRUE" && "$ENABLE_CUSTOM" != "TRUE" ]]; then
    # gen impl python files
    local all_targets=$(cmake --build . --target help)
    if echo "${all_targets}" | grep -wq "ascendc_impl_gen"; then
      cmake --build . --target ascendc_impl_gen -- ${VERBOSE} -j $THREAD_NUM
      if [ $? -ne 0 ]; then exit 1; fi
    fi
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  cmake --build . --target package -- ${VERBOSE} -j $THREAD_NUM
  echo "--------------- build package end ---------------"
}

build_ut() {
  echo $dotted_line
  echo "Start to build ut"
  clean_build

  git submodule init && git submodule update
  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  cmake --build . --target ${UT_TARGES[@]} -- ${VERBOSE} -j $THREAD_NUM
  if [[ "$ENABLE_CONVERAGE" =~ "TRUE" ]]; then
    cmake --build . --target generate_ops_cpp_cov -- -j $THREAD_NUM
  fi
}

build_example() {
  echo $dotted_line
  echo "Start to run examples,name:${EXAMPLE_NAME} mode:${EXAMPLE_MODE}"
  clean_build
  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi

  cd "${BUILD_PATH}"
  if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
    file=$(find ../ -path "*/${EXAMPLE_NAME}/examples/*" -name test_aclnn_*.cpp)
    if [ -z "$file" ]; then
      echo "ERROR: ${EXAMPLE_NAME} do not have eager examples"
      exit 1
    fi

    for f in $file; do
      echo "Start compile and run examples file: $f"
      if [[ "${PKG_MODE}" == "" ]]; then
        g++ ${f} -I ${INCLUDE_PATH} -I ${ACLNN_INCLUDE_PATH} -L ${EAGER_LIBRARY_PATH} -lopapi_cv -lascendcl -lnnopbase -o test_aclnn_${EXAMPLE_NAME}
      elif [[ "${PKG_MODE}" == "cust" ]]; then
        echo "pkg_mode:${PKG_MODE} vendor_name:${VENDOR}"
        export CUST_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_cv/op_api/lib"     # 仅自定义算子需要
        export CUST_INCLUDE_PATH="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_cv/op_api/include" # 仅自定义算子需要
        CUST_ACLNNOP_INCLUDE_PATH="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_cv/op_api/include/aclnnop"
        ln -s ${CUST_INCLUDE_PATH} ${CUST_ACLNNOP_INCLUDE_PATH}
        g++ ${f} -I ${INCLUDE_PATH} -I ${INCLUDE_PATH}/aclnnop -I ${CUST_INCLUDE_PATH} -L ${CUST_LIBRARY_PATH} -L ${EAGER_LIBRARY_PATH} -lcust_opapi -lascendcl -lnnopbase -o test_aclnn_${EXAMPLE_NAME} -Wl,-rpath=${CUST_LIBRARY_PATH}
        rm ${CUST_ACLNNOP_INCLUDE_PATH}
      else
        echo "Error: pkg_mode(${PKG_MODE}) must be cust."
        exit 1
      fi
      ./test_aclnn_${EXAMPLE_NAME}
    done
  elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
    file=$(find ../ -path "*/${EXAMPLE_NAME}/examples/*" -name test_geir_*.cpp)
    if [ -z "$file" ]; then
      echo "ERROR: ${EXAMPLE_NAME} do not have graph examples"
      exit 1
    fi
    for f in $file; do
      echo "Start compile and run examples file: $f"
      g++ ${f} -I ${GRAPH_INCLUDE_PATH} -I ${GE_INCLUDE_PATH} -I ${INCLUDE_PATH} -I ${INC_INCLUDE_PATH} -L ${GRAPH_LIBRARY_STUB_PATH} -L ${GRAPH_LIBRARY_PATH} -lgraph -lge_runner -lgraph_base -o test_geir_${EXAMPLE_NAME}
      ./test_geir_${EXAMPLE_NAME}
    done
  else
    usage
    exit 1
  fi
}

gen_op() {
  if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
    echo "Error: op_class or op_name is not set."
    usage "genop"
  fi

  echo $dotted_line
  echo "Start to create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

  if [ ! -d "${GENOP_TYPE}" ]; then
    mkdir -p "${GENOP_TYPE}"
    cp examples/CMakeLists.txt "${GENOP_TYPE}/CMakeLists.txt"
    sed -i '/add_subdirectory(conversion)/a add_subdirectory('"${GENOP_TYPE}"')' CMakeLists.txt
  fi

  BASE_DIR=${GENOP_TYPE}/${GENOP_NAME}
  mkdir -p "${BASE_DIR}"

  cp -r examples/add_example/* "${BASE_DIR}/"

  rm -rf "${BASE_DIR}/examples"
  rm -rf "${BASE_DIR}/op_host/config"

  for file in $(find "${BASE_DIR}" -name "*.h" -o -name "*.cpp"); do
    head -n 14 "$file" >"${file}.tmp"
    cat "${file}.tmp" >"$file"
    rm "${file}.tmp"
  done

  for file in $(find "${BASE_DIR}" -type f); do
    sed -i "s/add_example/${GENOP_NAME}/g" "$file"
  done

  cd ${BASE_DIR}
  for file in $(find ./ -name "add_example*"); do
    new_file=$(echo "$file" | sed "s/add_example/${GENOP_NAME}/g")
    mv "$file" "$new_file"
  done

  echo "Create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE} success"
}

gen_aicpu_op() {
  if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
    echo "Error: op_class or op_name is not set."
    usage "genop"
  fi

  echo $dotted_line
  echo "Start to create the AI CPU initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

  if [ ! -d "${GENOP_TYPE}" ]; then
    mkdir -p "${GENOP_TYPE}"
    cp examples/CMakeLists.txt "${GENOP_TYPE}/CMakeLists.txt"
    sed -i '/add_subdirectory(conversion)/a add_subdirectory('"${GENOP_TYPE}"')' CMakeLists.txt
  fi

  BASE_DIR=${GENOP_TYPE}/${GENOP_NAME}
  mkdir -p "${BASE_DIR}"

  cp -r examples/add_example_aicpu/* "${BASE_DIR}/"

  rm -rf "${BASE_DIR}/examples"
  rm -rf "${BASE_DIR}/op_host/config"

  for file in $(find "${BASE_DIR}" -name "*.h" -o -name "*.cpp"); do
    head -n 14 "$file" >"${file}.tmp"
    cat "${file}.tmp" >"$file"
    rm "${file}.tmp"
  done

  for file in $(find "${BASE_DIR}" -type f); do
    sed -i "s/add_example_aicpu/${GENOP_NAME}/g" "$file"
  done

  cd ${BASE_DIR}
  for file in $(find ./ -name "add_example_aicpu*"); do
    new_file=$(echo "$file" | sed "s/add_example_aicpu/${GENOP_NAME}_aicpu/g")
    mv "$file" "$new_file"
  done

  for file in $(find ./ -name "add_example*"); do
    new_file=$(echo "$file" | sed "s/add_example/${GENOP_NAME}/g")
    mv "$file" "$new_file"
  done

  echo "Create the AI CPU initial directory for ${GENOP_NAME} under ${GENOP_TYPE} success"
}

main() {
  checkopts "$@"
  if [ "$THREAD_NUM" -gt "$CORE_NUMS" ]; then
    echo "compile thread num:$THREAD_NUM over core num:$CORE_NUMS, adjust to core num"
    THREAD_NUM=$CORE_NUMS
  fi
  assemble_cmake_args
  echo "CMAKE_ARGS: ${CMAKE_ARGS}"
  if [ "$ENABLE_CREATE_LIB" == "TRUE" ]; then
    build_lib
  fi
  if [[ "$ENABLE_BINARY" == "TRUE" || "$ENABLE_CUSTOM" == "TRUE" ]]; then
    build_binary
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    build_package
  fi
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    build_ut
  fi
  if [[ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]]; then
    build_example
  fi
  if [[ "$ENABLE_GENOP" == "TRUE" ]]; then
    gen_op
  fi
  if [[ "$ENABLE_GENOP_AICPU" == "TRUE" ]]; then
    gen_aicpu_op
  fi
}

set -o pipefail
if [ $# -eq 0 ]; then
  usage
fi
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
