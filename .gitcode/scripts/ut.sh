#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set +e

REPOSITORY_NAME="ops-cv"
sudo update-alternatives --set gcc /usr/bin/gcc-14
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
gcc --version

if [ -z "${ASCEND_3RD_LIB_PATH}" ]; then
    export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
fi

if [ -f /home/jenkins/Ascend/cann/bin/setenv.bash ]; then
    source /home/jenkins/Ascend/cann/bin/setenv.bash
fi

LOG_HEAD()
{
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

LOG_DO()
{
    echo "[LOG_DO] $*"
    "$@"
}

DP_ASSERT_EQUAL()
{
    local actual="$1"
    local expected="$2"
    local msg="$3"
    if [ "${actual}" != "${expected}" ]; then
        echo "::error::ASSERT FAILED: ${msg} (expected=${expected}, actual=${actual})"
        exit 1
    fi
}

LOG_HEAD "UT ${REPOSITORY_NAME}."
cd "${WORKSPACE}/" || exit

if [ "${GIT_TARGET_BRANCH}x" = "masterx" ]; then
    pip3 install tensorflow
    if grep -q "scripts/ci/mirror_update_time.txt" "${WORKSPACE}/pr_filelist.txt"; then
        ops_line=$(bash "${WORKSPACE}/update_image.sh" "$ops")
        ops=$(echo "$ops_line" | tail -n1)
        echo "ops: ${ops}"
        LOG_DO sh build.sh -u --ops="$ops" -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
    elif [ "${GE_ST_RT2}X" == "classifyX" ]; then
        cd "${WORKSPACE}/CI/cann/pipeline/bin/common"
        python3 count_files.py "${WORKSPACE}/classify_rule.yaml" ops-cv "${WORKSPACE}" "${WORKSPACE}"
    else
        LOG_DO sh build.sh -u --cov -f "pr_filelist.txt" -j16 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
    fi
else
    LOG_DO sh build.sh -u --ophost --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -f "pr_filelist.txt" -j16
    DP_ASSERT_EQUAL "$?" "0" "Run UT TESTCASE"
    LOG_DO sh build.sh -u --opapi --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -f "pr_filelist.txt" -j16
    DP_ASSERT_EQUAL "$?" "0" "Run UT TESTCASE"
    LOG_DO sh build.sh -u --opkernel --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
fi
DP_ASSERT_EQUAL "$?" "0" "Run UT TESTCASE"

echo "ut_process=coverage" >> "${ATOMGIT_OUTPUT}"