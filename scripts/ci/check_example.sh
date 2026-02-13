#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

current_dir=$(pwd)
variables_cmake="cmake/variables.cmake"
op_category_list=$(perl -0777 -ne 'if (/set\(OPS_CATEGORY_LIST\s*(.*?)\)/s) { print $1 }' "$current_dir/$variables_cmake" | grep -oP '"[^"]+"' | sed 's/"//g' | xargs)

IFS=' ' read -r -a op_categories <<< "$op_category_list"
valid_dirs=()
pr_file=$(realpath "${1:-pr_filelist.txt}")

for category in "${op_categories[@]}"
do
    first_level="$current_dir/$category"
    if [ -d "$first_level" ]; then
        for second_level in "$first_level"/*/
        do
            examples_dir="$second_level""examples"
            if [ -d "$examples_dir" ]; then
                if ls "$examples_dir"/test_aclnn_* 1> /dev/null 2> /dev/null; then
                    dir_name=$(basename "$second_level")
                    valid_dirs+=("$dir_name")
                fi
            fi
        done
    fi
done

ops_name=()
mapfile -t lines < ${pr_file}
for file_path in "${lines[@]}"
do
    file_path=$(echo "$file_path" | xargs)
    if [ -z "$file_path" ]; then
        continue
    fi
    if [[ "$file_path" == *.md ]]; then
        continue
    fi
    for dir in "${valid_dirs[@]}"
    do
        if [[ "$file_path" == *"/$dir/"* ]]; then
            if [[ ! " ${ops_name[@]} " =~ " $dir " ]]; then
                ops_name+=("$dir")
            fi
            break
        fi
    done
done

for name in "${ops_name[@]}"
do
    ./single/cann-ops-cv-${name}_linux*.run
    bash build.sh --run_example $name eager cust --vendor_name=$name
    status=$?
    if [ $status -ne 0 ]; then
        echo "${name} example fail"
        exit 1
    fi
done