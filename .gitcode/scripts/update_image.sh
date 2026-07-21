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
# 如果检测到更新镜像，就输出算子名


current_ops_str="$1"
chip_type="${2:-${CHIP_TYPE:-}}"  #

# 定义算子列表
image_ops_910b=("grid_sample" "upsample_nearest3d" "upsample_linear1d" "upsample_bicubic2d" "grid_sampler2_d_grad")
objdetec_ops_910b=("iou_v2")
operator_list_910b=("${image_ops_910b[@]}" "${objdetec_ops_910b[@]}")

image_ops_950=("resize_bilinear_v2" "resize_nearest_neighbor_v2" "resize_bicubic_v2" "resize_linear")
objdetec_ops_950=("iou_v2")
operator_list_950=("${image_ops_950[@]}" "${objdetec_ops_950[@]}")

# 根据 chip_type 决定要添加的算子
predefined_ops=()
if [ -n "$chip_type" ]; then
    case "$chip_type" in
        910b)
            predefined_ops=("${operator_list_910b[@]}")
            ;;
        950)
            predefined_ops=("${operator_list_950[@]}")
            ;;
        *)
            echo "Warning: Unknown chip_type '$chip_type', adding all predefined operators." >&2
            # 合并两个列表
            all_ops=("${operator_list_910b[@]}" "${operator_list_950[@]}")
            # 去重（可选，但后面整体去重时会做）
            predefined_ops=("${all_ops[@]}")
            ;;
    esac
else
    echo "No chip_type specified, adding all predefined operators." >&2
    all_ops=("${operator_list_910b[@]}" "${operator_list_950[@]}")
    predefined_ops=("${all_ops[@]}")
fi

# 将 current_ops_str 转换为数组
if [ -n "$current_ops_str" ]; then
    IFS=' ' read -r -a current_ops <<< "$current_ops_str"
else
    current_ops=()
fi

# 合并并去重
# 先复制当前数组
merged_ops=("${current_ops[@]}")
# 遍历预定义算子，如果不在 merged_ops 中则添加
for op in "${predefined_ops[@]}"; do
    if [[ ! " ${merged_ops[@]} " =~ " ${op} " ]]; then
        merged_ops+=("$op")
    fi
done

# 输出合并后的字符串
if [ ${#merged_ops[@]} -gt 0 ]; then
    printf "%s" "${merged_ops[0]}"
    for op in "${merged_ops[@]:1}"; do
        printf ",%s" "$op"
    done
fi
printf "\n"
