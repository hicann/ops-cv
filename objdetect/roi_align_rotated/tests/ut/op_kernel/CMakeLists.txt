# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # arch22 kernel UT：tiling 依赖为 arch22 通用实现单份（经 TILING_DIR 选编形态）
    set(roi_align_rotated_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch22/roi_align_rotated_tiling.cpp
        )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本，多个以分号分隔，例如："ascend950PR_9599;AscendB1"
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合，不需要则传入空字符串，例如："-DDTYPE_X=float"，多个使用空格分隔，例如："-DDTYPE_X=float -DDTYPE_Y=float"
    # param4：该算子依赖的所有tiling源码文件
    AddOpTestCase(roi_align_rotated "ascend910B1" "" "${roi_align_rotated_tiling_files}")

    # ascend950（arch35）kernel UT（e 方案归并：950 SIMT 用例 test_roi_align_rotated_950.cpp，
    # 经 soc 过滤与arch22 ascend910B1 调用互不混编——--soc=ascend950 时仅本调用注册）；
    # tiling 依赖为 arch35 独立实现单份（自带 IMPL_OP_OPTILING 注册，与 arch22 互斥）
    set(roi_align_rotated_tiling_files_950
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/roi_align_rotated_tiling.cpp
        )
    AddOpTestCase(roi_align_rotated "Ascend950PR_9599" "-DDTYPE_X=float" "${roi_align_rotated_tiling_files_950}")
endif()
