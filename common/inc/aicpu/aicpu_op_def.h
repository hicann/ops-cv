/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aicpu_op_def.h
 * \brief
 */
#ifndef CV_COMMON_AICPU_OP_DEF_H
#define CV_COMMON_AICPU_OP_DEF_H
namespace ops {
    const std::string CV_AICPU_KERNEL_SO = "libcv_aicpu_kernels.so";
    const std::string OP_INFO_WORKSPACESIZE = "opInfo.workspaceSize";
    const std::string DEFAULT_WORKSPACE_SIZE = "100"; // 记录AICPU算子日志信息内存大小，单位 KB
    const std::string OP_INFO_FORMAT_AGNOSTIC = "opInfo.formatAgnostic";// 是否格式敏感，默认不敏感
    const std::string FALSE_FORMAT_AGNOSTIC = "False";
    const std::string TRUE_FORMAT_AGNOSTIC = "True";
    const std::string OP_INFO_OPS_FLAG = "opInfo.opsFlag"; // 是否支持常量折叠，默认不支持
    const std::string CLOSE_OPS_FLAG = "OPS_FLAG_CLOSE";
    const std::string OPEN_OPS_FLAG = "OPS_FLAG_OPEN";
    const std::string OP_INFO_SUB_TYPE_OF_INFERSHAPE =  "opInfo.subTypeOfInferShape"; // 算子类型，默认是1类算子
    const std::string DEFAULT_SUB_TYPE_OF_INFERSHAPE_1 =  "1";
    const std::string DEFAULT_SUB_TYPE_OF_INFERSHAPE_2 =  "2"; 
    const std::string DEFAULT_SUB_TYPE_OF_INFERSHAPE_3 =  "3"; 
    const std::string DEFAULT_SUB_TYPE_OF_INFERSHAPE_4 =  "4"; 


    template <typename TOpDef>
    inline void ApplyCvAicpuDefaultCfg(TOpDef &opDef)
    {
        opDef.AICPU().OpKernelLib("CUSTAICPUKernel");
        opDef.AICPU().KernelSo(CV_AICPU_KERNEL_SO.c_str());
        opDef.AICPU().UserDefined(true);
        opDef.AICPU().ExtendCfgInfo(OP_INFO_WORKSPACESIZE.c_str(), DEFAULT_WORKSPACE_SIZE.c_str());
        opDef.AICPU().ExtendCfgInfo(OP_INFO_FORMAT_AGNOSTIC.c_str(), FALSE_FORMAT_AGNOSTIC.c_str());
        opDef.AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), CLOSE_OPS_FLAG.c_str());
        opDef.AICPU().ExtendCfgInfo(OP_INFO_SUB_TYPE_OF_INFERSHAPE.c_str(), DEFAULT_SUB_TYPE_OF_INFERSHAPE_1.c_str());  
    }
} // namespace op
#endif
