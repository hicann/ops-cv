/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_upsample_nearest_2d_v2.h"
#include <iomanip>
#include "image/resize_nearest_neighbor_v2/op_host/op_api/resize_nearest_neighbor_v2.h"
#include "image/upsample_nearest/op_host/op_api/upsample_nearest_exact2d.h"
#include "upsample_nearest_3d.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const int64_t FOURDIMS = 4;
static const int64_t ZERO = 0;
static const int64_t ONE = 1;
static const uint64_t TWO = 2;
static const int64_t THREE = 3;
static const float HALF_FLOAT = 0.5f;  // 特殊处理后的scale
static const int64_t SMALL_SIZE = 128; // 用于判断是否特殊处理scale的shape的大小
static const int64_t PRECISION_LEN = 6; // 保证浮点数精度的长度

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
    op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT8,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType> &GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}
static bool CheckDtypeValid(const aclTensor *self)
{
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    return true;
}

static bool CheckDtypeEqual(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *self, const aclIntArray *outputSize)
{
    uint64_t size = outputSize->Size();
    if (size != TWO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output size equals to %lu, but got size %lu.", TWO, size);
        return false;
    }

    auto selfShape = self->GetViewShape();
    if (selfShape.GetDimNum() != FOURDIMS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "It is expected input size equals to %ld, but got size %lu.",
            FOURDIMS,
            selfShape.GetDimNum());
        return false;
    }

    if (selfShape.GetDim(TWO) == ZERO || selfShape.GetDim(THREE) == ZERO || outputSize->GetData()[ZERO] == ZERO ||
        outputSize->GetData()[ONE] == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output sizes should be greater than %ld, "
            "but got input (H: %ld, W: %ld) output (H: %ld, W: %ld).",
            ZERO,
            selfShape.GetDim(TWO),
            selfShape.GetDim(THREE),
            outputSize->GetData()[ZERO],
            outputSize->GetData()[ONE]);
        return false;
    }

    if (selfShape.GetDim(ONE) == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Non-empty 4D data tensor expected but got a tensor with size %s.",
            op::ToString(selfShape).GetString());
        return false;
    }

    return true;
}

static bool CheckOutEqual(const aclTensor *out, const aclIntArray *outputSize)
{
    auto outShape = out->GetViewShape();
    if (out->GetStorageFormat() == op::Format::FORMAT_NCHW) {
        if (outputSize->GetData()[ZERO] != outShape.GetDim(TWO) ||
            outputSize->GetData()[ONE] != outShape.GetDim(THREE)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected output size (H: %ld, W: %ld), but got output tensor size (H: %ld, W: %ld).",
                outputSize->GetData()[ZERO],
                outputSize->GetData()[ONE],
                outShape.GetDim(TWO),
                outShape.GetDim(THREE));
            return false;
        }
        return true;
    } else if (out->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        if (outputSize->GetData()[ZERO] != outShape.GetDim(ONE) || outputSize->GetData()[ONE] != outShape.GetDim(TWO)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected output size (H: %ld, W: %ld), but got output tensor size (H: %ld, W: %ld).",
                outputSize->GetData()[ZERO],
                outputSize->GetData()[ONE],
                outShape.GetDim(TWO),
                outShape.GetDim(THREE));
            return false;
        }
        return true;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support this format of input and output.");
        return false;
    }
}

static aclnnStatus CheckParams(const aclTensor *self, const aclIntArray *outputSize, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out数据类型是否相同
    CHECK_RET(CheckDtypeEqual(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否支持
    CHECK_RET(CheckShape(self, outputSize), ACLNN_ERR_PARAM_INVALID);

    // 5.检查self和out N/C轴的大小是否一致
    CHECK_RET(CheckNCDimValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 6.检查out和outputSize的H和W的大小是否一致
    CHECK_RET(CheckOutEqual(out, outputSize), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor *View4dAs5d(const aclTensor *input, aclOpExecutor *executor)
{
    // NCHW -> unsqueeze(2) -> reformat -> NCDHW
    // 新增一维
    const int64_t appendDim[] = {2};
    aclIntArray *dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);
    return reformatInput;
}

static const aclTensor *View5dAs4d(const aclTensor *input, aclOpExecutor *executor)
{
    // NCDHW -> squeeze -> reformat -> NCHW
    // squeeze out into 4D
    const int64_t removeDim[] = {2};
    aclIntArray *dimSqueeze = executor->AllocIntArray(removeDim, 1);
    CHECK_RET(dimSqueeze != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(input, dimSqueeze, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);

    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

const aclTensor *transpose4D(const aclTensor *input, const op::Format &format, aclOpExecutor *executor)
{
    if (input->GetStorageFormat() != format) {
        int64_t permuteList[] = {ZERO, TWO, THREE, ONE};
        if (format == op::Format::FORMAT_NCHW) {
            permuteList[ZERO] = ZERO;
            permuteList[ONE] = THREE;
            permuteList[TWO] = ONE;
            permuteList[THREE] = static_cast<int64_t>(TWO);
        }
        auto permuteArray = executor->AllocIntArray(permuteList, FOURDIMS);
        CHECK_RET(permuteArray != nullptr, nullptr);

        auto selfTranspose = l0op::Transpose(input, permuteArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);

        if (format == op::Format::FORMAT_NCHW) {
            return selfTranspose;
        }
        auto reformatInput = l0op::ReFormat(selfTranspose, format);
        CHECK_RET(reformatInput != nullptr, nullptr);
        return reformatInput;
    }
    return input;
}

const aclTensor *upsampleNearest2dV2AiCpuCompute(
    const aclTensor *selfContiguous, const aclTensor *outContiguous, const aclTensor *size, aclOpExecutor *executor)
{
    if (selfContiguous->GetStorageFormat() == op::Format::FORMAT_NCHW) {
        const int64_t permuteNHWCList[] = {0, 2, 3, 1};
        auto permuteNHWCArray = executor->AllocIntArray(permuteNHWCList, FOURDIMS);
        CHECK_RET(permuteNHWCArray != nullptr, nullptr);

        auto selfTranspose = l0op::Transpose(selfContiguous, permuteNHWCArray, executor);
        CHECK_RET(selfTranspose != nullptr, nullptr);
        auto outTranspose = l0op::Transpose(outContiguous, permuteNHWCArray, executor);
        CHECK_RET(outTranspose != nullptr, nullptr);

        const aclTensor *resizeNearestOutAiCpu =
            l0op::ResizeNearestNeighborV2(selfTranspose, size, nullptr, outTranspose, executor);
        CHECK_RET(resizeNearestOutAiCpu != nullptr, nullptr);

        const int64_t permuteNCHWList[] = {0, 3, 1, 2};
        auto permuteNCHWArray = executor->AllocIntArray(permuteNCHWList, FOURDIMS);
        CHECK_RET(permuteNCHWArray != nullptr, nullptr);

        return l0op::Transpose(resizeNearestOutAiCpu, permuteNCHWArray, executor);
    } else {
        return l0op::ResizeNearestNeighborV2(selfContiguous, size, nullptr, outContiguous, executor);
    }
}

float computeScale(const aclTensor *selfContiguous, int64_t input, int64_t output, int64_t output2, float scale) {
    if (input == output) {
        return 1.0f;
    } else if (static_cast<int64_t>(TWO) * input == output) {
        //此处对标cpu，当输出是输入的两倍且满足一下两个条件之一时，传入的scale无效且固定为0.5：
        // 1. Format为NHWC且channel大于3
        // 2. 输出的H和W之和小于128
        if ((selfContiguous->GetStorageFormat() == op::Format::FORMAT_NHWC &&
            selfContiguous->GetViewShape().GetDim(ONE) > THREE) ||
            ((output + output2) <= SMALL_SIZE)) {
            return HALF_FLOAT;
        }
        return scale;
    }else {
        return scale;
    }
}

const aclTensor *upsampleNearest2dAiCoreCompute(const aclTensor *selfContiguous, const aclIntArray *outputSize,
    const aclFloatArray *scales, aclOpExecutor *executor)
{
    float scaleH = (*scales)[ONE];
    float scaleW = (*scales)[TWO];
    std::stringstream scaleHStream;
    std::stringstream scaleWStream;
    // 保证精度不丢失
    scaleHStream << std::fixed << std::setprecision(PRECISION_LEN) << scaleH;
    scaleWStream << std::fixed << std::setprecision(PRECISION_LEN) << scaleW;
    std::string scaleHStr = scaleHStream.str();
    std::string scaleWStr = scaleWStream.str();
    float scalesH = static_cast<float>(1.0 / std::stod(scaleHStr));
    float scalesW = static_cast<float>(1.0 / std::stod(scaleWStr));

    const int64_t inputH = selfContiguous->GetViewShape().GetDim(TWO);
    const int64_t inputW = selfContiguous->GetViewShape().GetDim(THREE);
    const int64_t outputH = (*outputSize)[ZERO];
    const int64_t outputW = (*outputSize)[ONE];

    scalesH = computeScale(selfContiguous, inputH, outputH, outputW, scalesH);
    scalesW = computeScale(selfContiguous, inputW, outputW, outputH, scalesW);

    float scalesD = 1.0;
    vector<float> scalesCastList{};
    scalesCastList.push_back(scalesD);
    scalesCastList.push_back(scalesH);
    scalesCastList.push_back(scalesW);
    const aclFloatArray *castScales = executor->AllocFloatArray(scalesCastList.data(), scalesCastList.size());
    CHECK_RET(castScales != nullptr, nullptr);

    vector<int64_t> sizeList{};
    sizeList.push_back(1);
    sizeList.push_back((*outputSize)[ZERO]);
    sizeList.push_back((*outputSize)[ONE]);
    const aclIntArray *size = executor->AllocIntArray(sizeList.data(), sizeList.size());
    CHECK_RET(size != nullptr, nullptr);

    auto self = View4dAs5d(selfContiguous, executor);
    CHECK_RET(self != nullptr, nullptr);
    auto outUpsampleNearest = l0op::UpsampleNearest3dNcdhw(self, size, scales, castScales, executor);
    CHECK_RET(outUpsampleNearest != nullptr, nullptr);

    return View5dAs4d(outUpsampleNearest, executor);
}
} // namespace

aclnnStatus aclnnUpsampleNearest2dV2GetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize,
    float scalesH, float scalesW, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnUpsampleNearest2dV2, DFX_IN(self, outputSize, scalesH, scalesW), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = static_cast<uint64_t>(0);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto ret = CheckParams(self, outputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto size = uniqueExecutor.get()->ConvertToTensor(outputSize, op::ToOpDataType(ACL_INT32));
    const aclTensor *resizeNearestOut = nullptr;

    if (CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        bool isAscendCSupport = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                                GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
                                GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
        if (scalesH > 0 && scalesW > 0 && isAscendCSupport) {
            vector<float> scalesList{};
            scalesList.push_back(1.0);
            scalesList.push_back(scalesH);
            scalesList.push_back(scalesW);
            const aclFloatArray *scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
            CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto selfNCHWContiguous = transpose4D(selfContiguous, op::Format::FORMAT_NCHW, uniqueExecutor.get());
            auto resizeNearestNCHWOut =
                upsampleNearest2dAiCoreCompute(selfNCHWContiguous, outputSize, scales, uniqueExecutor.get());
            resizeNearestOut = transpose4D(resizeNearestNCHWOut, self->GetStorageFormat(), uniqueExecutor.get());
        } else {
            if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
                vector<float> scalesList{};
                scalesList.push_back(scalesH);
                scalesList.push_back(scalesW);
                const aclFloatArray *scales = uniqueExecutor->AllocFloatArray(scalesList.data(), scalesList.size());
                CHECK_RET(scales != nullptr, ACLNN_ERR_INNER_NULLPTR);

                resizeNearestOut =
                    l0op::ResizeNearestNeighborV2(selfContiguous, size, scales, outContiguous, uniqueExecutor.get());
            } else {
                auto selfTransdata =
                    l0op::TransDataSpecial(selfContiguous, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
                CHECK_RET(selfTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

                auto outTransdata =
                    l0op::TransDataSpecial(outContiguous, op::Format::FORMAT_NC1HWC0, 0, uniqueExecutor.get());
                CHECK_RET(outTransdata != nullptr, ACLNN_ERR_INNER_NULLPTR);

                auto outAiCore =
                    l0op::ResizeNearestNeighborV2(selfTransdata, size, nullptr, outTransdata, uniqueExecutor.get());
                CHECK_RET(outAiCore != nullptr, ACLNN_ERR_INNER_NULLPTR);

                resizeNearestOut = l0op::TransData(outAiCore, self->GetStorageFormat(), 0, uniqueExecutor.get());
            }
        }
    } else {
        resizeNearestOut = upsampleNearest2dV2AiCpuCompute(selfContiguous, outContiguous, size, uniqueExecutor.get());
    }
    CHECK_RET(resizeNearestOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(resizeNearestOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUpsampleNearest2dV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUpsampleNearest2dV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
