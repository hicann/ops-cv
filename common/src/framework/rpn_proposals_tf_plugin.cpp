/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>

#include "cv_plugin_util.h"

namespace domi {
namespace {
constexpr int32_t kTaskNum = 9;
constexpr size_t kRequiredAttrNum = 5U;

const char* const kNmsV2 = "NonMaxSuppressionV2";
const char* const kGreater = "Greater";
const char* const kMinimum = "Minimum";

// 原插件按下标取常量，下标与算子原型的输入端口一一对应：
//   NonMaxSuppressionV2 : input(2)=max_output_size, input(3)=iou_threshold
//   Greater/Minimum     : input(0)=x1, input(1)=x2
const char* const kPortIouThreshold = "iou_threshold";
const char* const kPortMaxOutputSize = "max_output_size";
const char* const kPortX1 = "x1";
const char* const kPortX2 = "x2";

const char* const kSuffixFilteredBoxes = "filtered_boxes";
const char* const kSuffixMinimum = "Minimum";
const char* const kSuffixGreater = "Greater";

bool GetNodeName(const ge::Operator& node, std::string& name)
{
    ge::AscendString ascend_name;
    if ((node.GetName(ascend_name) != ge::GRAPH_SUCCESS) || (ascend_name.GetString() == nullptr)) {
        return false;
    }
    name = ascend_name.GetString();
    return true;
}

bool GetNodeType(const ge::Operator& node, std::string& type)
{
    ge::AscendString ascend_type;
    if ((node.GetOpType(ascend_type) != ge::GRAPH_SUCCESS) || (ascend_type.GetString() == nullptr)) {
        return false;
    }
    type = ascend_type.GetString();
    return true;
}

// 等价于原插件对 task_0 ~ task_8/generate_rpn_proposals/<suffix> 的逐一枚举
bool IsRpnNode(const std::string& name, const char* suffix)
{
    for (int32_t i = 0; i < kTaskNum; ++i) {
        if (name == ("task_" + std::to_string(i) + "/generate_rpn_proposals/" + suffix)) {
            return true;
        }
    }
    return false;
}

template <typename T>
bool ReadConst(const ge::Operator& node, const char* port, T& value)
{
    ge::Tensor data;
    if (node.GetInputConstData(port, data) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const uint8_t* buf = data.GetData();
    if ((buf == nullptr) || (data.GetSize() < sizeof(T))) {
        return false;
    }
    (void)memcpy(&value, buf, sizeof(T));
    return true;
}
} // namespace

static Status RpnProposalsParams(const std::vector<ge::Operator>& inside_nodes, ge::Operator& op)
{
    OP_LOGI(GetOpName(op).c_str(), "Enter RpnProposals fusion parser.");

    size_t attr_num = 0U;
    for (const auto& node : inside_nodes) {
        std::string name;
        std::string type;
        if (!GetNodeName(node, name) || !GetNodeType(node, type)) {
            OP_LOGE(GetOpName(op).c_str(), "Get inside node name or type failed.");
            return FAILED;
        }

        if ((type == kGreater) && IsRpnNode(name, kSuffixFilteredBoxes)) {
            float score_threshold = 0.0F;
            if (!ReadConst(node, kPortX2, score_threshold)) {
                OP_LOGE(GetOpName(op).c_str(), "Convert score_threshold data failed.");
                return PARAM_INVALID;
            }
            (void)op.SetAttr("score_threshold", score_threshold);
            attr_num++;
        }

        if ((type == kMinimum) && IsRpnNode(name, kSuffixMinimum)) {
            int32_t k = 0;
            if (!ReadConst(node, kPortX1, k)) {
                OP_LOGE(GetOpName(op).c_str(), "Convert topk k data failed.");
                return PARAM_INVALID;
            }
            (void)op.SetAttr("k", k);
            attr_num++;
        }

        if ((type == kGreater) && IsRpnNode(name, kSuffixGreater)) {
            float min_size = 0.0F;
            if (!ReadConst(node, kPortX2, min_size)) {
                OP_LOGE(GetOpName(op).c_str(), "Convert min_size data failed.");
                return PARAM_INVALID;
            }
            (void)op.SetAttr("min_size", min_size);
            attr_num++;
        }

        if (type == kNmsV2) {
            float nms_threshold = 0.0F;
            int32_t post_nms_num = 0;
            if (!ReadConst(node, kPortIouThreshold, nms_threshold) ||
                !ReadConst(node, kPortMaxOutputSize, post_nms_num)) {
                OP_LOGE(GetOpName(op).c_str(), "Convert nms_threshold or post_nms_num data failed.");
                return PARAM_INVALID;
            }
            (void)op.SetAttr("nms_threshold", nms_threshold);
            (void)op.SetAttr("post_nms_num", post_nms_num);
            attr_num += 2U;
        }
    }

    if (attr_num != kRequiredAttrNum) {
        OP_LOGE(GetOpName(op).c_str(), "Can not find right num of attr node in rpn_proposals, got %zu.", attr_num);
        return FAILED;
    }

    OP_LOGI(GetOpName(op).c_str(), "Obtain attributes for rpn_proposals SUCCESS.");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("RpnProposals")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RpnProposals")
    .FusionParseParamsFn(RpnProposalsParams)
    .ImplyType(ImplyType::TVM);
} // namespace domi
