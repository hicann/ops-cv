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
 * \file resize_nearest_neighbor_v2_tiling.cc
 * \brief resize_nearest_neighbor_v2_tiling
 */
#include "resize_nearest_neighbor_v2_tiling_base.h"
#include "../../op_kernel/arch35/resize_nearest_neighbor_v2_tiling_key.h"

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include <cmath>

namespace optiling {
constexpr size_t ATTR_ALIGN_CORERS_IDX = 0;
constexpr size_t ATTR_HALF_PIXEL_CENTERS_IDX = 1;
constexpr size_t ATTR_SCALES_IDX = 2;
constexpr size_t SCALES_NUM = 2;
constexpr size_t SCALE_H = 0;
constexpr size_t SCALE_W = 1;
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t DIM_LEN_4D = 4;
constexpr size_t DIM_LEN_5D = 5;
constexpr size_t C_DIM_IDX_NHWC = 3;
constexpr size_t C_DIM_IDX_NCHW = 1;
constexpr size_t N_DIM_IDX = 0;
constexpr size_t H_DIM_IDX_NCHW = 2;
constexpr size_t H_DIM_IDX_NHWC = 1;
constexpr size_t W_DIM_IDX_NCHW = 3;
constexpr size_t W_DIM_IDX_NHWC = 2;
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t SCHEDULE_ID_DATA_COPY_SMALL_C = 0;
constexpr uint64_t SCHEDULE_ID_DATA_COPY_BIG_C = 1;
constexpr uint64_t SCHEDULE_ID_DATA_COPY_JH_C = 2;
constexpr uint64_t SCHEDULE_ID_SIMT_COMMON = 3;
constexpr uint64_t SCHEDULE_ID_SIMT_INPUT_EQ_OUTPUT = 4;
constexpr uint64_t SCHEDULE_ID_SIMT_INPUT_EQ_ONE = 5;
constexpr float ENLARGE_SCALE_THRESHOLD = 4;
constexpr float REDUCE_SCALE_THRESHOLD = 0.25;
constexpr int64_t ONE_BLOCK_SIZE = 32;
constexpr int64_t UNIT_PROC_BYTES = 256;
constexpr int64_t TEMPLATE02_C_DIM_LOWER = 64;
constexpr float HALF_PIXEL = 0.5;
constexpr int64_t MIN_C_SIZE = 128;
constexpr int32_t DIM_1 = 1;
constexpr int64_t NUM_2 = 2;
constexpr int64_t NUM_3 = 3;
constexpr int64_t NUM_4 = 4;
constexpr int64_t NUM_5 = 5;
constexpr int64_t NUM_8 = 8;
constexpr int64_t BUST_COUNT = 4095;
constexpr int64_t INPUT_IDX_SIZE = 1;
constexpr int64_t IDX_DST_H = 0;
constexpr int64_t IDX_DST_W = 1;

class ResizeNearestNeighborV2AscendCTilingImpl {
public:
    explicit ResizeNearestNeighborV2AscendCTilingImpl(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init(const ResizeNearestNeighborV2CompileInfo* compileInfo);
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckFormatMatchDims();
    void MatchTilingStrategyAndSetTilingKey();
    int64_t CalcSrcLenByDesLen(int64_t desLen, float scale) const;
    int64_t CalcUnitWCountPerUB(int64_t unitDesWBytes);
    bool IsMatchTilingStrategy03() const;
    void DoTilingStrategyGather();
    void DoTilingGatherWLessThanUnitProc(int64_t unitDesWBytes);
    void DoTilingGatherWMoreThanUnitProc(int64_t unitDesWBytes);
    void TilingStrategy();
    void SetDimsByFormat();
    void SetScales();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus GetDtypeInfo();
    ge::graphStatus CheckInputSize();
    void FillTilingData();
    void PrintTilingData();
    int64_t CalTimes(int64_t a, int64_t b) const;
    bool IsMatchTiling_NHWC();
    void DoTilingSmallC();
    void DoTilingBigC();
    void DoTilingJHC();

private:
    uint64_t schId_ = 0;
    uint64_t idxUseInt32_ = 0;
    int32_t dtypeSize_ = 0;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t lenC_ = 0;
    int64_t lenN_ = 0;
    int64_t lenSrcH_ = 0;
    int64_t lenSrcW_ = 0;
    int64_t lenDesH_ = 0;
    int64_t lenDesW_ = 0;
    int64_t splitBlockFactor_ = 0;
    int64_t splitBlockTailFactor_ = 0;
    int64_t alignCorners_ = 0;
    int64_t lenDstHwcNum_ = 0;
    int64_t halfPixelCenters_ = 0;
    float scaleW_ = 0.0;
    float scaleH_ = 0.0;
    float originalScaleW_ = 0.0f;
    float originalScaleH_ = 0.0f;
    // Gather mode parameters
    int64_t splitFactorDesW_ = 0;
    int64_t splitFactorTailDesW_ = 0;
    int64_t splitCountDesW_ = 0;
    int64_t splitFactorDesH_ = 0;
    int64_t splitFactorTailDesH_ = 0;
    int64_t splitCountDesH_ = 0;
    int64_t splitBlockFullCount_ = 0;
    // Copy mode parameters
    int64_t lenCAlign_ = 0;
    int64_t condition_ = 0;
    int64_t switchParams_ = 0;
    int64_t hwcNum_ = 0;
    int64_t dstHwcNum_ = 0;
    int64_t wcNum_ = 0;
    int64_t dstWcNum_ = 0;
    int64_t nLoop_ = 0;
    int64_t nLoopTimesBefore_ = 0;
    int64_t nLoopTimesLast_ = 0;
    int64_t nLoopTailLast_ = 0;
    int64_t wcLoop_ = 0;
    int64_t wcLoopTimesBefore_ = 0;
    int64_t wcLoopTailBefore_ = 0;
    int64_t wcLoopTimesLast_ = 0;
    int64_t wcLoopTailLast_ = 0;
    int64_t onceUbNum_ = 0;
    int64_t hwNum_ = 0;
    int64_t splitBlockFactorTail_ = 0;
    bool isAlign_ = false;
    int64_t maxUbNum_ = 0;
    bool cutNd_ = true;
    ge::DataType dtype_ = ge::DT_MAX;
    ge::Format format_ = ge::FORMAT_MAX;
    gert::Shape xShape_;
    gert::Shape yShape_;
    gert::TilingContext* context_ = nullptr;
    ResizeNearestNeighborV2TilingData tilingData_;
};

ge::graphStatus ResizeNearestNeighborV2AscendCTilingImpl::CheckFormatMatchDims() {
    OP_CHECK_IF(alignCorners_ && halfPixelCenters_,
                    OP_LOGE(context_->GetNodeName(),
                    "alignCorners and halfPixelCenters is all true, not support"),
                    return ge::GRAPH_FAILED);
    condition_ = alignCorners_ ? NUM_2 : 0;
    if ((format_ == ge::FORMAT_NCHW) || (format_ == ge::FORMAT_NHWC) || (format_ == ge::FORMAT_ND)) {
      OP_CHECK_IF(xShape_.GetDimNum() != DIM_LEN_4D || yShape_.GetDimNum() != DIM_LEN_4D,
                      OP_LOGE(context_->GetNodeName(), "format dismatch dims. format:%s, shape:%s.",
                            Ops::Base::ToString(format_).c_str(), Ops::Base::ToString(xShape_).c_str()),
                      return ge::GRAPH_FAILED);
    }

    const int64_t inputSize = xShape_.GetShapeSize();
    const int64_t outSize = yShape_.GetShapeSize();
    OP_CHECK_IF(inputSize == 0 || outSize == 0,
                    OP_LOGE(context_->GetNodeName(),
                                                    "input or output size is zero"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ResizeNearestNeighborV2AscendCTilingImpl::SetDimsByFormat() {
    lenN_ = xShape_.GetDim(N_DIM_IDX);
    if ((format_ == ge::FORMAT_NCHW) || (format_ == ge::FORMAT_ND)) {
      lenC_ = xShape_.GetDim(C_DIM_IDX_NCHW);
      lenSrcH_ = xShape_.GetDim(H_DIM_IDX_NCHW);
      lenDesH_ = yShape_.GetDim(H_DIM_IDX_NCHW);
      lenSrcW_ = xShape_.GetDim(W_DIM_IDX_NCHW);
      lenDesW_ = yShape_.GetDim(W_DIM_IDX_NCHW);
    } else if (format_ == ge::FORMAT_NHWC) {
      lenC_ = xShape_.GetDim(C_DIM_IDX_NHWC);
      lenSrcH_ = xShape_.GetDim(H_DIM_IDX_NHWC);
      lenDesH_ = yShape_.GetDim(H_DIM_IDX_NHWC);
      lenSrcW_ = xShape_.GetDim(W_DIM_IDX_NHWC);
      lenDesW_ = yShape_.GetDim(W_DIM_IDX_NHWC);
    }

    OP_LOGI(context_->GetNodeName(), "lenN_:%ld , lenC_: %ld, srcH:%ld, srcW:%ld, dstH:%ld, dstW:%ld",
            lenN_, lenC_, lenSrcH_, lenSrcW_, lenDesH_, lenDesW_);
    wcNum_ = lenSrcW_ * lenC_;
    dstWcNum_ = lenDesW_ * lenC_;
    hwcNum_ = lenSrcH_ * wcNum_;
    dstHwcNum_ = lenDesH_ * dstWcNum_;
    isAlign_ = false;
    if ((lenC_ * dtypeSize_) % ONE_BLOCK_SIZE == 0) {
      isAlign_ = true;
    }
    lenCAlign_ = ((lenC_ * dtypeSize_ + ONE_BLOCK_SIZE - 1) / ONE_BLOCK_SIZE * ONE_BLOCK_SIZE) / dtypeSize_;
}

void ResizeNearestNeighborV2AscendCTilingImpl::SetScales() {
    if (alignCorners_) {
        if (lenDesH_ > 1) {
            scaleH_ = static_cast<float>(lenSrcH_ - 1) / (lenDesH_ - 1);
        } else {
            scaleH_ = 0.0f;
        }
        if (lenDesW_ > 1) {
            scaleW_ = static_cast<float>(lenSrcW_ - 1) / (lenDesW_ - 1);
        } else {
            scaleW_ = 0.0f;
        }
    } else {
        if (originalScaleH_ > 0.0f) {
            scaleH_ = 1.0f / originalScaleH_;
        } else {
            scaleH_ = static_cast<float>(lenSrcH_) / lenDesH_;
        }
        if (originalScaleW_ > 0.0f) {
            scaleW_ = 1.0f / originalScaleW_;
        } else {
            scaleW_ = static_cast<float>(lenSrcW_) / lenDesW_;
        }
    }
    OP_LOGI(context_->GetNodeName(), "SetScales scaleH_:%f , scaleW_:%f", scaleH_, scaleW_);
}

bool ResizeNearestNeighborV2AscendCTilingImpl::IsMatchTilingStrategy03() const {
    // When C is the last dim, C'len must be 1.
    if ((format_ == ge::FORMAT_NHWC) && (lenC_ != 1)) {
        return false;
    }
    // The scale must be within the range [0.25, 4]
    if ((scaleW_ > ENLARGE_SCALE_THRESHOLD) || (scaleW_ < REDUCE_SCALE_THRESHOLD) ||
        (scaleH_ > ENLARGE_SCALE_THRESHOLD) || (scaleH_ < REDUCE_SCALE_THRESHOLD)) {
        return false;
    }

    return true;
}

int64_t ResizeNearestNeighborV2AscendCTilingImpl::CalcSrcLenByDesLen(int64_t desLen, float scale) const {
    float newDesLen = static_cast<float>(desLen);
    if (halfPixelCenters_) {
        newDesLen += HALF_PIXEL;
    }

    int64_t srcLen = 0;
    if (alignCorners_) {
        srcLen = static_cast<int64_t>(newDesLen * scale);
    } else {
        srcLen = std::ceil(newDesLen * scale);
    }

    if (srcLen == 0) {
        srcLen = 1;
    }
    return srcLen;
}

int64_t ResizeNearestNeighborV2AscendCTilingImpl::CalcUnitWCountPerUB(int64_t unitDesWBytes) {
    int64_t procUnitSrcWCount = CalcSrcLenByDesLen(unitDesWBytes / dtypeSize_, scaleW_);
    int64_t srcWBytesAlignBlockSize = (procUnitSrcWCount > lenSrcW_) ?
                                      Ops::Base::CeilAlign(lenSrcW_ * dtypeSize_, ONE_BLOCK_SIZE) :
                                      Ops::Base::CeilAlign(procUnitSrcWCount * dtypeSize_, ONE_BLOCK_SIZE);
    // 输出H轴先只切1
    int64_t unitSrcHCount = CalcSrcLenByDesLen(1, scaleH_);
    // UB内预留搬入块所需空间
    int64_t unitSrcBytes = srcWBytesAlignBlockSize * unitSrcHCount * dtypeSize_;
    // UB内预留坐标计算所需空间
    int64_t unitLocationBytes = Ops::Base::CeilAlign(static_cast<int64_t>(1 * GetSizeByDataType(ge::DT_FLOAT)),  ONE_BLOCK_SIZE);
    int64_t unitWTotalBytes = unitDesWBytes + unitSrcBytes + unitLocationBytes;
    OP_LOGI(context_->GetNodeName(), "CalcUnitWCountPerUB: srcWFactor: %ld, srcHFactor: %ld, \
            unitDesWBytes: %ld, unitSrcBytes: %ld, unitLocationBytes: %ld",
            procUnitSrcWCount, unitSrcHCount, unitDesWBytes, unitSrcBytes, unitLocationBytes);
    return Ops::Base::FloorDiv((ubSize_ - ONE_BLOCK_SIZE) / NUM_2, unitWTotalBytes);
}

void ResizeNearestNeighborV2AscendCTilingImpl::DoTilingGatherWLessThanUnitProc(int64_t unitDesWBytes) {
    int64_t unitWCountPerUB = CalcUnitWCountPerUB(unitDesWBytes);
    if (unitWCountPerUB >= lenDesH_) {
        splitCountDesW_ = 1;
        splitFactorDesW_ = lenDesW_;
        splitFactorTailDesW_ = splitFactorDesW_;
        splitCountDesH_ = 1;
        splitFactorDesH_ = lenDesH_;
        splitFactorTailDesH_ = splitFactorDesH_;
    } else {
        splitCountDesW_ = 1;
        splitFactorDesW_ = lenDesW_;
        splitFactorTailDesW_ = splitFactorDesW_;
        splitCountDesH_ = Ops::Base::CeilDiv(lenDesH_, unitWCountPerUB);
        splitFactorDesH_ = unitWCountPerUB;
        splitFactorTailDesH_ = lenDesH_ - splitFactorDesH_ * (splitCountDesH_ - 1);
    }
}

void ResizeNearestNeighborV2AscendCTilingImpl::DoTilingGatherWMoreThanUnitProc(int64_t unitDesWBytes) {
    int64_t unitWCountPerUB = CalcUnitWCountPerUB(unitDesWBytes);
    bool isExceedWSize = false;
    while ((unitWCountPerUB > lenDesH_) && !isExceedWSize) {
        OP_LOGI(context_->GetNodeName(),
                "BEGIN DoTilingGatherWMoreThanUnitProc: unitWCountPerUB: %ld", unitWCountPerUB);
        int64_t unitDesWBytesAdd = unitDesWBytes + UNIT_PROC_BYTES;
        if (unitDesWBytesAdd <= lenDesW_ * dtypeSize_) {
            unitDesWBytes += UNIT_PROC_BYTES;
        } else {
            isExceedWSize = true;
            unitDesWBytes = Ops::Base::CeilAlign(lenDesW_ * dtypeSize_, ONE_BLOCK_SIZE);
        }
        unitWCountPerUB = CalcUnitWCountPerUB(unitDesWBytes);
    }

    if (isExceedWSize) {
        splitCountDesW_ = 1;
        splitFactorDesW_ = lenDesW_;
        splitFactorTailDesW_ = splitFactorDesW_;
    } else {
        splitCountDesW_ = Ops::Base::CeilDiv(lenDesW_, unitDesWBytes / dtypeSize_);
        splitFactorDesW_ = unitDesWBytes / dtypeSize_;
        splitFactorTailDesW_ = lenDesW_ - splitFactorDesW_ * (splitCountDesW_ - 1);
    }
    splitCountDesH_ = Ops::Base::CeilDiv(lenDesH_, unitWCountPerUB);
    splitFactorDesH_ = (unitWCountPerUB > lenDesH_) ? lenDesH_ : unitWCountPerUB;
    splitFactorTailDesH_ = lenDesH_ - splitFactorDesH_ * (splitCountDesH_ - 1);
}

void ResizeNearestNeighborV2AscendCTilingImpl::DoTilingStrategyGather() {
    int64_t blockSplitTotal = 0;
    int64_t numPerCore = 0;

    blockSplitTotal = lenN_ * lenC_;
    // 计算block切分数据
    realCoreNum_ = (blockSplitTotal < coreNum_) ? blockSplitTotal : coreNum_;
    splitBlockFullCount_ = blockSplitTotal % realCoreNum_;
    numPerCore = blockSplitTotal / realCoreNum_;
    splitBlockTailFactor_ = numPerCore;
    splitBlockFactor_ = splitBlockFullCount_ == 0 ? 0 : numPerCore + 1;

    int64_t unitDesWBytes = Ops::Base::CeilAlign(lenDesW_ * dtypeSize_, ONE_BLOCK_SIZE);
    if (unitDesWBytes <= UNIT_PROC_BYTES) {
        DoTilingGatherWLessThanUnitProc(unitDesWBytes);
    } else {
        unitDesWBytes = UNIT_PROC_BYTES;
        DoTilingGatherWMoreThanUnitProc(unitDesWBytes);
    }
}

bool ResizeNearestNeighborV2AscendCTilingImpl::IsMatchTiling_NHWC() {
  OP_CHECK_IF((format_ != ge::FORMAT_NHWC),
                  OP_LOGI(context_->GetNodeName(), "format is not eligible"),
                  return false);
  OP_CHECK_IF((lenC_ * dtypeSize_ < MIN_C_SIZE),
                  OP_LOGI(context_->GetNodeName(), "c is small"),
                  return false);

  maxUbNum_ = (((ubSize_ / NUM_2) / ONE_BLOCK_SIZE) * ONE_BLOCK_SIZE) / dtypeSize_;
  OP_LOGI(context_->GetNodeName(), "maxUbNum_ is %ld", maxUbNum_);
  splitBlockFactor_ = (lenN_ + coreNum_ - 1) / coreNum_;
  realCoreNum_ = (lenN_ + splitBlockFactor_ - 1) / splitBlockFactor_;
  splitBlockFactorTail_ = lenN_ - (realCoreNum_ - 1) * splitBlockFactor_;
  onceUbNum_ = splitBlockFactor_ * lenCAlign_;
  hwNum_ = lenDesH_ * lenDesW_;
  int64_t hwfactor = (hwNum_ + coreNum_ - 1) / coreNum_;
  int64_t hwcore = (hwNum_ + hwfactor - 1) / hwfactor;
  if (((hwfactor >= splitBlockFactor_) && (hwcore >= realCoreNum_)) || (lenN_ * lenCAlign_ <= maxUbNum_)) {
    cutNd_ = false;
  }
  return true;
}

void ResizeNearestNeighborV2AscendCTilingImpl::DoTilingSmallC() {
    switchParams_ = cutNd_ ? 0 : NUM_2;
    nLoop_ = maxUbNum_ / lenCAlign_;
    nLoop_ = nLoop_ <= BUST_COUNT ? nLoop_ : BUST_COUNT;
    if (cutNd_) {
      nLoopTimesBefore_ = CalTimes(splitBlockFactor_, nLoop_);
      splitBlockTailFactor_ = splitBlockFactor_ - nLoopTimesBefore_ * nLoop_;
      nLoopTimesLast_ = CalTimes(splitBlockFactorTail_, nLoop_);
      nLoopTailLast_ = splitBlockFactorTail_ - nLoopTimesLast_ * nLoop_;
      ubSize_ = onceUbNum_ * dtypeSize_;
      if (onceUbNum_ > maxUbNum_) {
        ubSize_ = nLoop_ * lenCAlign_ * dtypeSize_;
      }
    } else {
      nLoopTimesBefore_ = CalTimes (lenN_, nLoop_);
      splitBlockTailFactor_ = lenN_ - nLoopTimesBefore_ * nLoop_;
      nLoopTimesLast_ = nLoopTimesBefore_;
      nLoopTailLast_ = splitBlockTailFactor_;
      splitBlockFactor_ = (hwNum_ + coreNum_ - 1) / coreNum_;
      realCoreNum_ = (hwNum_ + splitBlockFactor_ - 1) / splitBlockFactor_;
      wcLoopTailBefore_ = splitBlockFactor_;
      wcLoopTailLast_ = hwNum_ - wcLoopTailBefore_ * (realCoreNum_ - 1);
      ubSize_ = nLoop_ * lenCAlign_* dtypeSize_;
    }
    wcLoopTimesBefore_ = splitBlockFactor_ * hwcNum_;
    wcLoop_ = splitBlockFactor_ * dstHwcNum_;
    wcLoopTimesLast_ = nLoop_ * hwcNum_;
    lenCAlign_ = nLoop_ * dstHwcNum_;
}

int64_t ResizeNearestNeighborV2AscendCTilingImpl::CalTimes(int64_t a, int64_t b) const {
    OP_CHECK_IF(b == 0,
        OP_LOGE(context_->GetNodeName(), "b = 0 is not support"),
        return -1);  
    int64_t c = a / b;
    if (a % b == 0) {
        c = c -1;
    }
    return c;
}

void ResizeNearestNeighborV2AscendCTilingImpl::DoTilingBigC() {
    switchParams_ = cutNd_ ? 1 : NUM_3;
    nLoop_ = 1;
    ubSize_ = maxUbNum_ * dtypeSize_;
    wcLoop_ = maxUbNum_;
    if (switchParams_ == 1) {
      nLoopTimesBefore_ = splitBlockFactor_;
      nLoopTimesLast_ = splitBlockFactorTail_;
    } else {
      splitBlockFactor_ = (hwNum_ + coreNum_ - 1) / coreNum_;
      realCoreNum_ = (hwNum_ + splitBlockFactor_ - 1) / splitBlockFactor_;
      splitBlockFactorTail_ = hwNum_ - (realCoreNum_ - 1) * splitBlockFactor_;
      nLoopTimesBefore_ = lenN_;
      nLoopTimesLast_ = lenN_;
      splitBlockTailFactor_ = splitBlockFactor_;
      nLoopTailLast_  = splitBlockFactorTail_;
    }
    wcLoopTimesBefore_ = CalTimes(lenC_, wcLoop_);
    wcLoopTailBefore_ = lenC_ - wcLoopTimesBefore_ * wcLoop_;
    wcLoopTimesLast_ = splitBlockFactor_ * hwcNum_;
    wcLoopTailLast_ = splitBlockFactor_ * dstHwcNum_;
}

void ResizeNearestNeighborV2AscendCTilingImpl::DoTilingJHC() {
    switchParams_ = cutNd_ ? NUM_4 : NUM_5;
    if (switchParams_ == NUM_4) {
      if (splitBlockFactor_*  lenCAlign_ < maxUbNum_) {
        nLoop_ = splitBlockFactor_ <= BUST_COUNT ? splitBlockFactor_ : BUST_COUNT;
        wcLoop_ = maxUbNum_ / (nLoop_* lenCAlign_);
      } else {
        nLoop_ = maxUbNum_ / lenCAlign_;
        wcLoop_ = 1;
      }
      nLoopTimesBefore_ = CalTimes (splitBlockFactor_, nLoop_);
      splitBlockTailFactor_ = splitBlockFactor_ - nLoopTimesBefore_ * nLoop_;
      nLoopTimesLast_ = CalTimes(splitBlockFactorTail_, nLoop_);
      nLoopTailLast_ = splitBlockFactorTail_ - nLoopTimesLast_ * nLoop_;
      wcLoopTimesBefore_ = CalTimes(hwNum_, wcLoop_);
      wcLoopTailBefore_ = hwNum_ - wcLoopTimesBefore_ * wcLoop_;
      wcLoopTimesLast_ = wcLoopTimesBefore_;
      wcLoopTailLast_ = wcLoopTailBefore_;
    } else {
      splitBlockFactor_ = (hwNum_ + coreNum_ - 1) / coreNum_;
      realCoreNum_ = (hwNum_ + splitBlockFactor_ - 1) / splitBlockFactor_;
      splitBlockFactorTail_ = hwNum_ - (realCoreNum_ - 1) * splitBlockFactor_;
      if (lenN_ * lenCAlign_ < maxUbNum_) {
        nLoop_ = lenN_ <= BUST_COUNT ? lenN_ : BUST_COUNT;
        nLoopTimesBefore_ = CalTimes (lenN_, nLoop_);
        splitBlockTailFactor_ = lenN_ - nLoopTimesBefore_ * nLoop_;
        nLoopTimesLast_ = nLoopTimesBefore_;
        nLoopTailLast_ = splitBlockTailFactor_;
        wcLoop_ = maxUbNum_ / (nLoop_ * lenCAlign_);
      } else {
        nLoop_ = maxUbNum_ / lenCAlign_;
        nLoopTimesBefore_ = CalTimes (lenN_, nLoop_);
        splitBlockTailFactor_ = lenN_ - nLoopTimesBefore_ * nLoop_;
        nLoopTimesLast_ = CalTimes(lenN_, nLoop_);
        nLoopTailLast_ = lenN_ - nLoopTimesLast_ * nLoop_;
        wcLoop_ = 1;
      }
      wcLoopTimesBefore_ = CalTimes(splitBlockFactor_, wcLoop_);
      wcLoopTailBefore_ = splitBlockFactor_ - wcLoopTimesBefore_ * wcLoop_;
      wcLoopTimesLast_ = CalTimes(splitBlockFactorTail_, wcLoop_);
      wcLoopTailLast_ = splitBlockFactorTail_ - wcLoopTimesLast_ * wcLoop_;
    }
    ubSize_ = nLoop_ * wcLoop_ * lenCAlign_ * dtypeSize_;
    lenCAlign_ = nLoop_ * hwcNum_;
    lenDstHwcNum_ = nLoop_ * dstHwcNum_;
    lenN_ = splitBlockFactor_ * dstHwcNum_;
}

void ResizeNearestNeighborV2AscendCTilingImpl::MatchTilingStrategyAndSetTilingKey() {
    if (IsMatchTiling_NHWC()) {
        if(isAlign_ && ((lenN_ * lenCAlign_ < maxUbNum_) || (lenC_ * dtypeSize_ < UNIT_PROC_BYTES))) {
            schId_ = SCHEDULE_ID_DATA_COPY_JH_C;
        } else if (lenCAlign_ <= maxUbNum_) {
            schId_ = SCHEDULE_ID_DATA_COPY_SMALL_C;
        } else {
            schId_ = SCHEDULE_ID_DATA_COPY_BIG_C;
        }
    } else {
        idxUseInt32_ = xShape_.GetShapeSize() < UINT32_MAX && yShape_.GetShapeSize() < UINT32_MAX;
        if (lenSrcH_ == lenDesH_ && lenSrcW_ == lenDesW_) {
            schId_ = SCHEDULE_ID_SIMT_INPUT_EQ_OUTPUT;
        } else if (lenSrcH_ == DIM_1 && lenSrcW_ == DIM_1) {
            schId_ = SCHEDULE_ID_SIMT_INPUT_EQ_ONE;
        } else {
            schId_ = SCHEDULE_ID_SIMT_COMMON;
        }
    }
}

void ResizeNearestNeighborV2AscendCTilingImpl::TilingStrategy()
{
    MatchTilingStrategyAndSetTilingKey();
    switch (schId_) {
        case SCHEDULE_ID_DATA_COPY_SMALL_C: {
            DoTilingSmallC();
            break;
        }
        case SCHEDULE_ID_DATA_COPY_BIG_C: {
            DoTilingBigC();
            break;
        }
        case SCHEDULE_ID_DATA_COPY_JH_C: {
            DoTilingJHC();
            break;
        }
        default: {
            realCoreNum_ = (yShape_.GetShapeSize() < coreNum_) ? yShape_.GetShapeSize() : coreNum_;
            splitBlockFactor_ = Ops::Base::FloorDiv(yShape_.GetShapeSize(), realCoreNum_);
            splitBlockTailFactor_ = yShape_.GetShapeSize() - splitBlockFactor_ * realCoreNum_;
            break;
        }
    }
}

void ResizeNearestNeighborV2AscendCTilingImpl::FillTilingData() {
    tilingData_.set_realCoreNum(realCoreNum_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_alignCorners(lenDstHwcNum_);
    tilingData_.set_halfPixelCenters(halfPixelCenters_);
    tilingData_.set_lenN(lenN_);
    tilingData_.set_lenC(lenC_);
    tilingData_.set_lenSrcH(lenSrcH_);
    tilingData_.set_lenSrcW(lenSrcW_);
    tilingData_.set_lenDesH(lenDesH_);
    tilingData_.set_lenDesW(lenDesW_);
    tilingData_.set_condition(condition_);
    tilingData_.set_switchParams(switchParams_);
    tilingData_.set_splitBlockFactor(splitBlockFactor_);
    tilingData_.set_splitBlockTailFactor(splitBlockTailFactor_);
    tilingData_.set_lenCAlign(lenCAlign_);
    tilingData_.set_hwcNum(hwcNum_);
    tilingData_.set_dstHwcNum(dstHwcNum_);
    tilingData_.set_wcNum(wcNum_);
    tilingData_.set_dstWcNum(dstWcNum_);
    tilingData_.set_nLoop(nLoop_);
    tilingData_.set_nLoopTimesBefore(nLoopTimesBefore_);
    tilingData_.set_nLoopTimesLast(nLoopTimesLast_);
    tilingData_.set_nLoopTailLast(nLoopTailLast_);
    tilingData_.set_wcLoop(wcLoop_);
    tilingData_.set_wcLoopTimesBefore(wcLoopTimesBefore_);
    tilingData_.set_wcLoopTailBefore(wcLoopTailBefore_) ;
    tilingData_.set_wcLoopTimesLast(wcLoopTimesLast_);
    tilingData_.set_wcLoopTailLast(wcLoopTailLast_);
    tilingData_.set_splitBlockFullCount(splitBlockFullCount_);
    tilingData_.set_splitFactorDesH(splitFactorDesH_);
    tilingData_.set_splitFactorTailDesH(splitFactorTailDesH_);
    tilingData_.set_splitCountDesH(splitCountDesH_);
    tilingData_.set_splitFactorDesW(splitFactorDesW_);
    tilingData_.set_splitFactorTailDesW(splitFactorTailDesW_);
    tilingData_.set_splitCountDesW(splitCountDesW_);
    tilingData_.set_scaleW(scaleW_);
    tilingData_.set_scaleH(scaleH_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void ResizeNearestNeighborV2AscendCTilingImpl::PrintTilingData() {
    OP_LOGI(context_->GetNodeName(),
            "tilingData is realCoreNum:%ld, ubSize: %ld, alignCorners:%ld, halfPixelCenters:%ld, \
            lenN:%ld, lenC:%ld, lenSrcH:%ld, lenSrcW:%ld, lenDesH:%ld, lenDesW:%ld, \
            condition:%ld, switchParams:%ld, splitBlockFactor:%ld, splitBlockTailFactor: %ld, lenCAlign: %ld, \
            hwcNum: %ld, dstHwcNum:%ld, wcNum:%ld, dstWcNum:%ld, nLoop: %ld, nLoopTimesBefore: %ld, \
            nLoopTimesLast is %ld, nLoopTailLast: %ld, wcLoop: %ld, wcLoopTimesBefore: %ld, \
            wcLoopTailBefore: %ld, wcLoopTimesLast: %ld, wcLoopTailLast: %ld, scaleW: %f, scaleH：%f",
            tilingData_.get_realCoreNum(),
            tilingData_.get_ubSize(),
            tilingData_.get_alignCorners(),
            tilingData_.get_halfPixelCenters(),
            tilingData_.get_lenN(),
            tilingData_.get_lenC(),
            tilingData_.get_lenSrcH(),
            tilingData_.get_lenSrcW(),
            tilingData_.get_lenDesH(),
            tilingData_.get_lenDesW(),
            tilingData_.get_condition(),
            tilingData_.get_switchParams(),
            tilingData_.get_splitBlockFactor(),
            tilingData_.get_splitBlockTailFactor(),
            tilingData_.get_lenCAlign(),
            tilingData_.get_hwcNum(),
            tilingData_.get_dstHwcNum(),
            tilingData_.get_wcNum(),
            tilingData_.get_dstWcNum(),
            tilingData_.get_nLoop(),
            tilingData_.get_nLoopTimesBefore(),
            tilingData_.get_nLoopTimesLast(),
            tilingData_.get_nLoopTailLast(),
            tilingData_.get_wcLoop(),
            tilingData_.get_wcLoopTimesBefore(),
            tilingData_.get_wcLoopTailBefore(),
            tilingData_.get_wcLoopTimesLast(),
            tilingData_.get_wcLoopTailLast(),
            tilingData_.get_scaleW(),
            tilingData_.get_scaleH());
}

ge::graphStatus ResizeNearestNeighborV2AscendCTilingImpl::CheckInputSize()
{
    auto size = context_->GetInputShape(INPUT_IDX_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, size);
    gert::Shape sizeShape = size->GetStorageShape();
    int32_t sizeDims = sizeShape.GetShapeSize();
    OP_CHECK_IF(sizeDims != NUM_2,
        OP_LOGE(context_->GetNodeName(), "size shape dims is not two"), return ge::GRAPH_FAILED);
    const gert::Tensor *sizeTensor = context_->GetInputTensor(DIM_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sizeTensor);
    OP_CHECK_IF(sizeTensor->GetDataType() != ge::DT_INT32,
        OP_LOGE(context_->GetNodeName(), "size dtype only support int32"),
        return ge::GRAPH_FAILED);
    std::vector<int64_t> sizeList(NUM_2);
    auto *tensorData = sizeTensor->GetData<int32_t>();
    OP_CHECK_IF(tensorData == nullptr,
        OP_LOGE(context_->GetNodeName(), "tensorData is nullptr"), return ge::GRAPH_FAILED);

    for (int32_t i = 0; i < NUM_2; i++) {
        sizeList[i] = static_cast<int64_t>(*(tensorData + i));
    }
    OP_CHECK_IF((sizeList[0] != lenDesH_) || (sizeList[1] != lenDesW_),
        OP_LOGE(context_->GetNodeName(), "size not equal output h w"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(lenDesH_ <= 0 || lenDesW_ <= 0 || lenSrcH_ <= 0 || lenSrcW_ <= 0,
        OP_LOGE(context_->GetNodeName(), "input h w and output h w must greater than zero"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(lenC_ <= 0 || lenN_ <= 0,
        OP_LOGE(context_->GetNodeName(), "n and c must greater than zero"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeNearestNeighborV2AscendCTilingImpl::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    if (attrs->GetAttrNum() > ATTR_ALIGN_CORERS_IDX) {
        alignCorners_ = *(attrs->GetAttrPointer<bool>(ATTR_ALIGN_CORERS_IDX)) ? 1 : 0;
    }
    if (attrs->GetAttrNum() > ATTR_HALF_PIXEL_CENTERS_IDX) {
        halfPixelCenters_ = *(attrs->GetAttrPointer<bool>(ATTR_HALF_PIXEL_CENTERS_IDX)) ? 1 : 0;
    }
    OP_CHECK_IF(
        alignCorners_ && halfPixelCenters_,
        OP_LOGE(context_->GetNodeName(), "alignCorners and halfPixelCenters do not support both being true"),
        return ge::GRAPH_FAILED);

    if (attrs->GetAttrNum() > ATTR_SCALES_IDX) {
        auto scales = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SCALES_IDX);
        int64_t scalesNum = scales->GetSize();
        OP_CHECK_IF(scalesNum != SCALES_NUM,
                        OP_LOGE(context_->GetNodeName(), "scales size %ld is invalid.", scalesNum),
                        return ge::GRAPH_FAILED);
        const float* scalesData = reinterpret_cast<const float*>(scales->GetData());
        OP_CHECK_NULL_WITH_CONTEXT(context_, scalesData);
        originalScaleH_ = scalesData[SCALE_H];
        originalScaleW_ = scalesData[SCALE_W];
        OP_LOGI(context_->GetNodeName(), "original scales(%f, %f)", originalScaleH_, originalScaleW_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeNearestNeighborV2AscendCTilingImpl::GetDtypeInfo()
{
    // Get dtype, format
    auto inputXDesc = context_->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto outputXDesc = context_->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputXDesc);
    dtype_ = inputXDesc->GetDataType();
    format_ = static_cast<ge::Format>(ge::GetPrimaryFormat(inputXDesc->GetStorageFormat()));
    dtypeSize_ = GetSizeByDataType(dtype_);
    auto outputDtypeSize = GetSizeByDataType(outputXDesc->GetDataType());
    OP_CHECK_IF(dtypeSize_ <= 0 || outputDtypeSize <= 0 || dtype_ != outputXDesc->GetDataType(),
                    OP_LOGE(context_->GetNodeName(), "Input or output dtype is invalid."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(format_ != static_cast<ge::Format>(ge::GetPrimaryFormat(outputXDesc->GetStorageFormat())),
                    OP_LOGE(context_->GetNodeName(), "Input or output format is invalid."),
                    return ge::GRAPH_FAILED);

    // Get xshape, yshape
    auto xStorage = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    xShape_ = Ops::Cv::OpTiling::EnsureNotScalar(xStorage->GetStorageShape());
    auto yStorage = context_->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yStorage);

    yShape_ = Ops::Cv::OpTiling::EnsureNotScalar(yStorage->GetStorageShape());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeNearestNeighborV2AscendCTilingImpl::Init(const ResizeNearestNeighborV2CompileInfo* compileInfo) {
    OP_LOGD(context_->GetNodeName(), "Enter ResizeNearestNeighborV2AscendCTilingImpl init.");

    coreNum_ = compileInfo->core_num;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(coreNum_ <= 0 || ubSize_ <= 0,
                    OP_LOGE(context_->GetNodeName(), "coreNum or ubSize is small than zero"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "coreNum_ is %ld, ubSize_ is %ld", coreNum_, ubSize_);
    // Get attrs: alignCorners, halfPixelCenters, scales
    OP_CHECK_IF((GetAttrInfo() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "GetAttrInfo failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((GetDtypeInfo() != ge::GRAPH_SUCCESS), OP_LOGE(context_->GetNodeName(), "GetDtypeInfo failed."),
        return ge::GRAPH_FAILED);

    // Check format and dims match or not.
    if (CheckFormatMatchDims() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Set N,C,H,W dim length
    SetDimsByFormat();

    // Check input size
    OP_CHECK_IF(CheckInputSize() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "get input size failed"),
        return ge::GRAPH_FAILED);

    // compute new scales
    SetScales();
    OP_LOGD(context_->GetNodeName(), "Exit ResizeNearestNeighborV2AscendCTilingImpl init.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResizeNearestNeighborV2AscendCTilingImpl::DoTiling() {
    OP_LOGD(context_->GetNodeName(), "Enter ResizeNearestNeighborV2AscendCTilingImpl DoTiling.");

    TilingStrategy();
    FillTilingData();
    PrintTilingData();

    context_->SetBlockDim(tilingData_.get_realCoreNum());
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId_, (uint64_t)format_, (uint64_t)alignCorners_,
                                                  (uint64_t)halfPixelCenters_, idxUseInt32_);
    OP_LOGI(context_->GetNodeName(),
            "schId is %ld, format is %ld, alignCorners is %ld, halfPixelCenters is %ld, idxUseInt32 is %ld", 
            static_cast<int64_t>(schId_), static_cast<int64_t>(format_), static_cast<int64_t>(alignCorners_), 
            static_cast<int64_t>(halfPixelCenters_), static_cast<int64_t>(idxUseInt32_));
    context_->SetTilingKey(tilingKey);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_SIZE;
    OP_LOGD(context_->GetNodeName(), "Exit ResizeNearestNeighborV2AscendCTilingImpl DoTiling.");
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus Tiling4ResizeNearestNeighborV2ForAscendC(gert::TilingContext* context,
                                                           const ResizeNearestNeighborV2CompileInfo* compileInfo) {
    OP_LOGD(context->GetNodeName(), "Start Tiling4ResizeNearestNeighborV2ForAscendC.");

    ResizeNearestNeighborV2AscendCTilingImpl tilingImpl = ResizeNearestNeighborV2AscendCTilingImpl(context);
    if (tilingImpl.Init(compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4ResizeNearestNeighborV2ForAscendC init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4ResizeNearestNeighborV2ForAscendC do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}
