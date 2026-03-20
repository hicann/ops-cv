/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BARYCENTRIC_FROM_IMGCOORD_BASE_AIV_H
#define BARYCENTRIC_FROM_IMGCOORD_BASE_AIV_H

#include <cstdint>
#include <limits>
#include "kernel_operator.h"

using namespace AscendC;

namespace BarycentricFromImgcoord {

constexpr uint8_t BUFFER_NUM = 2;

constexpr uint32_t ELENUM_BLOCK_FP32 = 8;
constexpr uint32_t ELENUM_REPEAT_FP32 = 64;

constexpr uint32_t MAX_PROC_ELENUM = 1920; // max proc pix num
constexpr uint32_t TRANS_COL_ELENUM = 96; // trans pix num

constexpr uint32_t FACE_VERT_NUM = 3;
constexpr uint32_t VERT_COORD_NUM = 4;
constexpr uint32_t BARY_COORD_NUM = 3;

constexpr float SCREEN_PIX_COEF = 0.5f;

template <typename T>
class BarycentricFromImgcoordAIV {
public:
    __aicore__ inline BarycentricFromImgcoordAIV(){};
    __aicore__ inline void Init(GM_ADDR v, GM_ADDR f,
        GM_ADDR findices, GM_ADDR barycentric, GM_ADDR workspace, RasterizerTilingData *tilingData);
    __aicore__ inline void Process();

    __aicore__ inline void InitParam();
    __aicore__ inline void InitUbuf();
    __aicore__ inline void ReuseUbuf();
    __aicore__ inline void ParseTilingData(RasterizerTilingData *tilingData);

    __aicore__ inline void GenMaskDataInGM();

    __aicore__ inline void PreProcess(uint32_t loopId, uint32_t curCalPixNum);
    __aicore__ inline void CopyInDepth(uint32_t loopId, uint32_t preLoopId, uint32_t curPrePixNum,  uint32_t pingFlag);
    __aicore__ inline void TransposeDepth(uint32_t curPrePixNum,  uint32_t pingFlag);
    __aicore__ inline void ReduceMinDepth(uint32_t preLoopId, uint32_t curPrePixNum, uint32_t pingFlag);

    __aicore__ inline void GenFindicesAndPreVertData(uint32_t loopId, uint32_t curCalPixNum);
    __aicore__ inline void GetVertData(int32_t faceIdx, uint32_t pixId);

    __aicore__ inline void CalculateBarycentric(uint32_t loopId, uint32_t curCalPixNum);
    __aicore__ inline void GenScreenVertData(uint32_t loopId, uint32_t curCalPixNum);
    __aicore__ inline void CalcSignedArea(uint32_t curCalPixNum);
    __aicore__ inline void GenBarycentricCoord(uint32_t curCalPixNum);

    __aicore__ inline void CopyOutRslt(uint32_t loopId, uint32_t curCalPixNum);

private:
    TPipe pipe;

    GlobalTensor<T> invGM;
    GlobalTensor<int32_t> infGM;
    GlobalTensor<int32_t> findicesGM;
    GlobalTensor<T> barycentricGM;
    GlobalTensor<int32_t> zBufIdxGM; // zbuffer save (idx + 1)
    GlobalTensor<float> zBufDepthGM; // zbuffer save depth
    GlobalTensor<uint32_t> maskGM; // save gather mask

    TBuf<QuePosition::VECCALC> uBuf;

    LocalTensor<float> ubufLocal;
    LocalTensor<float> inDepthLocal;
    LocalTensor<float> transDepthLocal;
    LocalTensor<float> reduceMinIdxLocal;
    LocalTensor<int32_t> barycentricFlagLocal;
    LocalTensor<uint8_t> areaZeroMaskLocal;

    // origin coordinate (x, y, z, w)
    LocalTensor<float> vert0xLocal;
    LocalTensor<float> vert1xLocal;
    LocalTensor<float> vert2xLocal;

    LocalTensor<float> vert0yLocal;
    LocalTensor<float> vert1yLocal;
    LocalTensor<float> vert2yLocal;

    LocalTensor<float> vert0wLocal;
    LocalTensor<float> vert1wLocal;
    LocalTensor<float> vert2wLocal;

    // screen coordinate (x, y)
    LocalTensor<float> screenVertxLocal;
    LocalTensor<float> screenVertyLocal;
    LocalTensor<float> pixIdTmpLocal;

    LocalTensor<float> screenVert0xLocal;
    LocalTensor<float> screenVert1xLocal;
    LocalTensor<float> screenVert2xLocal;

    LocalTensor<float> screenVert0yLocal;
    LocalTensor<float> screenVert1yLocal;
    LocalTensor<float> screenVert2yLocal;

    // save barycentric (x, y, z)
    LocalTensor<float> baryCentricxLocal;
    LocalTensor<float> baryCentricyLocal;
    LocalTensor<float> baryCentriczLocal;

    uint32_t vecIdx = 0;
    uint64_t vecCoreNum = 0;

    uint32_t numFaces = 0;
    uint32_t numVertices = 0;
    int32_t height = 4096;
    int32_t width = 4096;

    float heightF32 = 4096.0f;
    float widthF32 = 4096.0f;

    uint32_t totalPixNum = 0;

    uint32_t calPixNum = 0;
    uint32_t startPixId = 0;

    uint32_t rsv = 64;
};
}  // namespace BarycentricFromImgcoord

#endif  // BARYCENTRIC_FROM_IMGCOORD_BASE_AIV_H