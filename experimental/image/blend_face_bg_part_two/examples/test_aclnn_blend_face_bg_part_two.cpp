/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

// Multi-case BlendFaceBgPartTwo verification harness; runs aclnn forward and
// compares against a host-side fp32 reference.  Self-contained for cannsim
// Ascend950 simulator runs and 910b real-NPU smoke runs.
//
// Kernel formula (fp32):
//     fused_img = (acc_face / (acc_mask + epsilon)) * max_mask
//                + bg_img * (1 - max_mask)
//
// bg_img supports FLOAT and UINT8.  fused_img remains FLOAT.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "aclnn_blend_face_bg_part_two.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto d : shape)
        s *= d;
    return s;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, void** dev, aclDataType dt,
                           aclTensor** t)
{
    auto sz = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(dev, sz, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ret;
    ret = aclrtMemcpy(*dev, sz, host.data(), sz, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
        return ret;
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *t = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                         shape.size(), *dev);
    return 0;
}

// Distribution selectors for alpha (= max_mask).
enum class AlphaDist { Uniform01, AllZero, AllOne, Bimodal };
enum class BgDType { Float32, Uint8 };

struct Case {
    const char* name;
    std::vector<int64_t> shape;
    double epsilon;
    AlphaDist alpha;
    BgDType bgDType;
    float tol;
};

struct CaseDeviceData {
    void *fDev = nullptr, *mDev = nullptr, *xDev = nullptr, *bDev = nullptr, *oDev = nullptr;
    aclTensor *fT = nullptr, *mT = nullptr, *xT = nullptr, *bT = nullptr, *oT = nullptr;
};

static void make_inputs(const Case& c, std::vector<float>& accFace, std::vector<float>& accMask,
                        std::vector<float>& maxMask, std::vector<float>& bgImg)
{
    int64_t n = GetShapeSize(c.shape);
    accFace.resize(n);
    accMask.resize(n);
    maxMask.resize(n);
    bgImg.resize(n);

    std::mt19937 rng(0xB1E11Du);
    std::uniform_real_distribution<float> faceDist(0.05f, 5.0f);
    std::uniform_real_distribution<float> maskDist(0.05f, 5.0f); // acc_mask kept >= 0.05 to mirror real pipeline
    std::uniform_real_distribution<float> bgDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> alphaDist(0.0f, 1.0f);
    std::bernoulli_distribution bimodal(0.5);

    for (int64_t i = 0; i < n; ++i) {
        accFace[i] = faceDist(rng);
        accMask[i] = maskDist(rng);
        bgImg[i] = bgDist(rng);
        switch (c.alpha) {
            case AlphaDist::Uniform01:
                maxMask[i] = alphaDist(rng);
                break;
            case AlphaDist::AllZero:
                maxMask[i] = 0.0f;
                break;
            case AlphaDist::AllOne:
                maxMask[i] = 1.0f;
                break;
            case AlphaDist::Bimodal:
                maxMask[i] = bimodal(rng) ? 1.0f : 0.0f;
                break;
        }
    }
}

static void ref_blend(const std::vector<float>& accFace, const std::vector<float>& accMask,
                      const std::vector<float>& maxMask, const std::vector<float>& bgImg, double epsilon,
                      std::vector<float>& fused)
{
    int64_t n = static_cast<int64_t>(accFace.size());
    fused.resize(n);
    float eps = static_cast<float>(epsilon);
    for (int64_t i = 0; i < n; ++i) {
        float fusionFace = accFace[i] / (accMask[i] + eps);
        fused[i] = fusionFace * maxMask[i] + bgImg[i] * (1.0f - maxMask[i]);
    }
}

static int CreateCaseTensors(const Case& c, const std::vector<float>& accFace, const std::vector<float>& accMask,
                             const std::vector<float>& maxMask, const std::vector<float>& bgImg,
                             const std::vector<float>& outHost, CaseDeviceData& data)
{
    int rc = CreateAclTensor(accFace, c.shape, &data.fDev, ACL_FLOAT, &data.fT);
    if (rc)
        return rc;
    rc = CreateAclTensor(accMask, c.shape, &data.mDev, ACL_FLOAT, &data.mT);
    if (rc)
        return rc;
    rc = CreateAclTensor(maxMask, c.shape, &data.xDev, ACL_FLOAT, &data.xT);
    if (rc)
        return rc;
    if (c.bgDType == BgDType::Uint8) {
        std::vector<uint8_t> bgImgU8(bgImg.size());
        for (size_t i = 0; i < bgImg.size(); ++i) {
            bgImgU8[i] = static_cast<uint8_t>(std::round(bgImg[i]));
        }
        rc = CreateAclTensor(bgImgU8, c.shape, &data.bDev, ACL_UINT8, &data.bT);
        if (rc)
            return rc;
    } else {
        rc = CreateAclTensor(bgImg, c.shape, &data.bDev, ACL_FLOAT, &data.bT);
        if (rc)
            return rc;
    }
    rc = CreateAclTensor(outHost, c.shape, &data.oDev, ACL_FLOAT, &data.oT);
    if (rc)
        return rc;
    return ACL_SUCCESS;
}

static int PrepareExecutor(const Case& c, const CaseDeviceData& data, uint64_t& wsSize, aclOpExecutor*& executor)
{
    int rc = aclnnBlendFaceBgPartTwoGetWorkspaceSize(data.fT, data.mT, data.xT, data.bT, c.epsilon, data.oT, &wsSize,
                                                     &executor);
    if (rc != ACL_SUCCESS) {
        LOG_PRINT("[FAIL %s] GetWorkspaceSize=%d\n", c.name, rc);
    }
    return rc;
}

static int RunExecutor(const Case& c, aclrtStream stream, uint64_t wsSize, aclOpExecutor* executor, void*& wsAddr)
{
    if (wsSize > 0) {
        aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    int rc = aclnnBlendFaceBgPartTwo(wsAddr, wsSize, executor, stream);
    if (rc != ACL_SUCCESS) {
        LOG_PRINT("[FAIL %s] aclnnBlendFaceBgPartTwo=%d\n", c.name, rc);
        return rc;
    }
    aclrtSynchronizeStream(stream);
    return ACL_SUCCESS;
}

static void CopyOutputAndCompare(const CaseDeviceData& data, const std::vector<float>& refOut, int nElems,
                                 float& maxErr)
{
    std::vector<float> dev(nElems);
    aclrtMemcpy(dev.data(), nElems * sizeof(float), data.oDev, nElems * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    maxErr = 0.0f;
    for (int i = 0; i < nElems; ++i) {
        float diff = std::fabs(dev[i] - refOut[i]);
        if (diff > maxErr)
            maxErr = diff;
    }
}

static void DestroyCaseTensors(const CaseDeviceData& data, void* wsAddr)
{
    aclDestroyTensor(data.fT);
    aclDestroyTensor(data.mT);
    aclDestroyTensor(data.xT);
    aclDestroyTensor(data.bT);
    aclDestroyTensor(data.oT);
    aclrtFree(data.fDev);
    aclrtFree(data.mDev);
    aclrtFree(data.xDev);
    aclrtFree(data.bDev);
    aclrtFree(data.oDev);
    if (wsAddr)
        aclrtFree(wsAddr);
}

static int run_case(aclrtStream stream, const Case& c, float& maxErr, int& nElems)
{
    nElems = static_cast<int>(GetShapeSize(c.shape));
    std::vector<float> accFace, accMask, maxMask, bgImg, refOut;
    make_inputs(c, accFace, accMask, maxMask, bgImg);
    if (c.bgDType == BgDType::Uint8) {
        for (size_t i = 0; i < bgImg.size(); ++i) {
            bgImg[i] = static_cast<float>((i * 37U + 11U) % 256U);
        }
    }
    ref_blend(accFace, accMask, maxMask, bgImg, c.epsilon, refOut);

    CaseDeviceData data;
    std::vector<float> outHost(nElems, 0.0f);
    int rc = CreateCaseTensors(c, accFace, accMask, maxMask, bgImg, outHost, data);
    if (rc)
        return rc;

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    rc = PrepareExecutor(c, data, wsSize, executor);
    if (rc != ACL_SUCCESS)
        return rc;

    void* wsAddr = nullptr;
    rc = RunExecutor(c, stream, wsSize, executor, wsAddr);
    if (rc != ACL_SUCCESS)
        return rc;

    CopyOutputAndCompare(data, refOut, nElems, maxErr);
    DestroyCaseTensors(data, wsAddr);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Tolerances calibrated for fp32 fused-multiply-add: max ~1 ulp ≈ 1e-6 relative.
    // Use 5e-5 absolute to absorb worst-case roundtrip on Div + Muls + Add chain.
    // Shape coverage:
    //   tiny     : 1 core, 1 tile  (numel < tileSize)
    //   single   : 1 core, multi-tile (single-core 8192 elems vs max tileSize=2048)
    //   multicore: 48 AIV cores fully utilised, several tiles per core
    //   ragged   : non-aligned numel exercises CopyIn/Out tail-block path
    //   alpha=0  : output = bg only (face contribution masked out)
    //   alpha=1  : output = (face/(mask+eps)) only (bg masked out)
    //   bimodal  : sharp transitions, covers Mul/Add precision under jumps
    Case cases[] = {
        {"scalar_min", {1, 1, 1, 1}, 1e-12, AlphaDist::Uniform01, BgDType::Float32, 5e-5f},
        {"tiny_uniform", {1, 3, 4, 4}, 1e-12, AlphaDist::Uniform01, BgDType::Float32, 5e-5f},
        {"single_core_multi", {1, 1, 64, 128}, 1e-12, AlphaDist::Uniform01, BgDType::Float32, 5e-5f},
        {"multicore_aligned", {1, 3, 256, 256}, 1e-12, AlphaDist::Uniform01, BgDType::Float32, 5e-5f},
        {"ragged_tail", {1, 3, 17, 31}, 1e-12, AlphaDist::Uniform01, BgDType::Float32, 5e-5f},
        {"alpha_all_zero", {1, 3, 32, 32}, 1e-12, AlphaDist::AllZero, BgDType::Float32, 5e-5f},
        {"alpha_all_one", {1, 3, 32, 32}, 1e-12, AlphaDist::AllOne, BgDType::Float32, 5e-5f},
        {"alpha_bimodal", {1, 3, 32, 32}, 1e-12, AlphaDist::Bimodal, BgDType::Float32, 5e-5f},
        {"epsilon_default", {1, 3, 4, 4}, 1e-12, AlphaDist::Uniform01, BgDType::Float32, 5e-5f},
        {"epsilon_loose", {1, 3, 4, 4}, 1e-3, AlphaDist::Uniform01, BgDType::Float32, 5e-5f},
        {"bg_uint8_uniform", {1, 3, 17, 31}, 1e-12, AlphaDist::Uniform01, BgDType::Uint8, 1e-4f},
    };
    const int total = sizeof(cases) / sizeof(cases[0]);
    int passed = 0, failed = 0;

    LOG_PRINT("=== BlendFaceBgPartTwo multi-case verification ===\n");
    for (int i = 0; i < total; ++i) {
        float maxErr = -1.0f;
        int nElems = 0;
        int rc = run_case(stream, cases[i], maxErr, nElems);
        bool ok = (rc == ACL_SUCCESS) && (maxErr <= cases[i].tol);
        LOG_PRINT("[%s] %-22s n=%-7d max_err=%.3e (tol=%.0e) rc=%d\n", ok ? "PASS" : "FAIL", cases[i].name, nElems,
                  maxErr, cases[i].tol, rc);
        if (ok)
            ++passed;
        else
            ++failed;
    }
    LOG_PRINT("=== SUMMARY: %d/%d PASS ===\n", passed, total);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failed == 0 ? 0 : 1;
}
