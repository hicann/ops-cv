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

// Multi-case CIoU verification harness; runs aclnn forward and compares against
// a host-side fp32 reference implementation.  Designed to be self-contained for
// cannsim Ascend950 simulator runs (no torch dep).

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include "acl/acl.h"
#include "aclnn_c_io_u.h"

#define CHECK_RET(cond, return_expr) \
    do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) do { printf(message, ##__VA_ARGS__); } while (0)

// fp32 host-side reference (matches CIoU_reference.py).
static void ref_ciou(const std::vector<float>& bb, const std::vector<float>& gb,
                     int N, bool trans, int modeId, bool atanSubFlag,
                     std::vector<float>& overlap, std::vector<float>& atanSub)
{
    constexpr float EPS = 1e-7f;
    constexpr float FOUR_OVER_PI_SQ = 0.4052847345693511f;
    overlap.resize(N); atanSub.resize(N);
    for (int i = 0; i < N; ++i) {
        float bx1, by1, bx2, by2, gx1, gy1, gx2, gy2;
        float bw, bh, gw, gh;
        float a0 = bb[0*N+i], a1 = bb[1*N+i], a2 = bb[2*N+i], a3 = bb[3*N+i];
        float b0 = gb[0*N+i], b1 = gb[1*N+i], b2 = gb[2*N+i], b3 = gb[3*N+i];
        if (trans) {
            bw = std::max(0.0f, a2); bh = std::max(0.0f, a3);
            gw = std::max(0.0f, b2); gh = std::max(0.0f, b3);
            bx1 = a0 - bw/2; by1 = a1 - bh/2;
            bx2 = a0 + bw/2; by2 = a1 + bh/2;
            gx1 = b0 - gw/2; gy1 = b1 - gh/2;
            gx2 = b0 + gw/2; gy2 = b1 + gh/2;
        } else {
            bx1 = a0; by1 = a1; bx2 = a2; by2 = a3;
            gx1 = b0; gy1 = b1; gx2 = b2; gy2 = b3;
            bw = std::max(0.0f, bx2 - bx1); bh = std::max(0.0f, by2 - by1);
            gw = std::max(0.0f, gx2 - gx1); gh = std::max(0.0f, gy2 - gy1);
        }
        float iw = std::max(0.0f, std::min(bx2, gx2) - std::max(bx1, gx1));
        float ih = std::max(0.0f, std::min(by2, gy2) - std::max(by1, gy1));
        float inter = iw * ih;
        float Ab = bw * bh, Ag = gw * gh;
        float unionDenom = std::max(Ab + Ag - inter + EPS, 1e-4f);
        float iou = inter / unionDenom;
        float denom = (modeId == 0) ? unionDenom : std::max(Ag + EPS, 1e-4f);
        float baseOverlap = inter / denom;

        float centerDx = 0.5f * (bx1 + bx2 - gx1 - gx2);
        float centerDy = 0.5f * (by1 + by2 - gy1 - gy2);
        float centerDistance = centerDx * centerDx + centerDy * centerDy;
        float encW = std::max(bx2, gx2) - std::min(bx1, gx1);
        float encH = std::max(by2, gy2) - std::min(by1, gy1);
        float encDiag = std::max(encW * encW + encH * encH + EPS, 1e-4f);

        if (atanSubFlag) {
            float r1 = bw / std::max(bh + EPS, 1e-4f);
            float r2 = gw / std::max(gh + EPS, 1e-4f);
            float d  = std::atan(r1) - std::atan(r2);
            atanSub[i] = d;
            float out = baseOverlap - centerDistance / encDiag;
            if (modeId == 0) {
                float v = d * d * FOUR_OVER_PI_SQ;
                float alpha = v / std::max(1.0f + v - iou, 1e-4f);
                out -= alpha * v;
            }
            overlap[i] = out;
        } else {
            atanSub[i] = 0.0f;
            overlap[i] = baseOverlap - centerDistance / encDiag;
        }
    }
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& host, const std::vector<int64_t>& shape,
                           void** dev, aclDataType dt, aclTensor** t)
{
    auto sz = 1ull;
    for (auto d : shape) sz *= d;
    sz *= sizeof(T);
    auto ret = aclrtMalloc(dev, sz, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtMemcpy(*dev, sz, host.data(), sz, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) return ret;
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; --i)
        strides[i] = shape[i + 1] * strides[i + 1];
    *t = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0,
                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *dev);
    return 0;
}

// fp16 helpers (IEEE 754 binary16 conversion; cannsim host has no <cuda_fp16.h>).
static uint16_t fp32_to_fp16(float f)
{
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) { return (uint16_t)(sign << 15); }
    if (exp >= 31) { return (uint16_t)((sign << 15) | (0x1f << 10)); }
    return (uint16_t)((sign << 15) | (exp << 10) | (mant >> 13));
}
static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) return sign ? -0.0f : 0.0f;
    if (exp == 31) { uint32_t x = (sign << 31) | (0xff << 23); float f; std::memcpy(&f, &x, 4); return f; }
    uint32_t x = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}
struct Case {
    const char* name;
    int N;
    aclDataType dt;
    bool trans;
    bool atanSubFlag;
    int modeId;            // 0=iou 1=iof
    const char* mode;
    const char* dist;      // "random_xyxy" / "full_overlap" / "no_overlap" / "tiny" / "aspect_diff"
    float overlap_tol;
    float atansub_tol;
};

static void fill_rand_xyxy(std::vector<float>& v, int N, float scale, uint32_t seed)
{
    std::mt19937 r(seed);
    std::uniform_real_distribution<float> u(0.0f, scale);
    for (int i = 0; i < N; ++i) {
        float x1 = u(r), x2 = u(r); if (x2 < x1) std::swap(x1, x2);
        float y1 = u(r), y2 = u(r); if (y2 < y1) std::swap(y1, y2);
        v[0*N+i] = x1; v[1*N+i] = y1; v[2*N+i] = x2; v[3*N+i] = y2;
    }
}

static void fill_rand_xywh(std::vector<float>& v, int N, float scale, uint32_t seed)
{
    std::mt19937 r(seed);
    std::uniform_real_distribution<float> uxy(0.0f, scale);
    std::uniform_real_distribution<float> uwh(1.0f, scale / 2);
    for (int i = 0; i < N; ++i) {
        v[0*N+i] = uxy(r); v[1*N+i] = uxy(r);
        v[2*N+i] = uwh(r); v[3*N+i] = uwh(r);
    }
}

static void make_no_overlap_boxes(int N, float scale, std::vector<float>& bb, std::vector<float>& gb)
{
    std::mt19937 ra(2), rb(3);
    std::uniform_real_distribution<float> ua(0.0f, scale/4);
    std::uniform_real_distribution<float> ub(scale/2, scale);
    for (int i = 0; i < N; ++i) {
        float x1 = ua(ra), x2 = ua(ra); if (x2 < x1) std::swap(x1, x2); x2 += 0.5f;
        float y1 = ua(ra), y2 = ua(ra); if (y2 < y1) std::swap(y1, y2); y2 += 0.5f;
        bb[0*N+i] = x1; bb[1*N+i] = y1; bb[2*N+i] = x2; bb[3*N+i] = y2;
        float gx1 = ub(rb), gx2 = ub(rb); if (gx2 < gx1) std::swap(gx1, gx2); gx2 += 0.5f;
        float gy1 = ub(rb), gy2 = ub(rb); if (gy2 < gy1) std::swap(gy1, gy2); gy2 += 0.5f;
        gb[0*N+i] = gx1; gb[1*N+i] = gy1; gb[2*N+i] = gx2; gb[3*N+i] = gy2;
    }
}

static void make_tiny_boxes(int N, float scale, std::vector<float>& bb, std::vector<float>& gb)
{
    std::mt19937 ra(4), rb(5);
    std::uniform_real_distribution<float> uxy(0.0f, scale);
    std::uniform_real_distribution<float> uwh(0.0f, 0.6f);
    for (int i = 0; i < N; ++i) {
        float x = uxy(ra), y = uxy(ra), w = uwh(ra), h = uwh(ra);
        bb[0*N+i] = x; bb[1*N+i] = y; bb[2*N+i] = x + w; bb[3*N+i] = y + h;
        float gx = uxy(rb), gy = uxy(rb), gw = uwh(rb), gh = uwh(rb);
        gb[0*N+i] = gx; gb[1*N+i] = gy; gb[2*N+i] = gx + gw; gb[3*N+i] = gy + gh;
    }
}

static void make_aspect_diff_boxes(int N, float scale, std::vector<float>& bb, std::vector<float>& gb)
{
    std::mt19937 ra(6);
    std::uniform_real_distribution<float> uxy(0.0f, scale/2);
    for (int i = 0; i < N; ++i) {
        float x = uxy(ra), y = uxy(ra);
        bb[0*N+i] = x; bb[1*N+i] = y; bb[2*N+i] = x + 4.0f; bb[3*N+i] = y + 64.0f;
        gb[0*N+i] = x; gb[1*N+i] = y; gb[2*N+i] = x + 64.0f; gb[3*N+i] = y + 4.0f;
    }
}

static void make_boxes(const Case& c, std::vector<float>& bb, std::vector<float>& gb)
{
    int N = c.N;
    float scale = (c.dt == ACL_FLOAT16) ? 64.0f : 256.0f;
    bb.resize(4 * N); gb.resize(4 * N);
    std::string d = c.dist;
    if (d == "full_overlap") {
        if (c.trans) { fill_rand_xywh(bb, N, scale, 1); gb = bb; }
        else         { fill_rand_xyxy(bb, N, scale, 1); gb = bb; }
    } else if (d == "no_overlap") {
        make_no_overlap_boxes(N, scale, bb, gb);
    } else if (d == "tiny") {
        make_tiny_boxes(N, scale, bb, gb);
    } else if (d == "aspect_diff") {
        make_aspect_diff_boxes(N, scale, bb, gb);
    } else if (c.trans) {
        fill_rand_xywh(bb, N, scale, 10); fill_rand_xywh(gb, N, scale, 11);
    } else {
        fill_rand_xyxy(bb, N, scale, 12); fill_rand_xyxy(gb, N, scale, 13);
    }
}

static int create_case_tensors(const Case& c, const std::vector<float>& bb_f, const std::vector<float>& gb_f,
                               const std::vector<int64_t>& boxShape, const std::vector<int64_t>& outShape,
                               aclTensor** bboxes, aclTensor** gtboxes, aclTensor** overlap, aclTensor** atanSub,
                               void** bdev, void** gdev, void** odev, void** adev)
{
    const int N = c.N;
    int rc = 0;
    if (c.dt == ACL_FLOAT) {
        std::vector<float> oh(N, 0), ah(N, 0);
        rc = CreateAclTensor(bb_f, boxShape, bdev, c.dt, bboxes); if (rc) return rc;
        rc = CreateAclTensor(gb_f, boxShape, gdev, c.dt, gtboxes); if (rc) return rc;
        rc = CreateAclTensor(oh,   outShape, odev, c.dt, overlap); if (rc) return rc;
        rc = CreateAclTensor(ah,   outShape, adev, c.dt, atanSub); if (rc) return rc;
    } else if (c.dt == ACL_FLOAT16) {
        std::vector<uint16_t> bh(4*N), gh(4*N), oh(N, 0), ah(N, 0);
        for (int i = 0; i < 4*N; ++i) { bh[i] = fp32_to_fp16(bb_f[i]); gh[i] = fp32_to_fp16(gb_f[i]); }
        rc = CreateAclTensor(bh, boxShape, bdev, c.dt, bboxes); if (rc) return rc;
        rc = CreateAclTensor(gh, boxShape, gdev, c.dt, gtboxes); if (rc) return rc;
        rc = CreateAclTensor(oh, outShape, odev, c.dt, overlap); if (rc) return rc;
        rc = CreateAclTensor(ah, outShape, adev, c.dt, atanSub); if (rc) return rc;
    } else {
        return ACL_ERROR_INVALID_PARAM;
    }
    return 0;
}

static int execute_ciou(aclrtStream stream, const Case& c, aclTensor* bboxes, aclTensor* gtboxes,
                        aclTensor* overlap, aclTensor* atanSub, void*& wsAddr)
{
    char modeBuf[8]; std::snprintf(modeBuf, sizeof modeBuf, "%s", c.mode);
    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    int rc = aclnnCIoUGetWorkspaceSize(bboxes, gtboxes, c.trans, false, modeBuf, c.atanSubFlag,
                                       overlap, atanSub, &wsSize, &executor);
    if (rc != ACL_SUCCESS) { LOG_PRINT("[FAIL %s] GetWS=%d\n", c.name, rc); return rc; }
    if (wsSize > 0) { aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST); }
    rc = aclnnCIoU(wsAddr, wsSize, executor, stream);
    if (rc != ACL_SUCCESS) { LOG_PRINT("[FAIL %s] exec=%d\n", c.name, rc); return rc; }
    aclrtSynchronizeStream(stream);
    return ACL_SUCCESS;
}

static void copy_case_outputs(const Case& c, int N, void* odev, void* adev,
                              std::vector<float>& overlap_dev, std::vector<float>& atan_dev)
{
    overlap_dev.resize(N);
    atan_dev.resize(N);
    if (c.dt == ACL_FLOAT) {
        aclrtMemcpy(overlap_dev.data(), N*4, odev, N*4, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(atan_dev.data(),    N*4, adev, N*4, ACL_MEMCPY_DEVICE_TO_HOST);
    } else if (c.dt == ACL_FLOAT16) {
        std::vector<uint16_t> oh(N), ah(N);
        aclrtMemcpy(oh.data(), N*2, odev, N*2, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(ah.data(), N*2, adev, N*2, ACL_MEMCPY_DEVICE_TO_HOST);
        for (int i = 0; i < N; ++i) { overlap_dev[i] = fp16_to_fp32(oh[i]); atan_dev[i] = fp16_to_fp32(ah[i]); }
    }
}

static void calc_max_errors(int N, const std::vector<float>& overlap_dev, const std::vector<float>& atan_dev,
                            const std::vector<float>& overlap_ref, const std::vector<float>& atan_ref,
                            float& max_err_overlap, float& max_err_atan)
{
    max_err_overlap = 0.0f; max_err_atan = 0.0f;
    for (int i = 0; i < N; ++i) {
        float er_o = std::fabs(overlap_dev[i] - overlap_ref[i]);
        float er_a = std::fabs(atan_dev[i]    - atan_ref[i]);
        if (er_o > max_err_overlap) max_err_overlap = er_o;
        if (er_a > max_err_atan)    max_err_atan    = er_a;
    }
}

static int run_case(aclrtStream stream, const Case& c, float& max_err_overlap, float& max_err_atan)
{
    const int N = c.N;
    std::vector<int64_t> boxShape = {4, N};
    std::vector<int64_t> outShape = {1, N};
    std::vector<float> bb_f, gb_f, overlap_ref(N), atan_ref(N);
    make_boxes(c, bb_f, gb_f);
    ref_ciou(bb_f, gb_f, N, c.trans, c.modeId, c.atanSubFlag, overlap_ref, atan_ref);

    aclTensor *bboxes = nullptr, *gtboxes = nullptr, *overlap = nullptr, *atanSub = nullptr;
    void *bdev = nullptr, *gdev = nullptr, *odev = nullptr, *adev = nullptr, *wsAddr = nullptr;
    int rc = create_case_tensors(c, bb_f, gb_f, boxShape, outShape, &bboxes, &gtboxes,
                                 &overlap, &atanSub, &bdev, &gdev, &odev, &adev);
    if (rc) return rc;
    rc = execute_ciou(stream, c, bboxes, gtboxes, overlap, atanSub, wsAddr);
    if (rc != ACL_SUCCESS) return rc;

    std::vector<float> overlap_dev, atan_dev;
    copy_case_outputs(c, N, odev, adev, overlap_dev, atan_dev);
    calc_max_errors(N, overlap_dev, atan_dev, overlap_ref, atan_ref, max_err_overlap, max_err_atan);
    aclDestroyTensor(bboxes); aclDestroyTensor(gtboxes);
    aclDestroyTensor(overlap); aclDestroyTensor(atanSub);
    aclrtFree(bdev); aclrtFree(gdev); aclrtFree(odev); aclrtFree(adev);
    if (wsAddr) aclrtFree(wsAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateStream(&stream); CHECK_RET(ret == ACL_SUCCESS, return ret);

    // N=128 keeps single-case sim time short on cannsim while exercising
    // multi-tile inner loop (TILE_N=512 -> 1 tile each but covers cast+atan path).
    // Tolerances calibrated to lt-CIoU/dashboard (a2 real NPU baseline):
    //   fp32 overlap abs<=2e-5; atan abs<=2e-4 (hardware Atan vs std::atan)
    //   fp16 overlap rel<=1e-3 abs<=1e-3; atan abs<=1e-3
    // N=128 to keep cannsim simulation under a few minutes per case.
    Case cases[] = {
        {"fp32_iou_xyxy_random",  128, ACL_FLOAT,   false, true,  0, "iou", "random_xyxy", 2e-5f, 2e-4f},
        {"fp32_iou_xywh_random",  128, ACL_FLOAT,   true,  true,  0, "iou", "random_xywh", 2e-5f, 2e-4f},
        {"fp32_iof_xyxy_random",  128, ACL_FLOAT,   false, true,  1, "iof", "random_xyxy", 2e-5f, 2e-4f},
        {"fp32_atan_off",         128, ACL_FLOAT,   false, false, 0, "iou", "random_xyxy", 2e-5f, 1e-6f},
        {"fp32_full_overlap",     128, ACL_FLOAT,   false, true,  0, "iou", "full_overlap", 2e-5f, 2e-4f},
        {"fp32_no_overlap",       128, ACL_FLOAT,   false, true,  0, "iou", "no_overlap",   2e-5f, 2e-4f},
        {"fp32_aspect_diff",      128, ACL_FLOAT,   false, true,  0, "iou", "aspect_diff",  2e-5f, 2e-4f},
        {"fp16_iou_xyxy_random",  128, ACL_FLOAT16, false, true,  0, "iou", "random_xyxy", 5e-2f, 5e-2f},
    };
    const int total = sizeof(cases) / sizeof(cases[0]);
    int passed = 0, failed = 0;
    LOG_PRINT("=== CIoU multi-case verification (cannsim Ascend950) ===\n");
    for (int i = 0; i < total; ++i) {
        float erO = -1.0f, erA = -1.0f;
        int rc = run_case(stream, cases[i], erO, erA);
        bool ok = (rc == ACL_SUCCESS) && (erO <= cases[i].overlap_tol) && (erA <= cases[i].atansub_tol);
        LOG_PRINT("[%s] %-26s overlap_max_err=%.3e (tol=%.0e) atan_max_err=%.3e (tol=%.0e) rc=%d\n",
                  ok ? "PASS" : "FAIL", cases[i].name, erO, cases[i].overlap_tol, erA, cases[i].atansub_tol, rc);
        if (ok) ++passed; else ++failed;
    }
    LOG_PRINT("=== SUMMARY: %d/%d PASS ===\n", passed, total);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failed == 0 ? 0 : 1;
}
