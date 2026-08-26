/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTATED_BOX_ENCODE_KERNEL_H
#define ROTATED_BOX_ENCODE_KERNEL_H

#include "kernel_operator.h"
#include "adv_api/math/tan.h"
#include "adv_api/math/sin.h"
#include "adv_api/math/cos.h"
#include "rotated_box_encode_tiling_data.h"

// ===========================================================================
// Design constants (DESIGN-BRANCH-0 §0 / §5 / §6)
// ===========================================================================
constexpr int32_t RBE_BOX_CHANNELS = 5;                             // spec.yaml shape[1]==5
constexpr float RBE_PI_OVER_180 = 3.14159265358979323846f / 180.0f; // §6 C9

// Channel indices (x0, y0, x1, y1, θ_deg) — spec.yaml shape_constraints.notes
constexpr int32_t RBE_CH_X0 = 0;
constexpr int32_t RBE_CH_Y0 = 1;
constexpr int32_t RBE_CH_X1 = 2;
constexpr int32_t RBE_CH_Y1 = 3;
constexpr int32_t RBE_CH_ANG = 4;

// ===========================================================================
// VF functions — main compute chain (DESIGN-BRANCH-0 §5.3 / DESIGN-BRANCH-1 §5.3)
//
// All three are fp32-only __simd_vf__ functions invoked via asc_vf_call.
// Each operates on [5, tileCount] channel-contiguous UB views; the per-channel
// stride equals `tileCount` (passed as a uint32_t parameter).
//
// **Shared by both branches** (DESIGN-BRANCH-1 §5.3: "VF 函数体 fp32-only 且
// 两 dtype 共用"). Branch-0 calls them with B1/B2/B3 (Cast-staged fp32);
// Branch-1 calls them with B0/B1/B2 (CopyIn-direct fp32). The VF signature
// is dtype-agnostic — callers pass anchor as `b1` and gt as `b2`.
//
// Register pressure (DESIGN-BRANCH-0 §5.4 / DESIGN-BRANCH-1 §5.4): each VF
// peaks at ≤5 live RegTensor — DxDwVF/DyDhVF reach 5 (ax0/ax2/gx0/gx2 + 1
// temp), DThetaVF reaches 3 (tan_a/tan_g + 1 temp).
// ===========================================================================

// ---------------------------------------------------------------------------
// DxDwVF: C1+C3+C4+C5+C7 → Y0(dx, ch0) + Y2(dw, ch2)
//   inputs : anchor ch0/ch2 (ax0/ax2), gt ch0/ch2 (gx0/gx2)
//   outputs: B3 ch0 (dx), B3 ch2 (dw)
//   scalars: wx (weight[0]), ww (weight[2])
// ---------------------------------------------------------------------------
__simd_vf__ inline void DxDwVF(__ubuf__ float* b3, // output  [5, ubStride], writes ch0 & ch2
                               __ubuf__ float* b1, // anchor  [5, ubStride], reads ch0 & ch2
                               __ubuf__ float* b2, // gt      [5, ubStride], reads ch0 & ch2
                               uint32_t boxCount,  // valid box count per channel (may be < ubStride)
                               uint32_t ubStride,  // channel stride (32B-aligned, >= boxCount)
                               float wx, float ww) // weight[0] (dx), weight[2] (dw)
{
    constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float); // 64 (fp32)
    uint32_t remaining = boxCount;
    uint16_t repeatNum = static_cast<uint16_t>((boxCount + VL - 1) / VL);

    // Per-channel UB base pointers (channel stride = ubStride, 32B-aligned)
    __ubuf__ float* ax0Ptr = b1 + 0 * ubStride;
    __ubuf__ float* ax2Ptr = b1 + 2 * ubStride;
    __ubuf__ float* gx0Ptr = b2 + 0 * ubStride;
    __ubuf__ float* gx2Ptr = b2 + 2 * ubStride;
    __ubuf__ float* dxPtr = b3 + 0 * ubStride;
    __ubuf__ float* dwPtr = b3 + 2 * ubStride;

    for (uint16_t i = 0; i < repeatNum; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);

        // Load 4 channel slices
        AscendC::Reg::RegTensor<float> ax0, ax2, gx0, gx2;
        AscendC::Reg::LoadAlign(ax0, ax0Ptr + off);
        AscendC::Reg::LoadAlign(ax2, ax2Ptr + off);
        AscendC::Reg::LoadAlign(gx0, gx0Ptr + off);
        AscendC::Reg::LoadAlign(gx2, gx2Ptr + off);

        // C4: w_a = Maxs(ax2 - ax0, 1.0)  (propagates NaN, DESIGN §1.4.1)
        AscendC::Reg::RegTensor<float> wa;
        AscendC::Reg::Sub(wa, ax2, ax0, mask);
        AscendC::Reg::Maxs(wa, wa, 1.0f, mask);

        // C4': w_g = Maxs(gx2 - gx0, 1.0)
        AscendC::Reg::RegTensor<float> wg;
        AscendC::Reg::Sub(wg, gx2, gx0, mask);
        AscendC::Reg::Maxs(wg, wg, 1.0f, mask);

        // C1: cx_a = ax0 + w_a * 0.5
        AscendC::Reg::RegTensor<float> halfWa, cxa;
        AscendC::Reg::Muls(halfWa, wa, 0.5f, mask);
        AscendC::Reg::Add(cxa, ax0, halfWa, mask);

        // C3: cx_g = gx0 + w_g * 0.5
        AscendC::Reg::RegTensor<float> halfWg, cxg;
        AscendC::Reg::Muls(halfWg, wg, 0.5f, mask);
        AscendC::Reg::Add(cxg, gx0, halfWg, mask);

        // C5: dx = (cx_g - cx_a) / w_a * wx
        AscendC::Reg::RegTensor<float> dxdc, dxn, dx;
        AscendC::Reg::Sub(dxdc, cxg, cxa, mask);
        AscendC::Reg::Div(dxn, dxdc, wa, mask);
        AscendC::Reg::Muls(dx, dxn, wx, mask);
        AscendC::Reg::StoreAlign(dxPtr + off, dx, mask);

        // C7: dw = (Ln(w_g) - Ln(w_a)) * ww
        // Numerically-stable reformulation of Ln(w_g/w_a): avoids forming the
        // ratio w_g/w_a which can underflow to a denormal (when w_g << w_a)
        // and trigger hardware flush-to-zero (FTZ) → Ln(0) = -Inf, diverging
        // from the golden (which preserves denormals). Ln(w_g)-Ln(w_a) is the
        // mathematical identity ln(a/b)=ln(a)-ln(b); both operands w_g/w_a are
        // ≥1.0 (Maxs lower bound), so Ln inputs are ≥0 (no denormal). Design
        // §5.5 allows kernel-internal numerical-stability reformulations.
        AscendC::Reg::RegTensor<float> lnwg, lnwa, dwdiff, dw;
        AscendC::Reg::Ln(lnwg, wg, mask);
        AscendC::Reg::Ln(lnwa, wa, mask);
        AscendC::Reg::Sub(dwdiff, lnwg, lnwa, mask);
        AscendC::Reg::Muls(dw, dwdiff, ww, mask);
        AscendC::Reg::StoreAlign(dwPtr + off, dw, mask);
    }
}

// ---------------------------------------------------------------------------
// DyDhVF: C2+C3'+C4'+C6+C8 → Y1(dy, ch1) + Y3(dh, ch3)
//   inputs : anchor ch1/ch3 (ay0/ay1), gt ch1/ch3 (gy0/gy1)
//   outputs: B3 ch1 (dy), B3 ch3 (dh)
//   scalars: wy (weight[1]), wh (weight[3])
// ---------------------------------------------------------------------------
__simd_vf__ inline void DyDhVF(__ubuf__ float* b3, __ubuf__ float* b1, __ubuf__ float* b2, uint32_t boxCount,
                               uint32_t ubStride, float wy, float wh)
{
    constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);
    uint32_t remaining = boxCount;
    uint16_t repeatNum = static_cast<uint16_t>((boxCount + VL - 1) / VL);

    __ubuf__ float* ay0Ptr = b1 + 1 * ubStride;
    __ubuf__ float* ay1Ptr = b1 + 3 * ubStride;
    __ubuf__ float* gy0Ptr = b2 + 1 * ubStride;
    __ubuf__ float* gy1Ptr = b2 + 3 * ubStride;
    __ubuf__ float* dyPtr = b3 + 1 * ubStride;
    __ubuf__ float* dhPtr = b3 + 3 * ubStride;

    for (uint16_t i = 0; i < repeatNum; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::RegTensor<float> ay0, ay1, gy0, gy1;
        AscendC::Reg::LoadAlign(ay0, ay0Ptr + off);
        AscendC::Reg::LoadAlign(ay1, ay1Ptr + off);
        AscendC::Reg::LoadAlign(gy0, gy0Ptr + off);
        AscendC::Reg::LoadAlign(gy1, gy1Ptr + off);

        // C4: h_a = Maxs(ay1 - ay0, 1.0)
        AscendC::Reg::RegTensor<float> ha;
        AscendC::Reg::Sub(ha, ay1, ay0, mask);
        AscendC::Reg::Maxs(ha, ha, 1.0f, mask);

        // C4': h_g = Maxs(gy1 - gy0, 1.0)
        AscendC::Reg::RegTensor<float> hg;
        AscendC::Reg::Sub(hg, gy1, gy0, mask);
        AscendC::Reg::Maxs(hg, hg, 1.0f, mask);

        // C2: cy_a = ay0 + h_a * 0.5
        AscendC::Reg::RegTensor<float> halfHa, cya;
        AscendC::Reg::Muls(halfHa, ha, 0.5f, mask);
        AscendC::Reg::Add(cya, ay0, halfHa, mask);

        // C3': cy_g = gy0 + h_g * 0.5
        AscendC::Reg::RegTensor<float> halfHg, cyg;
        AscendC::Reg::Muls(halfHg, hg, 0.5f, mask);
        AscendC::Reg::Add(cyg, gy0, halfHg, mask);

        // C6: dy = (cy_g - cy_a) / h_a * wy
        AscendC::Reg::RegTensor<float> dydc, dyn, dy;
        AscendC::Reg::Sub(dydc, cyg, cya, mask);
        AscendC::Reg::Div(dyn, dydc, ha, mask);
        AscendC::Reg::Muls(dy, dyn, wy, mask);
        AscendC::Reg::StoreAlign(dyPtr + off, dy, mask);

        // C8: dh = (Ln(h_g) - Ln(h_a)) * wh
        // Numerically-stable reformulation of Ln(h_g/h_a): avoids forming the
        // ratio h_g/h_a which can underflow to a denormal and trigger hardware
        // flush-to-zero (FTZ) → Ln(0) = -Inf. See DxDwVF C7 comment for the
        // full rationale (ln(a/b)=ln(a)-ln(b) identity, both ≥1.0).
        AscendC::Reg::RegTensor<float> lnhg, lnha, dhdiff, dh;
        AscendC::Reg::Ln(lnhg, hg, mask);
        AscendC::Reg::Ln(lnha, ha, mask);
        AscendC::Reg::Sub(dhdiff, lnhg, lnha, mask);
        AscendC::Reg::Muls(dh, dhdiff, wh, mask);
        AscendC::Reg::StoreAlign(dhPtr + off, dh, mask);
    }
}

// ---------------------------------------------------------------------------
// DThetaVF: C9 → Y4(dθ, ch4)
//   inputs : B_out.ch4 (dθ_prewa, precomputed in S3/S5 via stable identity)
//   outputs: B_out ch4 (dθ = dθ_prewa × wa)
//   scalars: wa (weight[4])
//
//   The dθ_prewa is precomputed OUTSIDE the VF (ordinary code, S3/S5) using
//   the tan difference identity:
//     dθ_prewa = sin(θ_g_rad − θ_a_rad) / (cos(θ_g_rad) × cos(θ_a_rad))
//   which is mathematically equal to tan(θ_g) − tan(θ_a) but avoids
//   catastrophic cancellation when θ_a ≈ θ_g near the 90° singularity
//   (where |tan| → ∞ and subtracting two large nearly-equal values loses
//   precision). DESIGN §5.5 / §10.9.3 flag the tan singularity as a known
//   precision risk; the identity reformulation is the kernel-developer's
//   implementation decision to fill the DESIGN §5.4 precision gap. The VF
//   body is reduced to a single Reg::Muls (chain length 1); b1/b2 params
//   are retained in the signature for call-site uniformity with DxDwVF/
//   DyDhVF but are NOT read (DxDwVF/DyDhVF only consume ch0–ch3 of b1/b2,
//   never ch4, so the S3/S5 precompute overwriting ch4 with cos values is
//   safe). Both branches share this VF body (DESIGN-BRANCH-1 §5.3).
// ---------------------------------------------------------------------------
__simd_vf__ inline void DThetaVF(__ubuf__ float* b3, // output [5, ubStride], reads & writes ch4
                                 __ubuf__ float* b1, // anchor (UNUSED — signature compat with DxDwVF/DyDhVF)
                                 __ubuf__ float* b2, // gt     (UNUSED — signature compat)
                                 uint32_t boxCount,  // valid box count per channel (may be < ubStride)
                                 uint32_t ubStride,  // channel stride (32B-aligned, >= boxCount)
                                 float wa)           // weight[4] (dθ scalar multiplier)
{
    constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);
    uint32_t remaining = boxCount;
    uint16_t repeatNum = static_cast<uint16_t>((boxCount + VL - 1) / VL);

    // dθ_prewa lives at B_out.ch4 (offset 4*ubStride); read it, multiply by
    // wa, and write dθ back to the same location (in-place).
    __ubuf__ float* dthetaPrewaPtr = b3 + 4 * ubStride;

    for (uint16_t i = 0; i < repeatNum; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);

        // C9: dθ = dθ_prewa × wa
        AscendC::Reg::RegTensor<float> prewa;
        AscendC::Reg::LoadAlign(prewa, dthetaPrewaPtr + off);
        AscendC::Reg::RegTensor<float> dtheta;
        AscendC::Reg::Muls(dtheta, prewa, wa, mask);
        AscendC::Reg::StoreAlign(dthetaPrewaPtr + off, dtheta, mask);
    }
}

// ===========================================================================
// KernelRotatedBoxEncode — both-branch implementation (arch35 / ascend950)
//
// Template parameter DTYPE selects the branch (DESIGN §6):
//   DTYPE == ROTATED_BOX_ENCODE_DTYPE_FP16 (0) → branch-0 fp16-upcast (S1–S8)
//   DTYPE == ROTATED_BOX_ENCODE_DTYPE_FP32 (1) → branch-1 fp32-direct (S1–S5)
// Both branches share the same Process() outer-loop skeleton (GetBlockIdx →
// first/tail block double-loop → ProcessTile) and the same three __simd_vf__
// VF bodies. Per-tile pipeline differs: fp16 path has Cast steps (S2/S4/S7),
// fp32 path has none (direct fp32 IO + fp32 VF). Selection is via
// `if constexpr (DTYPE == ...)` inside Init / Process / ProcessTile.
// ===========================================================================
template <int DTYPE>
class KernelRotatedBoxEncode {
public:
    __aicore__ inline KernelRotatedBoxEncode() {}

    __aicore__ inline void Init(GM_ADDR anchor_box, GM_ADDR gt_box, GM_ADDR y, const RotatedBoxEncodeTilingData& td)
    {
        // Empty-tensor short-circuit (DESIGN-BRANCH-0 §0 / §2.6,
        // DESIGN-BRANCH-1 §0 / §2.6): kernel entry already guards dim0==0,
        // but keep a defensive check so Init is safe to call with an empty
        // TilingData (coreNum=0, blockFormer=0).
        if (td.dim0 == 0) {
            empty_ = true;
            return;
        }
        empty_ = false;
        td_ = &td;
        N_ = td.N;
        // B = dim0 / N (host guarantees dim0 == B*N exactly; DESIGN §2.1)
        B_ = td.dim0 / N_;

        // GM buffers — set up once; per-tile offsets computed in ProcessTile.
        // Layout: [B, 5, N] row-major → element(b,c,n) at b*5N + c*N + n.
        if constexpr (DTYPE == ROTATED_BOX_ENCODE_DTYPE_FP16) {
            anchorGm_.SetGlobalBuffer((__gm__ half*)anchor_box);
            gtGm_.SetGlobalBuffer((__gm__ half*)gt_box);
            outGm_.SetGlobalBuffer((__gm__ half*)y);

            // UB tile size (box count) — read from tiling data (host-computed
            // ubFormer, DESIGN-BRANCH-0 §2.3). All 4 TBuf are sized to the full
            // tile: B0 fp16 = tile*5*2 B, B1/B2/B3 fp32 = tile*5*4 B each.
            // P=4, perBoxBytes=70 B/box (DESIGN-BRANCH-0 §4).
            uint32_t tileBytesB0 = static_cast<uint32_t>(td.ubFormer * RBE_BOX_CHANNELS * sizeof(half));
            uint32_t tileBytesB1 = static_cast<uint32_t>(td.ubFormer * RBE_BOX_CHANNELS * sizeof(float));
            pipe_.InitBuffer(b0Buf_, tileBytesB0); // B0 fp16 IO staging
            pipe_.InitBuffer(b1Buf_, tileBytesB1); // B1 fp32 anchor (Cast target)
            pipe_.InitBuffer(b2Buf_, tileBytesB1); // B2 fp32 gt     (Cast target)
            pipe_.InitBuffer(b3Buf_, tileBytesB1); // B3 fp32 output (VF result)
        } else {
            // Branch-1 (fp32-direct, DESIGN-BRANCH-1 §4): P=3 fp32 TBuf.
            // B0 anchor fp32 / B1 gt fp32 / B2 output fp32 — reuses the same
            // b0Buf_/b1Buf_/b2Buf_ handles (TBuf is dtype-agnostic; dtype is
            // determined by Get<float> at use site). b3Buf_ is left
            // uninitialised — fp32 path has no Cast staging buffer.
            anchorGmF32_.SetGlobalBuffer((__gm__ float*)anchor_box);
            gtGmF32_.SetGlobalBuffer((__gm__ float*)gt_box);
            outGmF32_.SetGlobalBuffer((__gm__ float*)y);

            uint32_t tileBytesFp32 = static_cast<uint32_t>(td.ubFormer * RBE_BOX_CHANNELS * sizeof(float));
            pipe_.InitBuffer(b0Buf_, tileBytesFp32); // B0 fp32 anchor (CopyIn direct)
            pipe_.InitBuffer(b1Buf_, tileBytesFp32); // B1 fp32 gt     (CopyIn direct)
            pipe_.InitBuffer(b2Buf_, tileBytesFp32); // B2 fp32 output (VF result)
        }
    }

    __aicore__ inline void Process()
    {
        if (empty_) {
            return; // empty-tensor short-circuit (DESIGN-BRANCH-0/1 §2.6)
        }

        int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        // Over-allocated cores (DESIGN-BRANCH-0/1 §2.6): skip if beyond blockNum.
        if (blockIdx >= td_->blockNum) {
            return;
        }

        bool isLastBlock = (blockIdx == td_->blockNum - 1);
        int64_t loopNum = isLastBlock ? td_->ubLoopOfTailBlock : td_->ubLoopOfFormerBlock;
        int64_t tailNum = isLastBlock ? td_->ubTailOfTailBlock : td_->ubTailOfFormerBlock;

        int64_t boxOffset = blockIdx * td_->blockFormer;

        // Main loop: (loopNum - 1) full ubFormer tiles + 1 tail tile.
        // DESIGN-BRANCH-0/1 §2.5: loopNum = CeilDiv(blockFormer|blockTail, ubFormer),
        // tailNum = remainder. loopNum >= 1 always (blockFormer > 0 when non-empty).
        for (int64_t i = 0; i < loopNum - 1; ++i) {
            ProcessTile(boxOffset, td_->ubFormer);
            boxOffset += td_->ubFormer;
        }
        ProcessTile(boxOffset, tailNum);
    }

private:
    // -----------------------------------------------------------------------
    // ProcessTile — dispatch to the per-branch tile pipeline.
    //   fp16 path (DTYPE=FP16): S1–S8 (DESIGN-BRANCH-0 §3 / §5.1) — CopyIn →
    //     Cast fp16→fp32 → tan precompute → VF C1–C9 → Cast fp32→fp16 →
    //     Interleave+CopyOut.
    //   fp32 path (DTYPE=FP32): S1–S5 (DESIGN-BRANCH-1 §3 / §5.1) — CopyIn →
    //     tan precompute → VF C1–C9 → Interleave+CopyOut (no Cast).
    //   boxOffset : flat box index of the first box in this tile (0..dim0-1)
    //   boxCount  : number of boxes in this tile (ubFormer or tailNum)
    // -----------------------------------------------------------------------
    __aicore__ inline void ProcessTile(int64_t boxOffset, int64_t boxCount)
    {
        if constexpr (DTYPE == ROTATED_BOX_ENCODE_DTYPE_FP16) {
            ProcessTileFp16(boxOffset, boxCount);
        } else {
            ProcessTileFp32(boxOffset, boxCount);
        }
    }

    // -----------------------------------------------------------------------
    // ProcessTileFp16 — Branch-0 fp16-upcast tile pipeline (S1–S8).
    // DESIGN-BRANCH-0 §3 / §5.1.
    // -----------------------------------------------------------------------
    __aicore__ inline void ProcessTileFp16(int64_t boxOffset, int64_t boxCount)
    {
        // Aligned channel stride: ceilAlign(boxCount, 16) ensures every channel
        // start address is 32B-aligned for both fp16 (16*2=32B) and fp32
        // (16*4=64B). Required by LoadAlign/StoreAlign in Muls/Tan/VF.
        // For the max tile (ubFormer=3584, already 16-aligned), ubStride==boxCount
        // so no extra UB cost. Only small/non-aligned tiles use slightly more UB.
        int64_t ubStride = (boxCount + 15) & ~static_cast<int64_t>(15); // ceilAlign(boxCount, 16)

        // Zero-initialize B1/B2/B3 fp32 buffers to prevent Tan/VF from reading
        // uninitialized UB data beyond the valid boxCount elements (which can
        // cause 507035 hardware faults from Div-by-zero or overflow in the Tan
        // polynomial). The valid boxCount*5 elements are overwritten by Cast/VF;
        // the padding (boxCount*5 .. ubFormer*5) stays zero.
        // B0 (fp16) does not need zero-init — it is fully written by CopyIn
        // (fast path) or SetValue (fallback) before being read by Cast.
        {
            auto b1Local = b1Buf_.Get<float>();
            auto b2Local = b2Buf_.Get<float>();
            auto b3Local = b3Buf_.Get<float>();
            AscendC::Duplicate<float>(b1Local, (float)0, static_cast<int32_t>(td_->ubFormer * RBE_BOX_CHANNELS));
            AscendC::Duplicate<float>(b2Local, (float)0, static_cast<int32_t>(td_->ubFormer * RBE_BOX_CHANNELS));
            AscendC::Duplicate<float>(b3Local, (float)0, static_cast<int32_t>(td_->ubFormer * RBE_BOX_CHANNELS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S1 CopyIn anchor (MTE2) → B0 fp16 [5, ubStride] =====
        CopyInPerChannel(anchorGm_, boxOffset, boxCount, ubStride);
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S2 Cast fp16→fp32 anchor (V, ordinary) → B1 [5, ubStride] =====
        // Cast ubStride*5 elements (includes padding zeros between channels).
        {
            auto b0Local = b0Buf_.Get<half>();
            auto b1Local = b1Buf_.Get<float>();
            AscendC::Cast<float, half>(b1Local, b0Local, AscendC::RoundMode::CAST_NONE,
                                       static_cast<uint32_t>(ubStride * RBE_BOX_CHANNELS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S3 CopyIn gt (MTE2) → B0 fp16 (reuse) =====
        CopyInPerChannel(gtGm_, boxOffset, boxCount, ubStride);
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S4 Cast fp16→fp32 gt (V, ordinary) → B2 [5, ubStride] =====
        {
            auto b0Local = b0Buf_.Get<half>();
            auto b2Local = b2Buf_.Get<float>();
            AscendC::Cast<float, half>(b2Local, b0Local, AscendC::RoundMode::CAST_NONE,
                                       static_cast<uint32_t>(ubStride * RBE_BOX_CHANNELS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S5 dθ_prewa precompute (V, ordinary, stable identity on ch4) =====
        // Computes dθ_prewa = sin(θ_g_rad − θ_a_rad) / (cos(θ_g_rad) × cos(θ_a_rad))
        // using the tan difference identity (mathematically equal to
        // tan(θ_g) − tan(θ_a) but avoids catastrophic cancellation when both
        // angles are near the 90° singularity). DESIGN §5.5 / §10.9.3 flag
        // tan near singularity as a precision risk; the identity reformulation
        // fills the DESIGN §5.4 precision gap (kernel-developer impl decision).
        //
        // Buffer choreography (fp16 path: B1=anchor fp32, B2=gt fp32, B3=output):
        //   B1.ch4: θ_a_deg → θ_a_rad → cos(θ_a_rad)
        //   B2.ch4: θ_g_deg → θ_g_rad → cos(θ_g_rad) → cos_g × cos_a (cos_prod)
        //   B3.ch4: Δθ_rad → sin(Δθ) → dθ_prewa = sin(Δθ) / cos_prod
        // DxDwVF/DyDhVF only read ch0–ch3 of B1/B2 (never ch4), so overwriting
        // ch4 with cos values here is safe.
        {
            auto b1Local = b1Buf_.Get<float>(); // anchor
            auto b2Local = b2Buf_.Get<float>(); // gt
            auto b3Local = b3Buf_.Get<float>(); // output
            uint32_t cnt = static_cast<uint32_t>(boxCount);
            uint32_t angOff = static_cast<uint32_t>(RBE_CH_ANG * ubStride);

            // Step 1-2: θ_a_rad = θ_a_deg × π/180, θ_g_rad = θ_g_deg × π/180
            AscendC::Muls<float, false>(b1Local[angOff], b1Local[angOff], RBE_PI_OVER_180, static_cast<int32_t>(cnt));
            AscendC::Muls<float, false>(b2Local[angOff], b2Local[angOff], RBE_PI_OVER_180, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 3: Δθ = θ_g_rad − θ_a_rad → B3.ch4
            AscendC::Sub<float>(b3Local[angOff], b2Local[angOff], b1Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 4: sin(Δθ) → B3.ch4 (in-place)
            AscendC::Sin<float>(b3Local[angOff], b3Local[angOff], cnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 5-6: cos(θ_a_rad) → B1.ch4, cos(θ_g_rad) → B2.ch4 (in-place)
            AscendC::Cos<float>(b1Local[angOff], b1Local[angOff], cnt);
            AscendC::Cos<float>(b2Local[angOff], b2Local[angOff], cnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 7: cos_prod = cos(θ_g) × cos(θ_a) → B2.ch4 (in-place: B2 = B2 × B1)
            AscendC::Mul<float>(b2Local[angOff], b2Local[angOff], b1Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 8: dθ_prewa = sin(Δθ) / cos_prod → B3.ch4 (in-place: B3 = B3 / B2)
            AscendC::Div<float>(b3Local[angOff], b3Local[angOff], b2Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 9: Inf→NaN guard for out-of-hardware-range angles.
            // AscendC Sin/Cos valid range is [-65504, 65504] rad (≈ ±3.75e6°);
            // for |θ_rad| beyond this the hardware Cos returns 0.0 (argument
            // reduction fails), making dθ_prewa = sin/0 → ±Inf. The golden
            // (range-guarded, produces NaN for |θ_rad|>65504) must align, so
            // convert spurious ±Inf → NaN. IEEE-754 identity: inf*0 = NaN,
            // hence dθ_prewa + (0 * dθ_prewa) yields NaN for ±Inf/NaN inputs
            // and preserves finite values (finite + 0 = finite). Reuses B1.ch4
            // (cos_a, no longer needed) as scratch — no extra UB buffer.
            // Design §5.5: "由上游保证 θ 合理范围"; out-of-range → undefined → NaN.
            AscendC::Muls<float, false>(b1Local[angOff], b3Local[angOff], (float)0, static_cast<int32_t>(cnt));
            AscendC::Add<float>(b3Local[angOff], b3Local[angOff], b1Local[angOff], static_cast<int32_t>(cnt));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S6 VF Compute C1–C9 (V, asc_vf_call) → B3 [5, ubStride] =====
        // VF uses ubStride as channel stride; processes boxCount valid elements
        // per channel (mask handles the tail).
        {
            auto b1Local = b1Buf_.Get<float>();
            auto b2Local = b2Buf_.Get<float>();
            auto b3Local = b3Buf_.Get<float>();
            __ubuf__ float* b1Ptr = (__ubuf__ float*)b1Local.GetPhyAddr();
            __ubuf__ float* b2Ptr = (__ubuf__ float*)b2Local.GetPhyAddr();
            __ubuf__ float* b3Ptr = (__ubuf__ float*)b3Local.GetPhyAddr();
            uint32_t cnt = static_cast<uint32_t>(boxCount);
            uint32_t stride = static_cast<uint32_t>(ubStride);
            asc_vf_call<DxDwVF>(b3Ptr, b1Ptr, b2Ptr, cnt, stride, td_->weight[0], td_->weight[2]);
            asc_vf_call<DyDhVF>(b3Ptr, b1Ptr, b2Ptr, cnt, stride, td_->weight[1], td_->weight[3]);
            asc_vf_call<DThetaVF>(b3Ptr, b1Ptr, b2Ptr, cnt, stride, td_->weight[4]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S7 Cast fp32→fp16 output (V, ordinary) → B0 fp16 (reuse) =====
        {
            auto b0Local = b0Buf_.Get<half>();
            auto b3Local = b3Buf_.Get<float>();
            AscendC::Cast<half, float>(b0Local, b3Local, AscendC::RoundMode::CAST_NONE,
                                       static_cast<uint32_t>(ubStride * RBE_BOX_CHANNELS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S8 CopyOut (V→MTE3) → GM y =====
        CopyOutPerChannel(boxOffset, boxCount, ubStride);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // -----------------------------------------------------------------------
    // ProcessTileFp32 — Branch-1 fp32-direct tile pipeline (S1–S5).
    // DESIGN-BRANCH-1 §3 / §5.1.
    //   P=3 TBuf<VECCALC>: B0 anchor fp32 / B1 gt fp32 / B2 output fp32.
    //   No Cast steps (fp32 native IO + fp32 VF, DESIGN-BRANCH-1 §5).
    //   VF bodies DxDwVF/DyDhVF/DThetaVF shared with fp16 path (DESIGN-BRANCH-1
    //   §5.3) — called with (out=B2, anchor=B0, gt=B1).
    // -----------------------------------------------------------------------
    __aicore__ inline void ProcessTileFp32(int64_t boxOffset, int64_t boxCount)
    {
        // Aligned channel stride: ceilAlign(boxCount, 16) ensures every channel
        // start address is 64B-aligned for fp32 (16*4=64B), satisfying the
        // 32B LoadAlign/StoreAlign requirement with headroom. For the max tile
        // (ubFormer=4224, 16-aligned), ubStride==boxCount so no extra UB cost.
        int64_t ubStride = (boxCount + 15) & ~static_cast<int64_t>(15); // ceilAlign(boxCount, 16)

        // Zero-initialize B0/B1/B2 fp32 buffers to prevent Tan/VF from reading
        // uninitialized UB data beyond the valid boxCount elements (which can
        // cause 507035 hardware faults from Div-by-zero or overflow in the Tan
        // polynomial). The valid boxCount*5 elements are overwritten by
        // CopyIn/VF; the padding (boxCount*5 .. ubFormer*5) stays zero.
        {
            auto b0Local = b0Buf_.Get<float>();
            auto b1Local = b1Buf_.Get<float>();
            auto b2Local = b2Buf_.Get<float>();
            AscendC::Duplicate<float>(b0Local, (float)0, static_cast<int32_t>(td_->ubFormer * RBE_BOX_CHANNELS));
            AscendC::Duplicate<float>(b1Local, (float)0, static_cast<int32_t>(td_->ubFormer * RBE_BOX_CHANNELS));
            AscendC::Duplicate<float>(b2Local, (float)0, static_cast<int32_t>(td_->ubFormer * RBE_BOX_CHANNELS));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S1 CopyIn anchor (MTE2) → B0 fp32 [5, ubStride] =====
        CopyInPerChannelFp32(anchorGmF32_, b0Buf_, boxOffset, boxCount, ubStride);
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S2 CopyIn gt (MTE2) → B1 fp32 [5, ubStride] =====
        // (No Cast — fp32 native IO, DESIGN-BRANCH-1 §3)
        CopyInPerChannelFp32(gtGmF32_, b1Buf_, boxOffset, boxCount, ubStride);
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S3 dθ_prewa precompute (V, ordinary, stable identity on ch4) =====
        // Computes dθ_prewa = sin(θ_g_rad − θ_a_rad) / (cos(θ_g_rad) × cos(θ_a_rad))
        // using the tan difference identity (mathematically equal to
        // tan(θ_g) − tan(θ_a) but avoids catastrophic cancellation when both
        // angles are near the 90° singularity). DESIGN-BRANCH-1 §5.5 / §10.9.3
        // flag tan near singularity as a precision risk; the identity
        // reformulation fills the DESIGN §5.4 precision gap. Shared VF body
        // (DThetaVF) reads the precomputed dθ_prewa from B_out.ch4 and just
        // multiplies by wa — the VF body is shared with Branch-0 (DESIGN-
        // BRANCH-1 §5.3), and the S3 precompute is also the same algorithm.
        //
        // Buffer choreography (fp32 path: B0=anchor, B1=gt, B2=output):
        //   B0.ch4: θ_a_deg → θ_a_rad → cos(θ_a_rad)
        //   B1.ch4: θ_g_deg → θ_g_rad → cos(θ_g_rad) → cos_g × cos_a (cos_prod)
        //   B2.ch4: Δθ_rad → sin(Δθ) → dθ_prewa = sin(Δθ) / cos_prod
        // DxDwVF/DyDhVF only read ch0–ch3 of B0/B1 (never ch4), so overwriting
        // ch4 with cos values here is safe.
        {
            auto b0Local = b0Buf_.Get<float>(); // anchor
            auto b1Local = b1Buf_.Get<float>(); // gt
            auto b2Local = b2Buf_.Get<float>(); // output
            uint32_t cnt = static_cast<uint32_t>(boxCount);
            uint32_t angOff = static_cast<uint32_t>(RBE_CH_ANG * ubStride);

            // Step 1-2: θ_a_rad = θ_a_deg × π/180, θ_g_rad = θ_g_deg × π/180
            AscendC::Muls<float, false>(b0Local[angOff], b0Local[angOff], RBE_PI_OVER_180, static_cast<int32_t>(cnt));
            AscendC::Muls<float, false>(b1Local[angOff], b1Local[angOff], RBE_PI_OVER_180, static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 3: Δθ = θ_g_rad − θ_a_rad → B2.ch4
            AscendC::Sub<float>(b2Local[angOff], b1Local[angOff], b0Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 4: sin(Δθ) → B2.ch4 (in-place)
            AscendC::Sin<float>(b2Local[angOff], b2Local[angOff], cnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 5-6: cos(θ_a_rad) → B0.ch4, cos(θ_g_rad) → B1.ch4 (in-place)
            AscendC::Cos<float>(b0Local[angOff], b0Local[angOff], cnt);
            AscendC::Cos<float>(b1Local[angOff], b1Local[angOff], cnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 7: cos_prod = cos(θ_g) × cos(θ_a) → B1.ch4 (in-place: B1 = B1 × B0)
            AscendC::Mul<float>(b1Local[angOff], b1Local[angOff], b0Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 8: dθ_prewa = sin(Δθ) / cos_prod → B2.ch4 (in-place: B2 = B2 / B1)
            AscendC::Div<float>(b2Local[angOff], b2Local[angOff], b1Local[angOff], static_cast<int32_t>(cnt));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Step 9: Inf→NaN guard for out-of-hardware-range angles (same as
            // fp16 path S5 step 9). AscendC Sin/Cos valid range [-65504, 65504]
            // rad; beyond it Cos→0 → dθ_prewa=±Inf. Convert ±Inf→NaN via the
            // IEEE-754 identity inf*0=NaN (dθ_prewa + 0*dθ_prewa). Reuses B0.ch4
            // (cos_a, no longer needed) as scratch. Aligns with the golden's
            // range guard (|θ_rad|>65504 → NaN). Design §5.5 out-of-range → NaN.
            AscendC::Muls<float, false>(b0Local[angOff], b2Local[angOff], (float)0, static_cast<int32_t>(cnt));
            AscendC::Add<float>(b2Local[angOff], b2Local[angOff], b0Local[angOff], static_cast<int32_t>(cnt));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S4 VF Compute C1–C9 (V, asc_vf_call) → B2 fp32 [5, ubStride] =====
        // VF uses ubStride as channel stride; processes boxCount valid elements
        // per channel (mask handles the tail). VF signature: (out, anchor, gt,
        // ...). For Branch-1: out=B2, anchor=B0, gt=B1 (DESIGN-BRANCH-1 §5.1).
        {
            auto b0Local = b0Buf_.Get<float>();
            auto b1Local = b1Buf_.Get<float>();
            auto b2Local = b2Buf_.Get<float>();
            __ubuf__ float* b0Ptr = (__ubuf__ float*)b0Local.GetPhyAddr();
            __ubuf__ float* b1Ptr = (__ubuf__ float*)b1Local.GetPhyAddr();
            __ubuf__ float* b2Ptr = (__ubuf__ float*)b2Local.GetPhyAddr();
            uint32_t cnt = static_cast<uint32_t>(boxCount);
            uint32_t stride = static_cast<uint32_t>(ubStride);
            asc_vf_call<DxDwVF>(b2Ptr, b0Ptr, b1Ptr, cnt, stride, td_->weight[0], td_->weight[2]);
            asc_vf_call<DyDhVF>(b2Ptr, b0Ptr, b1Ptr, cnt, stride, td_->weight[1], td_->weight[3]);
            asc_vf_call<DThetaVF>(b2Ptr, b0Ptr, b1Ptr, cnt, stride, td_->weight[4]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ===== S5 CopyOut (V→MTE3) → GM y [B,5,N] =====
        // (Interleave+CopyOut fused via per-channel segment scatter, same as
        // fp16 path — GM layout [B,5,N] is written directly from UB [5,ubStride],
        // DESIGN-BRANCH-1 §5.1 / §3 flowchart.)
        CopyOutPerChannelFp32(boxOffset, boxCount, ubStride);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // -----------------------------------------------------------------------
    // CopyInPerChannel — MTE2 per-batch-segment per-channel copy (fp16 path).
    //
    // GM layout [B, 5, N]: within batch b, channel c's N values are contiguous
    // at [b*5N + c*N, b*5N + (c+1)*N). A tile may span batch boundaries; we
    // split into per-batch segments and issue one copy per (channel, segment).
    //
    // Alignment: on arch35, DataCopyPad (GM→UB) requires the GM source address
    // to be 32B-aligned. When N is a multiple of 16 (fp16 32B/sizeof(half)=16),
    // all channel addresses b*5N+c*N are 16-aligned → 32B byte-aligned → use
    // fast DataCopyPad path. When N is not 16-aligned, fall back to scalar
    // GetValue/SetValue (slow but correct; only hit by small-N test cases).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInPerChannel(AscendC::GlobalTensor<half>& gm, int64_t boxOffset, int64_t boxCount,
                                            int64_t ubStride)
    {
        auto b0Local = b0Buf_.Get<half>();
        // Fast path: N is 16-aligned → all channel GM addresses are 32B-aligned.
        if ((N_ % 16) == 0) {
            AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, (half)0};
            int64_t segStart = boxOffset;
            int64_t segEnd = boxOffset + boxCount;
            while (segStart < segEnd) {
                int64_t b = segStart / N_;
                int64_t nStart = segStart % N_;
                int64_t batchEnd = (b + 1) * N_;
                int64_t segStop = (segEnd < batchEnd) ? segEnd : batchEnd;
                int64_t segLen = segStop - segStart;

                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(segLen * sizeof(half));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;

                int64_t ubOff = segStart - boxOffset;
                for (int32_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
                    int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + c * N_ + nStart;
                    AscendC::DataCopyPad<half, AscendC::PaddingMode::Normal>(
                        b0Local[static_cast<uint32_t>(c * ubStride + ubOff)], gm[static_cast<uint64_t>(gmOff)],
                        copyParams, padParams);
                }
                segStart = segStop;
            }
        } else {
            // Fallback: scalar GetValue/SetValue for non-32B-aligned GM addresses.
            for (int64_t i = 0; i < boxCount; ++i) {
                int64_t boxIdx = boxOffset + i;
                int64_t b = boxIdx / N_;
                int64_t n = boxIdx % N_;
                for (int32_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
                    int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + c * N_ + n;
                    half val = gm.GetValue(static_cast<uint64_t>(gmOff));
                    b0Local.SetValue(static_cast<uint32_t>(c * ubStride + i), val);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // CopyOutPerChannel — MTE3 per-batch-segment per-channel scatter (fp16 path),
    // mirror of CopyInPerChannel. Uses the same 16-aligned fast path / scalar fallback.
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyOutPerChannel(int64_t boxOffset, int64_t boxCount, int64_t ubStride)
    {
        auto b0Local = b0Buf_.Get<half>();
        if ((N_ % 16) == 0) {
            int64_t segStart = boxOffset;
            int64_t segEnd = boxOffset + boxCount;
            while (segStart < segEnd) {
                int64_t b = segStart / N_;
                int64_t nStart = segStart % N_;
                int64_t batchEnd = (b + 1) * N_;
                int64_t segStop = (segEnd < batchEnd) ? segEnd : batchEnd;
                int64_t segLen = segStop - segStart;

                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(segLen * sizeof(half));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;

                int64_t ubOff = segStart - boxOffset;
                for (int32_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
                    int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + c * N_ + nStart;
                    AscendC::DataCopyPad<half, AscendC::PaddingMode::Normal>(
                        outGm_[static_cast<uint64_t>(gmOff)], b0Local[static_cast<uint32_t>(c * ubStride + ubOff)],
                        copyParams);
                }
                segStart = segStop;
            }
        } else {
            // Fallback: scalar GetValue/SetValue for non-32B-aligned GM addresses.
            for (int64_t i = 0; i < boxCount; ++i) {
                int64_t boxIdx = boxOffset + i;
                int64_t b = boxIdx / N_;
                int64_t n = boxIdx % N_;
                for (int32_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
                    int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + c * N_ + n;
                    half val = b0Local.GetValue(static_cast<uint32_t>(c * ubStride + i));
                    outGm_.SetValue(static_cast<uint64_t>(gmOff), val);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // CopyInPerChannelFp32 — MTE2 per-batch-segment per-channel copy (fp32 path).
    // DESIGN-BRANCH-1 §5.1 S1/S2. Mirror of fp16 CopyInPerChannel but operates
    // on GlobalTensor<float> / Get<float>(). The destination buffer is passed
    // in (fp32 path uses B0 for anchor, B1 for gt — distinct, no reuse unlike
    // fp16's B0 double-use). Fast-path alignment: fp32 needs 32B-aligned GM
    // source → N % 8 == 0 (8 × 4B = 32B). When N is not 8-aligned, fall back
    // to scalar GetValue/SetValue.
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInPerChannelFp32(AscendC::GlobalTensor<float>& gm,
                                                AscendC::TBuf<AscendC::TPosition::VECCALC>& dstBuf, int64_t boxOffset,
                                                int64_t boxCount, int64_t ubStride)
    {
        auto bLocal = dstBuf.Get<float>();
        // Fast path: N % 8 == 0 → all channel GM addresses are 32B-aligned.
        if ((N_ % 8) == 0) {
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            int64_t segStart = boxOffset;
            int64_t segEnd = boxOffset + boxCount;
            while (segStart < segEnd) {
                int64_t b = segStart / N_;
                int64_t nStart = segStart % N_;
                int64_t batchEnd = (b + 1) * N_;
                int64_t segStop = (segEnd < batchEnd) ? segEnd : batchEnd;
                int64_t segLen = segStop - segStart;

                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(segLen * sizeof(float));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;

                int64_t ubOff = segStart - boxOffset;
                for (int32_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
                    int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + c * N_ + nStart;
                    AscendC::DataCopyPad<float, AscendC::PaddingMode::Normal>(
                        bLocal[static_cast<uint32_t>(c * ubStride + ubOff)], gm[static_cast<uint64_t>(gmOff)],
                        copyParams, padParams);
                }
                segStart = segStop;
            }
        } else {
            // Fallback: scalar GetValue/SetValue for non-32B-aligned GM addresses.
            for (int64_t i = 0; i < boxCount; ++i) {
                int64_t boxIdx = boxOffset + i;
                int64_t b = boxIdx / N_;
                int64_t n = boxIdx % N_;
                for (int32_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
                    int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + c * N_ + n;
                    float val = gm.GetValue(static_cast<uint64_t>(gmOff));
                    bLocal.SetValue(static_cast<uint32_t>(c * ubStride + i), val);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // CopyOutPerChannelFp32 — MTE3 per-batch-segment per-channel scatter
    // (fp32 path), mirror of CopyInPerChannelFp32. Reads from B2 (output
    // buffer) and scatters to GM y [B,5,N]. Same N%8 fast path / scalar fallback.
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyOutPerChannelFp32(int64_t boxOffset, int64_t boxCount, int64_t ubStride)
    {
        // B2 holds VF output — read from b2Buf_ (not b0Buf_ as in fp16 path
        // where B0 is reused for output staging after Cast fp32→fp16).
        auto b2Local = b2Buf_.Get<float>();
        if ((N_ % 8) == 0) {
            int64_t segStart = boxOffset;
            int64_t segEnd = boxOffset + boxCount;
            while (segStart < segEnd) {
                int64_t b = segStart / N_;
                int64_t nStart = segStart % N_;
                int64_t batchEnd = (b + 1) * N_;
                int64_t segStop = (segEnd < batchEnd) ? segEnd : batchEnd;
                int64_t segLen = segStop - segStart;

                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(segLen * sizeof(float));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;

                int64_t ubOff = segStart - boxOffset;
                for (int32_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
                    int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + c * N_ + nStart;
                    AscendC::DataCopyPad<float, AscendC::PaddingMode::Normal>(
                        outGmF32_[static_cast<uint64_t>(gmOff)], b2Local[static_cast<uint32_t>(c * ubStride + ubOff)],
                        copyParams);
                }
                segStart = segStop;
            }
        } else {
            // Fallback: scalar GetValue/SetValue for non-32B-aligned GM addresses.
            for (int64_t i = 0; i < boxCount; ++i) {
                int64_t boxIdx = boxOffset + i;
                int64_t b = boxIdx / N_;
                int64_t n = boxIdx % N_;
                for (int32_t c = 0; c < RBE_BOX_CHANNELS; ++c) {
                    int64_t gmOff = b * RBE_BOX_CHANNELS * N_ + c * N_ + n;
                    float val = b2Local.GetValue(static_cast<uint32_t>(c * ubStride + i));
                    outGmF32_.SetValue(static_cast<uint64_t>(gmOff), val);
                }
            }
        }
    }

    // --- state ---
    AscendC::TPipe pipe_;
    // fp16 path (Branch-0, P=4): b0=fp16 IO / b1=fp32 anchor / b2=fp32 gt / b3=fp32 output.
    // fp32 path (Branch-1, P=3): b0=fp32 anchor / b1=fp32 gt / b2=fp32 output (b3 unused).
    AscendC::TBuf<AscendC::TPosition::VECCALC> b0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b3Buf_; // fp16 path only

    AscendC::GlobalTensor<half> anchorGm_; // fp16 path (branch-0)
    AscendC::GlobalTensor<half> gtGm_;
    AscendC::GlobalTensor<half> outGm_;
    AscendC::GlobalTensor<float> anchorGmF32_; // fp32 path (branch-1)
    AscendC::GlobalTensor<float> gtGmF32_;
    AscendC::GlobalTensor<float> outGmF32_;

    const RotatedBoxEncodeTilingData* td_ = nullptr;
    int64_t N_ = 0;
    int64_t B_ = 0;
    bool empty_ = true;
};

#endif // ROTATED_BOX_ENCODE_KERNEL_H
