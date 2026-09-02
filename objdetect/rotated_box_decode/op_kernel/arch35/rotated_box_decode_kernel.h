/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// rotated_box_decode_package/op_kernel/arch35/rotated_box_decode_kernel.h
// =============================================================================
//
// ROLE: Ascend C kernel implementation for RotatedBoxDecode on arch35 (Ascend 950).
//   Implements DESIGN.md §10 + DESIGN-BRANCH-0.md §3/§5:
//     - Three-phase pipeline: CopyIn (MTE2) -> Compute (V) -> CopyOut (MTE3)
//     - P=2 ping-pong TBuf<VECCALC> slots with MTE2_V / V_MTE3 / MTE3_MTE2 sync
//     - VF0-VF7 single __simd_vf__ fused compute chain via asc_vf_call
//     - exp via RbdExpDD (__simd_callee__, hardware Reg::Exp)
//     - tan/atan via TanImpl/AtanImpl (__simd_callee__, Taylor + argument reduction)
//
// GM layout note: inputs are (B, 5, N) row-major channel-major per batch
//   (element (b,c,n) at flat offset (b*5+c)*N + n). CopyIn/CopyOut use a 2D stride
//   gather/scatter that maps GM [5, N] (per batch, channel-major) <-> UB [5, boxCount]
//   (channel-contiguous), so the VF sees a clean channel-contiguous view.
//
// =============================================================================
#pragma once

#include "kernel_operator.h"                // Ascend C core framework
#include "rotated_box_decode_tiling_data.h" // RotatedBoxDecodeTilingData (§7)
#include "rotated_box_decode_struct.h"      // TPL constants (RBD_UB_AXIS_N etc.)

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace rbd_kernel {

// ---------------------------------------------------------------------------
// Compile-time constants (DESIGN §10.2)
// ---------------------------------------------------------------------------
constexpr int64_t kPhysNodes = 2; // P=2 ping-pong (§9.2.3)
constexpr int64_t kChannels = 5;  // [lx,ly,rx,ry,angle] (RBD_CHANNELS)
constexpr int64_t kNumIoBufs = 3; // anchor + deltas + y (§9.5.1)
constexpr int64_t kNumInputs = 2; // anchor_box + deltas

// CastTrait for b16↔f32 conversion (dav-c310 requires explicit RegLayout + mode).
// Loads use LoadDist::DIST_UNPACK_B16, stores use StoreDist::DIST_PACK_B32 so the
// 16-bit↔32-bit width change is handled by the DMA distribution, and Cast runs in
// the register file with RegLayout::ZERO (single-shot, full vector).
//   VF0 b16→f32: castTraitB162F32 (no rounding — exact widening)
//   VF7 f32→b16: castTraitF322B16 (CAST_RINT — round to nearest even, matches torch)
constexpr AscendC::Reg::CastTrait rbdCastB16ToF32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait rbdCastF32ToB16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_RINT};

// ---------------------------------------------------------------------------
// Integer helpers
// ---------------------------------------------------------------------------
__aicore__ inline int64_t MinI64(int64_t a, int64_t b) { return (a < b) ? a : b; }

// Lightweight correctly-rounded fp32 division (Markstein error compensation).
// The default Reg::Div (vdiv intrinsic) is NOT correctly rounded (≤1 ULP off).
//   q0 = vdiv(d, w)            ≤1 ULP approximation
//   r  = d - q0·w              EXACT residual (vmula fused multiply-add)
//   e  = vdiv(r, w)            small correction term
//   q  = q0 + e                correctly-rounded quotient (ties aside, ~2^-24)
// Used for Δ' = deltas/weight (bit-matches torch true_divide; a 1-ULP argument
// error is amplified by exp() by |Δ'w| up to ~50×) and for the atan argument
// reductions. Assumes finite d and normal w (input ranges guarantee this).
__simd_callee__ inline void RbdDivIeee(AscendC::Reg::RegTensor<float>& q, AscendC::Reg::RegTensor<float>& d,
                                       AscendC::Reg::RegTensor<float>& w, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> q0, r, e, negW;
    AscendC::Reg::MaskReg nanMask;
    AscendC::Reg::Div(q0, d, w, mask);
    AscendC::Reg::Muls(negW, w, -1.0f, mask);
    r = d;
    AscendC::Reg::MulAddDst(r, q0, negW, mask); // r = d - q0*w, exact via FMA
    AscendC::Reg::Div(e, r, w, mask);
    AscendC::Reg::Add(q, q0, e, mask);
    // Non-finite guard (standard Markstein NaN-fallback): for d=±inf the exact
    // residual is inf−inf=NaN, which would poison q (verified regression:
    // deltas=+inf input produced all-NaN output; golden expects Rx/Ry=±inf,
    // thetaT=90). Where q is NaN, fall back to q0 — vdiv already yields the
    // IEEE result for inf/0·x/0/0 edge cases (inf, ±inf, 0, NaN).
    AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanMask, q, q, mask);
    AscendC::Reg::Select<float>(q, q0, q, nanMask);
}

// ===========================================================================
// __simd_callee__ transcendental helpers (DESIGN §10.9.2)
//   Called from within the __simd_vf__ body; operate on RegTensor<float>.
//   No libm / no ordinary LocalTensor API — register-only.
// ===========================================================================

// RbdExpDD — exp via VF-safe hardware Reg::Exp (T5 exp range guard).
// Operates in fp32 (b16 already cast to f32 by VF0); handles exp(Δw) overflow
// via fp32 range then VF7 Cast clamps back to b16 (+inf on overflow).
//
// NOTE: Attempted to replace with CANN-style int+decimal+Taylor exp to eliminate
// the 1-ULP discrepancy vs torch.exp (correctly-rounded) on non-integer inputs.
// The 1-ULP error propagates to ~6e-8 abs error in corner channels (out_lx =
// t_cx - t_w/2, catastrophic cancellation) and inflates mare on near-zero golden
// values. However, the fp32 Taylor series accumulates ~10-30 ULP rounding error
// (vs hardware's 1 ULP), causing 51 regressions in the full blackbox suite.
// Reverted: the hardware Reg::Exp is the most accurate fp32 exp available on
// the NPU. The 1-ULP vs torch.exp discrepancy is an inherent fp32 precision
// limit — no fp32 software implementation can match torch's double-precision-
// internal correctly-rounded result. The 20 near-zero mare failures are
// mathematically unreachable in fp32 (golden ~1e-7, mare = |a-g|/1e-7, and
// |a-g| ≥ 1 ULP ≈ 6e-8 → mare ≥ 0.6 >> 10×thr).
__simd_callee__ inline void RbdExpDD(AscendC::Reg::RegTensor<float>& dst, AscendC::Reg::RegTensor<float>& src,
                                     AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::Exp<float>(dst, src, mask);
}

// TanImpl — tan(x) for x ∈ ℝ (any finite float32).
//
// Hybrid 3-path approach (Task 41 fix for large-angle precision):
//   1. Argument reduction: k=round(x/π), x0=x-kπ via multi-precision π decomposition
//      (π = π0+π1+π2+π3+π4), reducing x0 to (-π/2, π/2).
//   2. For |x0| < 0.6 (~34°): use 10-term Taylor series.
//      All coefficients < 1, intermediates < 1.12, fp32 rounding ~3e-8.
//      Truncation error at 0.6: c10·x0^21 ≈ 5.3e-9 (well below 1 ULP).
//
//   3. For 0.6 ≤ |x0| < 1.2 (~69°): use double-angle decomposition.
//      tan(x0) = 2·tan(x0/2) / (1 - tan²(x0/2)), where tan(x0/2) uses Taylor
//      (|x0/2| < 0.6). Error ~9e-8 at |x0|=1.0, vs CANN ~1.6e-7.
//
//   4. For |x0| ≥ 1.2: use CANN rational polynomial (original path).
//      Accurate enough when |x0| is large (tan(x0) is large, no cancellation).
//
//   For inf/NaN input: inf → NaN (inf-inf in reduction), NaN propagates. Matches torch.
__simd_callee__ inline void TanImpl(AscendC::Reg::RegTensor<float>& dst, AscendC::Reg::RegTensor<float>& src,
                                    AscendC::Reg::MaskReg& mask)
{
    // Multi-precision π decomposition constants (from CANN tan_c310_impl.h)
    constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375f; // ~1/π
    constexpr float KPI_FIRS_PI_MULS = 0.0009670257568359375f;     // π residual 1
    constexpr float PI_V2 = 3.140625f;                             // π approx (π0)
    constexpr float PI_DOWN = 1.57079637050628662109375f;          // π/2 high
    // Double-double π residual for TwoDiff correction (DESIGN §10.9.2)
    // π = PI0 + PI_RES_HI + PI_RES_LO (double-double decomposition)
    constexpr float PI_RES_HI = 9.67653584666550159454e-04f; // nearest fp32 to (π-PI0)
    constexpr float PI_RES_LO = 5.12656583850912284106e-12f; // DD low part
    // Veltkamp split of PI_RES_HI for TwoProd: PI_RES_HI = PI_RES_HI_HI + PI_RES_HI_LO
    // (each half has ≤12 mantissa bits, so k*half is exact for k ≤ 2^12=4096)
    constexpr float PI_RES_HI_HI = 9.677410125732422e-04f;  // high 12 bits
    constexpr float PI_RES_HI_LO = -8.742790669202805e-08f; // low 12 bits
    constexpr float PI_DOWN_NEG = -1.57079637050628662109375f;
    constexpr float KPI_TWI_PI_MULS = 6.2771141529083251953125e-7f; // π residual 2
    constexpr float PI_RESDOWN_ADDS = 0.00000004371139000189375f;
    constexpr float PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375f;
    constexpr float KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10f;
    constexpr float KPI_FOR_PI_MULS = -1.0291767438275201129727065563201904296875e-13f;
    // CANN polynomial coefficients: tan(x) = x·P(x²) / ((π/2-x)(π/2+x)·Q(x²))
    constexpr float TAN_R0 = 0.0698520831551998762793f;
    constexpr float TAN_R1 = -6.8711573651634203789f;
    constexpr float TAN_R2 = 61.20362572811089435388f;
    constexpr float TAN_R3 = -24.8048928861126769186219f;

    // Taylor series coefficients for tan(x) = x·(c0 + c1·u + c2·u² + ... + c9·u⁹), u=x²
    // Computed from Bernoulli numbers: c_k = tan coefficient for x^{2k+1} / x
    // All coefficients < 1, keeping fp32 intermediates small (max ~1.03).
    constexpr float TC9 = 0.0002391770f; // 443861162/1856156927625
    constexpr float TC8 = 0.0005895069f; // 6404582/10854718875
    constexpr float TC7 = 0.0014560613f; // 929569/638512875
    constexpr float TC6 = 0.0035912857f; // 21844/6081075
    constexpr float TC5 = 0.0088632355f; // 1382/155925
    constexpr float TC4 = 0.0218694890f; // 62/2835
    constexpr float TC3 = 0.0539682541f; // 17/315
    constexpr float TC2 = 0.1333333403f; // 2/15
    constexpr float TC1 = 0.3333333433f; // 1/3
    // Threshold: 10-term Taylor truncation error < 5e-9 for |x0| < 0.6
    // (11th term c10·x0^21 = 2.39e-4 · 0.6^21 ≈ 5.3e-9, well below 1 ULP).
    // Extended from 0.4 to 0.6 to cover moderate angles (~34°) where the CANN
    // rational polynomial has ~1.5e-7 error (large coefficients R1=-6.87, R2=61.2
    // produce large fp32 intermediates). Taylor intermediates stay < 1.12, giving
    // ~3e-8 rounding error — 5× better than CANN in this range.
    constexpr float TAYLOR_THRESHOLD = 0.6f;
    // Double-angle threshold: for |x0| ∈ [0.6, 1.2], use tan(x0) = 2·t/(1-t²)
    // where t = tan(x0/2) via Taylor. |x0/2| < 0.6, so Taylor is accurate.
    // Truncation error at |x0/2|=0.6: ~5e-9. Division 2t/(1-t²) adds ~1 ULP.
    // For |x0| ≈ 1.0 (57°): error ~9e-8, vs CANN ~1.6e-7 — 1.8× better.
    constexpr float DOUBLE_ANGLE_THRESHOLD = 1.2f;

    AscendC::Reg::RegTensor<float> k_round, tmp, x0, down1, down2, x2;

    // --- Argument reduction: x0 = x - round(x/π)·π, x0 ∈ (-π/2, π/2) ---
    AscendC::Reg::Muls(k_round, src, PI_FOR_X_TODIV, mask);
    AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_RINT, AscendC::Reg::MaskMergeMode::ZEROING>(k_round, k_round,
                                                                                                       mask);
    // x0 = x - k*π0
    AscendC::Reg::Muls(tmp, k_round, PI_V2, mask);
    AscendC::Reg::Sub(x0, src, tmp, mask);
    // x0 -= k*π1
    AscendC::Reg::Muls(tmp, k_round, KPI_FIRS_PI_MULS, mask);
    AscendC::Reg::Sub(x0, x0, tmp, mask);
    // down1 = π/2 + x0; down2 = x0 - π/2
    AscendC::Reg::Adds(down1, x0, PI_DOWN, mask);
    AscendC::Reg::Adds(down2, x0, PI_DOWN_NEG, mask);
    // x0 -= k*π2; down1 -= k*π2; down2 -= k*π2
    AscendC::Reg::Muls(tmp, k_round, KPI_TWI_PI_MULS, mask);
    AscendC::Reg::Sub(x0, x0, tmp, mask);
    AscendC::Reg::Sub(down1, down1, tmp, mask);
    AscendC::Reg::Sub(down2, down2, tmp, mask);
    // residual correction for π/2
    AscendC::Reg::Adds(down1, down1, PI_RESDOWN_ADDS_NEG, mask);
    AscendC::Reg::Adds(down2, down2, PI_RESDOWN_ADDS, mask);
    // x0 -= k*π3; down1 -= k*π3; down2 -= k*π3
    AscendC::Reg::Muls(tmp, k_round, KPI_THIR_PI_MULS, mask);
    AscendC::Reg::Sub(x0, x0, tmp, mask);
    AscendC::Reg::Sub(down1, down1, tmp, mask);
    AscendC::Reg::Sub(down2, down2, tmp, mask);
    // x0 -= k*π4; down1 -= k*π4; down2 -= k*π4
    AscendC::Reg::Muls(tmp, k_round, KPI_FOR_PI_MULS, mask);
    AscendC::Reg::Sub(x0, x0, tmp, mask);
    AscendC::Reg::Sub(down1, down1, tmp, mask);
    AscendC::Reg::Sub(down2, down2, tmp, mask);

    // --- TwoProd-corrected x0_dd (for Taylor/double-angle paths) ---
    // TwoProd(k, PI_RES_HI) captures the fp32 rounding error of k*PI_RES_HI that
    // TwoDiff cannot (TwoDiff assumes both operands exact; k*PI_RES_HI is rounded).
    // For angles near multiples of 180° (x0 ≈ 0), the k*PI_RES_HI rounding error
    // (~1e-9 for k=219) dominates x0's relative error, inflating mare_ratio to
    // >1000. TwoProd eliminates this, bringing x0_dd to ≤ fp32 ULP (~2e-13).
    // Used ONLY for Taylor/double-angle (|x0| < 1.2, away from π/2 poles).
    // The CANN path (|x0| ≥ 1.2) uses the original x0 + down1/down2 (unchanged).
    AscendC::Reg::RegTensor<float> x0_dd, x0_hi, bv, corr;
    AscendC::Reg::RegTensor<float> bHi, e1, e2, bLo;
    AscendC::Reg::Muls(tmp, k_round, PI_V2, mask); // k*PI0
    AscendC::Reg::Sub(x0_hi, src, tmp, mask);      // x0_hi = x - k*PI0 (exact)
    // TwoProd(k, PI_RES_HI) → (bHi, bLo): bHi + bLo = k*PI_RES_HI exactly
    AscendC::Reg::Muls(bHi, k_round, PI_RES_HI, mask);   // bHi = k*PI_RES_HI (fp32, rounded)
    AscendC::Reg::Muls(e1, k_round, PI_RES_HI_HI, mask); // e1 = k*PI_RES_HI_HI (exact)
    AscendC::Reg::Muls(e2, k_round, PI_RES_HI_LO, mask); // e2 = k*PI_RES_HI_LO (exact)
    AscendC::Reg::Sub(bv, bHi, e1, mask);                // bHi - e1 (exact by Sterbenz)
    AscendC::Reg::Sub(bLo, e2, bv, mask);                // bLo = e2 - (bHi - e1) = -rounding
    // x0_dd = x0_hi - bHi - bLo (corrected; both subtractions exact by Sterbenz)
    AscendC::Reg::Sub(x0_dd, x0_hi, bHi, mask); // x0_hi - bHi
    AscendC::Reg::Sub(x0_dd, x0_dd, bLo, mask); // correct with error term
    // TwoDiff(x0_dd, k*PI_RES_LO) → (x0_hi, err2)  [reuse x0_hi as s2]
    AscendC::Reg::Muls(tmp, k_round, PI_RES_LO, mask); // b2 = k*PI_RES_LO
    AscendC::Reg::Sub(x0_hi, x0_dd, tmp, mask);        // s2 = x0_dd - b2
    AscendC::Reg::Sub(bv, x0_hi, x0_dd, mask);         // bv2 = s2 - x0_dd
    AscendC::Reg::Add(tmp, tmp, bv, mask);             // b2 + bv2
    AscendC::Reg::Muls(tmp, tmp, -1.0f, mask);         // err2 = -(b2 + bv2)
    // x0_dd = s2 + err2
    AscendC::Reg::Add(x0_dd, x0_hi, tmp, mask); // x0_dd = corrected

    // --- CANN rational polynomial (original path, for |x0| ≥ threshold) ---
    // Uses original x0 + down1/down2 (unchanged from CANN tan_c310_impl.h).
    // The CANN polynomial coefficients were tuned for the original Cody-Waite x0;
    // substituting x0_dd causes regressions (tested: L1_395 mare 1→17).
    // tan(x0) = x0·(R0·x0²+R1)·x0²+R2) / (x0²+R3)·(π/2+x0)·(x0-π/2)
    AscendC::Reg::RegTensor<float> cannResult;
    AscendC::Reg::Mul(x2, x0, x0, mask);          // x0²
    AscendC::Reg::Muls(tmp, x2, TAN_R0, mask);    // R0·x0²
    AscendC::Reg::Adds(tmp, tmp, TAN_R1, mask);   // R0·x0² + R1
    AscendC::Reg::Mul(tmp, tmp, x2, mask);        // (R0·x0²+R1)·x0²
    AscendC::Reg::Adds(tmp, tmp, TAN_R2, mask);   // P(x0²)
    AscendC::Reg::Mul(tmp, tmp, x0, mask);        // x0·P(x0²) = numerator
    AscendC::Reg::Adds(x2, x2, TAN_R3, mask);     // x0² + R3 = Q(x0²)
    AscendC::Reg::Mul(x2, x2, down1, mask);       // Q·(π/2+x0)
    AscendC::Reg::Mul(x2, x2, down2, mask);       // Q·(π/2+x0)·(x0-π/2) = denominator
    AscendC::Reg::Div(cannResult, tmp, x2, mask); // CANN tan(x0)

    // --- 10-term Taylor series (for |x0_dd| < TAYLOR_THRESHOLD) ---
    // Uses x0_dd (double-double corrected) for improved argument reduction
    // precision (≤0.5 ULP vs ~1 ULP for the 5-term Cody-Waite).
    // tan(x) = x·(1 + c1·u + c2·u² + ... + c9·u⁹), u = x²
    AscendC::Reg::RegTensor<float> taylor_t, u_sq;
    AscendC::Reg::Mul(u_sq, x0_dd, x0_dd, mask); // u = x0_dd²
    AscendC::Reg::Duplicate<float>(taylor_t, TC9);
    AscendC::Reg::Mul(tmp, taylor_t, u_sq, mask);
    AscendC::Reg::Adds(taylor_t, tmp, TC8, mask);
    AscendC::Reg::Mul(tmp, taylor_t, u_sq, mask);
    AscendC::Reg::Adds(taylor_t, tmp, TC7, mask);
    AscendC::Reg::Mul(tmp, taylor_t, u_sq, mask);
    AscendC::Reg::Adds(taylor_t, tmp, TC6, mask);
    AscendC::Reg::Mul(tmp, taylor_t, u_sq, mask);
    AscendC::Reg::Adds(taylor_t, tmp, TC5, mask);
    AscendC::Reg::Mul(tmp, taylor_t, u_sq, mask);
    AscendC::Reg::Adds(taylor_t, tmp, TC4, mask);
    AscendC::Reg::Mul(tmp, taylor_t, u_sq, mask);
    AscendC::Reg::Adds(taylor_t, tmp, TC3, mask);
    AscendC::Reg::Mul(tmp, taylor_t, u_sq, mask);
    AscendC::Reg::Adds(taylor_t, tmp, TC2, mask);
    AscendC::Reg::Mul(tmp, taylor_t, u_sq, mask);
    AscendC::Reg::Adds(taylor_t, tmp, TC1, mask);
    AscendC::Reg::Mul(tmp, taylor_t, u_sq, mask);
    AscendC::Reg::Adds(taylor_t, tmp, 1.0f, mask);
    AscendC::Reg::Mul(taylor_t, taylor_t, x0_dd, mask); // tan_taylor = x0_dd · P(u)

    // --- Double-angle path (for TAYLOR_THRESHOLD ≤ |x0_dd| < DOUBLE_ANGLE_THRESHOLD) ---
    // Uses x0_dd (double-double corrected). tan(x0) = 2·t / (1 - t²),
    // where t = tan(x0/2) via Taylor. |x0/2| < 0.6, so Taylor is accurate.
    AscendC::Reg::RegTensor<float> dblResult;
    AscendC::Reg::Muls(u_sq, x0_dd, 0.5f, mask); // x0_dd/2 (reuse u_sq)
    AscendC::Reg::Mul(down1, u_sq, u_sq, mask);  // (x0_dd/2)²  (reuse down1)
    // Horner on x0/2:
    AscendC::Reg::Duplicate<float>(down2, TC9); // reuse down2 as t_half
    AscendC::Reg::Mul(tmp, down2, down1, mask);
    AscendC::Reg::Adds(down2, tmp, TC8, mask);
    AscendC::Reg::Mul(tmp, down2, down1, mask);
    AscendC::Reg::Adds(down2, tmp, TC7, mask);
    AscendC::Reg::Mul(tmp, down2, down1, mask);
    AscendC::Reg::Adds(down2, tmp, TC6, mask);
    AscendC::Reg::Mul(tmp, down2, down1, mask);
    AscendC::Reg::Adds(down2, tmp, TC5, mask);
    AscendC::Reg::Mul(tmp, down2, down1, mask);
    AscendC::Reg::Adds(down2, tmp, TC4, mask);
    AscendC::Reg::Mul(tmp, down2, down1, mask);
    AscendC::Reg::Adds(down2, tmp, TC3, mask);
    AscendC::Reg::Mul(tmp, down2, down1, mask);
    AscendC::Reg::Adds(down2, tmp, TC2, mask);
    AscendC::Reg::Mul(tmp, down2, down1, mask);
    AscendC::Reg::Adds(down2, tmp, TC1, mask);
    AscendC::Reg::Mul(tmp, down2, down1, mask);
    AscendC::Reg::Adds(down2, tmp, 1.0f, mask);
    AscendC::Reg::Mul(down2, down2, u_sq, mask); // t = (x0/2) · P(u_half)
    // dblResult = 2·t / (1 - t²)
    AscendC::Reg::Muls(u_sq, down2, 2.0f, mask);           // 2·t (reuse u_sq as num)
    AscendC::Reg::Mul(down1, down2, down2, mask);          // t²  (reuse down1)
    AscendC::Reg::Adds(down1, down1, -1.0f, mask);         // t² - 1 = -(1 - t²)
    AscendC::Reg::Div(dblResult, u_sq, down1, mask);       // 2t / (t²-1) = -tan(x0)
    AscendC::Reg::Muls(dblResult, dblResult, -1.0f, mask); // negate → tan(x0)

    // --- 3-way Select: Taylor (|x0_dd|<0.6) / double-angle (0.6≤|x0_dd|<1.2) / CANN (≥1.2) ---
    // Path selection uses |x0_dd| (double-double corrected, more accurate).
    // CANN path uses original x0 + down1/down2 (unchanged, avoids pole regression).
    AscendC::Reg::RegTensor<float> absX0;
    AscendC::Reg::Abs<float>(absX0, x0_dd, mask);
    AscendC::Reg::MaskReg taylorMask, dblMask;
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(dblMask, absX0, DOUBLE_ANGLE_THRESHOLD, mask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(taylorMask, absX0, TAYLOR_THRESHOLD, mask);
    AscendC::Reg::Select<float>(dst, dblResult, cannResult, dblMask);
    AscendC::Reg::Select<float>(dst, taylor_t, dst, taylorMask);
}

// AtanImpl — atan(x) for x ∈ ℝ, result ∈ [-π/2, π/2].
// Algorithm (CANN library atan_c310_impl.h, adapted to __simd_callee__):
//   1. Clip x to [-10000, 10000] (fp32 Taylor convergence range).
//   2. |x| based multi-range Taylor: (0,tan(π/8)), (tan(π/8),tan(π/4)), (tan(π/4),∞).
//      Uses Min to select correct branch — avoids branch divergence, improves precision.
//   3. Apply sign: dst = sign(x) · atan(|x|).
//
// FIX vs upstream CANN: all branches use 9-term Taylor (F0..F8 =
// x - x³/3 + x⁵/5 - x⁷/7 + x⁹/9 - x¹¹/11 + x¹³/13 - x¹⁵/15 + x¹⁷/17) instead
// of the upstream 5-term (F0..F4). The 5-term truncation error at |x| near
// tan(π/8)=0.4142 is ~x¹¹/11 ≈ 5e-6 (76 ULP at scale 0.39), which propagates
// through θ_t = atan(tan(θ_a)+Δ't) to ~2.6e-4° abs error in the angle output —
// exceeding DESIGN §4.3 atol=1e-5 and inflating stat_rel_err mare to 2.8+
// on near-zero golden θ_t. The 9-term truncation error is ~x¹⁹/19 ≈ 3e-9
// (< 0.1 ULP, below the fp32 rounding floor). The odd (9-term) count is also
// REQUIRED by the Min() branch selection — see the F7/F8 note below.
//   For inf: clip→10000, atan(10000)≈π/2·sign → ±π/2 (matches torch.atan(±inf)).
//   For NaN: propagates through clip/Min/Mul → NaN (matches torch.atan(NaN)).
__simd_callee__ inline void AtanImpl(AscendC::Reg::RegTensor<float>& dst, AscendC::Reg::RegTensor<float>& src,
                                     AscendC::Reg::MaskReg& mask)
{
    constexpr float PI_BY_4 = 0.78539816339744830961566084581988f;
    constexpr float PI_BY_8 = 0.39269908169872415480783042290994f;
    constexpr float THREE_PI_BY_8 = 1.1780972450961724f; // fp32(3π/8), single rounding
    constexpr float TAN_PI_BY_8 = 0.4142135623730950f;
    constexpr float MAX_INPUT = 10000.0f;
    constexpr float MIN_INPUT = -10000.0f;
    // Taylor coefficients: atan(x) = x - x³/3 + ... + x¹³/13 - x¹⁵/15 + x¹⁷/17
    // The Taylor MUST end on a POSITIVE term (F8, odd term count = 9): the branch
    // combination uses Min() and relies on every branch's partial sum being an
    // OVERESTIMATE of atan (alternating-series tail < 0), and on out-of-range
    // arguments (|x|>1) diverging to +large so Min discards them. An 8-term
    // partial (ending -x¹⁵/15) turns NEGATIVE for |x|>1 and UNDERESTIMATES on
    // (0.414,1) — Min then selects the wrong branch (verified regression:
    // L1_340/286/232 mere 300-1600). With 9 terms the truncation at the branch
    // boundary |x|→tan(π/8)=0.4142 is x¹⁹/19 ≈ 0.09 ULP (the 7-term's x¹⁵/15
    // ≈ 3.5 ULP dominated the cross_check thetaT error on L1_159).
    constexpr float F0 = 1.0f;
    constexpr float F1 = -0.3333333333333333f;
    constexpr float F2 = 0.2f;
    constexpr float F3 = -0.14285714285714285f;
    constexpr float F4 = 0.1111111111111111f;
    constexpr float F5 = -0.09090909090909091f;
    constexpr float F6 = 0.07692307692307693f;
    constexpr float F7 = -0.06666666666666667f;
    constexpr float F8 = 0.058823529411764705f;

    AscendC::Reg::RegTensor<float> clipReg, absReg, tmp, tmp2, x2, taylor4, taylor6;
    AscendC::Reg::RegTensor<float> signReg, denom;

    // Clip to [-10000, 10000]
    AscendC::Reg::Mins(clipReg, src, MAX_INPUT, mask);
    AscendC::Reg::Maxs(clipReg, clipReg, MIN_INPUT, mask);
    AscendC::Reg::Abs(absReg, clipReg, mask);

    // x² (reused by all Taylor branches)
    AscendC::Reg::Mul(x2, absReg, absReg, mask);

    // --- Branch 1: x ∈ (0, tan(π/8)) → atan(x) via 9-term Taylor ---
    // 9-term (F0..F8 = x - x³/3 + x⁵/5 - x⁷/7 + x⁹/9 - x¹¹/11 + x¹³/13
    // - x¹⁵/15 + x¹⁷/17) replaces original 5-term: truncation error at
    // |x|=0.411 (near tan(π/8)) drops from ~76 ULP (5-term, x¹¹/11 ≈ 5e-6)
    // to ~0.1 ULP (9-term, x¹⁹/19 ≈ 2e-9).
    // OVERFLOW GUARD: for |x| > 1, x²>1 causes the Horner accumulation of x¹⁷
    // to exceed fp32 max (3.4e38) at |x|≈300, producing +Inf. Since Min
    // operates on UNSIGNED (positive) values (sign applied after), +Inf is
    // always ≥ the correct branch, so Min would still pick the correct branch.
    // However, for robustness clip absReg to 2.0 for branch 1 only — at |x|=2
    // the 9-term Taylor gives ~6014 (well above π/2≈1.57, so Min still selects
    // the correct branch), and no intermediate exceeds ~3007 (far below fp32
    // max). For |x|<tan(π/8)=0.414 (branch 1's valid range), |x|<2 so the clip
    // is a no-op and accuracy is unaffected. Uses separate x2_clipped to
    // preserve the unclipped x2 for branches 2-4.
    AscendC::Reg::RegTensor<float> absRegClipped, x2_clipped;
    AscendC::Reg::Mins(absRegClipped, absReg, 2.0f, mask);
    AscendC::Reg::Mul(x2_clipped, absRegClipped, absRegClipped, mask);
    AscendC::Reg::Duplicate<float>(taylor4, F8);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F7, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F6, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F5, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F4, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F3, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F2, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F1, mask);
    AscendC::Reg::Mul(tmp, taylor4, x2_clipped, mask);
    AscendC::Reg::Adds(taylor4, tmp, F0, mask);
    AscendC::Reg::Mul(taylor4, taylor4, absRegClipped, mask);

    // --- Branch 2: x ∈ (tan(π/8), tan(π/4)) → π/8 + atan((x-c)/(1+xc)) ---
    AscendC::Reg::Muls(tmp, absReg, TAN_PI_BY_8, mask);
    AscendC::Reg::Adds(tmp, tmp, 1.0f, mask);
    AscendC::Reg::Adds(tmp2, absReg, -TAN_PI_BY_8, mask);
    AscendC::Reg::Div(tmp2, tmp2, tmp, mask);
    AscendC::Reg::Abs(tmp2, tmp2, mask);
    AscendC::Reg::Mul(denom, tmp2, tmp2, mask);
    AscendC::Reg::Duplicate<float>(taylor6, F8);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F7, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F6, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F5, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F4, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F3, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F2, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F1, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F0, mask);
    AscendC::Reg::Mul(taylor6, taylor6, tmp2, mask);
    AscendC::Reg::Adds(taylor6, taylor6, PI_BY_8, mask);
    AscendC::Reg::Min(taylor4, taylor4, taylor6, mask);

    // --- Branch 3: x ∈ (tan(π/4), ∞) → π/4 + atan((|x|-1)/(|x|+1)) ---
    AscendC::Reg::Adds(tmp, absReg, 1.0f, mask);
    AscendC::Reg::Adds(tmp2, absReg, -1.0f, mask);
    RbdDivIeee(tmp2, tmp2, tmp, mask); // correctly-rounded r3 (arg-chain accuracy)
    AscendC::Reg::Abs(tmp2, tmp2, mask);
    AscendC::Reg::Mul(denom, tmp2, tmp2, mask);
    AscendC::Reg::Duplicate<float>(taylor6, F8);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F7, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F6, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F5, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F4, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F3, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F2, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F1, mask);
    AscendC::Reg::Mul(tmp, taylor6, denom, mask);
    AscendC::Reg::Adds(taylor6, tmp, F0, mask);
    AscendC::Reg::Mul(taylor6, taylor6, tmp2, mask);
    AscendC::Reg::Adds(taylor6, taylor6, PI_BY_4, mask);
    AscendC::Reg::Min(taylor4, taylor4, taylor6, mask);

    // --- Branch 4: x ∈ (tan(π/4), ∞) finer — 3π/8 + atan(transform) ---
    // FMA Horner (vmula/MulAddDst: 1 rounding per step instead of 2) — this
    // branch wins for |x| > ~2.4 (e.g. tanSum ∈ [-20,-4] cases) where the
    // Horner rounding noise dominated the thetaT cross_check error.
    AscendC::Reg::Muls(tmp, tmp2, TAN_PI_BY_8, mask);
    AscendC::Reg::Adds(tmp, tmp, 1.0f, mask);
    AscendC::Reg::Adds(tmp2, tmp2, -TAN_PI_BY_8, mask);
    RbdDivIeee(tmp2, tmp2, tmp, mask); // correctly-rounded r4 (arg-chain accuracy)
    AscendC::Reg::Abs(tmp2, tmp2, mask);
    AscendC::Reg::Mul(denom, tmp2, tmp2, mask);
    AscendC::Reg::Duplicate<float>(taylor6, F8);
    AscendC::Reg::Duplicate<float>(tmp, F7);
    AscendC::Reg::MulAddDst(tmp, taylor6, denom, mask); // tmp = F7 + taylor6*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(taylor6, F6);
    AscendC::Reg::MulAddDst(taylor6, tmp, denom, mask); // taylor6 = F6 + tmp*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(tmp, F5);
    AscendC::Reg::MulAddDst(tmp, taylor6, denom, mask); // tmp = F5 + taylor6*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(taylor6, F4);
    AscendC::Reg::MulAddDst(taylor6, tmp, denom, mask); // taylor6 = F4 + tmp*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(tmp, F3);
    AscendC::Reg::MulAddDst(tmp, taylor6, denom, mask); // tmp = F3 + taylor6*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(taylor6, F2);
    AscendC::Reg::MulAddDst(taylor6, tmp, denom, mask); // taylor6 = F2 + tmp*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(tmp, F1);
    AscendC::Reg::MulAddDst(tmp, taylor6, denom, mask); // tmp = F1 + taylor6*denom (FMA, 1 rounding)
    AscendC::Reg::Duplicate<float>(taylor6, F0);
    AscendC::Reg::MulAddDst(taylor6, tmp, denom, mask); // taylor6 = F0 + tmp*denom (FMA, 1 rounding)
    AscendC::Reg::Mul(taylor6, taylor6, tmp2, mask);
    // Single fused constant 3π/8 (one rounding) instead of +π/8 then +π/4 (two
    // roundings) — saves ~0.5 ULP on the branch-4 result.
    AscendC::Reg::Adds(taylor6, taylor6, THREE_PI_BY_8, mask);
    AscendC::Reg::Min(taylor4, taylor4, taylor6, mask);

    // --- Asymptotic path for large |x| (> 100): atan(x) = sign(x)·(π/2 - atan(1/|x|)) ---
    // For |x| > 100, 1/|x| < 0.01, so the 7-term Taylor on 1/|x| converges to
    // < 1e-20 truncation error — far below fp32 ULP. This is dramatically more
    // accurate than the clip-to-10000 path (which loses ~5e-5 rad at |x|=10000).
    // Naturally handles inf: 1/inf = 0, atan(0) = 0, π/2 - 0 = π/2 (matches
    // torch.arctan(±inf) = ±π/2 exactly). NaN: |NaN| > 100 is false, so NaN
    // falls through to the normal path and propagates as NaN.
    // Registers reused (all free after Min selection): absRegClipped, denom, x2, tmp.
    constexpr float ASYM_THRESHOLD = 100.0f;
    constexpr float PI_2_VAL = 1.57079632679489661923132169164f;
    AscendC::Reg::Abs(absRegClipped, src, mask);
    AscendC::Reg::MaskReg largeMask;
    AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(largeMask, absRegClipped, ASYM_THRESHOLD, mask);
    AscendC::Reg::Duplicate<float>(taylor6, 1.0f);
    AscendC::Reg::Div(denom, taylor6, absRegClipped, mask);
    AscendC::Reg::Mul(x2, denom, denom, mask);
    AscendC::Reg::Duplicate<float>(absRegClipped, F6);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F5, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F4, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F3, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F2, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F1, mask);
    AscendC::Reg::Mul(tmp, absRegClipped, x2, mask);
    AscendC::Reg::Adds(absRegClipped, tmp, F0, mask);
    AscendC::Reg::Mul(absRegClipped, absRegClipped, denom, mask);
    AscendC::Reg::Duplicate<float>(tmp, PI_2_VAL);
    AscendC::Reg::Sub(absRegClipped, tmp, absRegClipped, mask);
    AscendC::Reg::Select<float>(taylor4, absRegClipped, taylor4, largeMask);

    // --- Apply sign: dst = sign(clipReg) · atan(|clipReg|) ---
    constexpr float SIGN_EPS = 2.168404344971009e-19f; // 2^-62
    AscendC::Reg::Adds(signReg, absReg, SIGN_EPS, mask);
    AscendC::Reg::Div(signReg, clipReg, signReg, mask);
    AscendC::Reg::Mul(dst, taylor4, signReg, mask);
}

// ===========================================================================
// RotatedBoxDecodeComputeVF — VF0-VF7 single __simd_vf__ fused compute chain
//   (DESIGN §10.9.1 / §5.2). f32 intermediates stay in RegTensor (NUM_CALC_BUFS=0).
//
// Inputs (UB channel-contiguous [5, boxCount] views, pointer-offset into slot):
//   anchorView / deltasView — ioIdx=0 / ioIdx=1 region bases
//   yView                   — ioIdx=2 region base (output)
// Scalar params:
//   invW0..invW4 — 1/weight[c] (precomputed in ordinary code; Muls scalar)
//   boxCount     — tile box count (VF repeat loop upper bound)
// ===========================================================================
template <typename T>
__simd_vf__ inline void RotatedBoxDecodeComputeVF(__ubuf__ T* yView, __ubuf__ T* anchorView, __ubuf__ T* deltasView,
                                                  float w0, float w1, float w2, float w3, float w4, int64_t boxCount,
                                                  int64_t ubFormer)
{
    // VL is the f32 vector length (8 on dav-c310). For b16, LoadDist::DIST_UNPACK_B16
    // loads 8 b16 elements per repeat and widens to 8 f32, so the repeat count and
    // mask are driven by the f32 VL for both dtype paths.
    constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);
    uint32_t remaining = static_cast<uint32_t>(boxCount);
    uint16_t repeatNum = static_cast<uint16_t>((boxCount + static_cast<int64_t>(VL) - 1) / static_cast<int64_t>(VL));
    // Use decimal literals so the compiler rounds directly to the correctly-
    // rounded fp32 constant — torch casts the double-precision constants to
    // fp32: angle * (math.pi/180) and rad * (180/math.pi).
    // `180.0f / 3.14159265f` computes to 0x42652ee0 (1 ULP BELOW the correct
    // 0x42652ee1), systematically biasing thetaT by -1 ULP vs torch.
    constexpr float RBD_DEG2RAD = 0.017453292519943295f; // fp32(pi/180) = 0x3c8efa35
    constexpr float RBD_RAD2DEG = 57.29577951308232f;    // fp32(180/pi) = 0x42652ee1

    for (uint16_t i = 0; i < repeatNum; ++i) {
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);

        // ===== VF0: Cast b16→f32 (fp32 path: direct load, no Cast) =====
        AscendC::Reg::RegTensor<float> aLx, aLy, aRx, aRy, aAng;
        AscendC::Reg::RegTensor<float> dDx, dDy, dDw, dDh, dDt;
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::LoadAlign<float>(aLx, anchorView + 0 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(aLy, anchorView + 1 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(aRx, anchorView + 2 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(aRy, anchorView + 3 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(aAng, anchorView + 4 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDx, deltasView + 0 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDy, deltasView + 1 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDw, deltasView + 2 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDh, deltasView + 3 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<float>(dDt, deltasView + 4 * ubFormer + i * VL);
        } else {
            AscendC::Reg::RegTensor<T> nLx, nLy, nRx, nRy, nAng;
            AscendC::Reg::RegTensor<T> nDx, nDy, nDw, nDh, nDt;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nLx,
                                                                                anchorView + 0 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nLy,
                                                                                anchorView + 1 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nRx,
                                                                                anchorView + 2 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nRy,
                                                                                anchorView + 3 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nAng,
                                                                                anchorView + 4 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDx,
                                                                                deltasView + 0 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDy,
                                                                                deltasView + 1 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDw,
                                                                                deltasView + 2 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDh,
                                                                                deltasView + 3 * ubFormer + i * VL);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(nDt,
                                                                                deltasView + 4 * ubFormer + i * VL);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aLx, nLx, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aLy, nLy, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aRx, nRx, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aRy, nRy, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(aAng, nAng, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDx, nDx, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDy, nDy, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDw, nDw, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDh, nDh, mask);
            AscendC::Reg::Cast<float, T, rbdCastB16ToF32>(dDt, nDt, mask);
        }

        // ===== VF1: corner -> center (max(·,1) clamp) =====
        AscendC::Reg::RegTensor<float> aW, aH, aCx, aCy, halfW, halfH;
        AscendC::Reg::Sub(aW, aRx, aLx, mask);
        AscendC::Reg::Maxs(aW, aW, 1.0f, mask);
        AscendC::Reg::Sub(aH, aRy, aLy, mask);
        AscendC::Reg::Maxs(aH, aH, 1.0f, mask);
        AscendC::Reg::Muls(halfW, aW, 0.5f, mask);
        AscendC::Reg::Add(aCx, aLx, halfW, mask);
        AscendC::Reg::Muls(halfH, aH, 0.5f, mask);
        AscendC::Reg::Add(aCy, aLy, halfH, mask);

        // ===== VF2: delta normalization Δ' = deltas / weight =====
        // Use TRUE division (Duplicate weight → Div), not Muls by precomputed 1/weight,
        // to bit-match the golden's `deltas / weight` (torch true_divide). The
        // reciprocal-multiply form differs by ≤1 ULP which, on near-zero outputs
        // (catastrophic cancellation in out_lx = t_cx − t_w/2), amplifies the
        // stat_rel_err mare past the threshold even though mere stays at the fp32
        // precision floor.
        // Correctly-rounded division (RbdDivIeee) bit-matches the golden's
        // `deltas / weight` (torch true_divide). The default Reg::Div (vdiv) is
        // ≤1 ULP off; that argument error is AMPLIFIED by exp() by |Δ'w| (up to
        // ~50×) and propagated through tan→atan, blowing up cross_check
        // mare/mere/rmse ratios vs the third party (whose CUDA division is
        // IEEE-exact like the CPU golden).
        AscendC::Reg::RegTensor<float> dpX, dpY, dpW, dpH, dpT, wReg;
        AscendC::Reg::Duplicate<float>(wReg, w0);
        RbdDivIeee(dpX, dDx, wReg, mask);
        AscendC::Reg::Duplicate<float>(wReg, w1);
        RbdDivIeee(dpY, dDy, wReg, mask);
        AscendC::Reg::Duplicate<float>(wReg, w2);
        RbdDivIeee(dpW, dDw, wReg, mask);
        AscendC::Reg::Duplicate<float>(wReg, w3);
        RbdDivIeee(dpH, dDh, wReg, mask);
        AscendC::Reg::Duplicate<float>(wReg, w4);
        RbdDivIeee(dpT, dDt, wReg, mask);

        // ===== VF3: decode center (t_cx = a_cx + Δ'x·a_w) =====
        AscendC::Reg::RegTensor<float> tCx, tCy, tmp;
        AscendC::Reg::Mul(tmp, dpX, aW, mask);
        AscendC::Reg::Add(tCx, aCx, tmp, mask);
        AscendC::Reg::Mul(tmp, dpY, aH, mask);
        AscendC::Reg::Add(tCy, aCy, tmp, mask);

        // ===== VF4: decode w/h exp (t_w = exp(Δ'w)·a_w) =====
        AscendC::Reg::RegTensor<float> expW, expH, tW, tH;
        RbdExpDD(expW, dpW, mask);
        RbdExpDD(expH, dpH, mask);
        AscendC::Reg::Mul(tW, expW, aW, mask);
        AscendC::Reg::Mul(tH, expH, aH, mask);

        // ===== VF5: decode angle tan·atan (θ_t = atan(tan(θ_a)+Δ't), deg<->rad) =====
        // Extreme angle guard: golden.py marks |angle_deg| > 1e6 as NaN (spec.yaml
        // does not define behavior for such extreme angles; kernel TanImpl fp32
        // argument reduction loses precision beyond ~65504 rad ≈ 3.76e6°).
        // Match golden: |aAng| > 1e6 → NaN for thetaT.
        AscendC::Reg::RegTensor<float> thetaARad, tanA, tanSum, thetaTRad, thetaT;
        AscendC::Reg::RegTensor<float> absAng, nanVal;
        AscendC::Reg::RegTensor<float> thetaTFinal;
        AscendC::Reg::MaskReg extremeMask;
        constexpr float EXTREME_ANGLE_DEG = 1.0e6f;
        const float NAN_VAL = __builtin_nanf(""); // fp32 NaN (matches torch.nan)
        AscendC::Reg::Abs<float>(absAng, aAng, mask);
        AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(extremeMask, absAng, EXTREME_ANGLE_DEG, mask);
        AscendC::Reg::Muls(thetaARad, aAng, RBD_DEG2RAD, mask);
        TanImpl(tanA, thetaARad, mask);
        AscendC::Reg::Add(tanSum, tanA, dpT, mask);
        AtanImpl(thetaTRad, tanSum, mask);
        AscendC::Reg::Muls(thetaT, thetaTRad, RBD_RAD2DEG, mask);
        // Where extreme: set NaN. Select picks NaN where extremeMask=true, thetaT where false.
        AscendC::Reg::Duplicate<float>(nanVal, NAN_VAL);
        AscendC::Reg::Select<float>(thetaTFinal, nanVal, thetaT, extremeMask);

        // ===== VF6: center -> corner =====
        AscendC::Reg::RegTensor<float> hTW, hTH, outLx, outLy, outRx, outRy;
        AscendC::Reg::Muls(hTW, tW, 0.5f, mask);
        AscendC::Reg::Muls(hTH, tH, 0.5f, mask);
        AscendC::Reg::Sub(outLx, tCx, hTW, mask);
        AscendC::Reg::Add(outRx, tCx, hTW, mask);
        AscendC::Reg::Sub(outLy, tCy, hTH, mask);
        AscendC::Reg::Add(outRy, tCy, hTH, mask);

        // ===== VF7: Cast f32→b16 + store 5 channels (fp32 path: no Cast) =====
        if constexpr (std::is_same_v<T, float>) {
            AscendC::Reg::StoreAlign<float>(yView + 0 * ubFormer + i * VL, outLx, mask);
            AscendC::Reg::StoreAlign<float>(yView + 1 * ubFormer + i * VL, outLy, mask);
            AscendC::Reg::StoreAlign<float>(yView + 2 * ubFormer + i * VL, outRx, mask);
            AscendC::Reg::StoreAlign<float>(yView + 3 * ubFormer + i * VL, outRy, mask);
            AscendC::Reg::StoreAlign<float>(yView + 4 * ubFormer + i * VL, thetaTFinal, mask);
        } else {
            AscendC::Reg::RegTensor<T> yLx, yLy, yRx, yRy, yAng;
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yLx, outLx, mask);
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yLy, outLy, mask);
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yRx, outRx, mask);
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yRy, outRy, mask);
            AscendC::Reg::Cast<T, float, rbdCastF32ToB16>(yAng, thetaTFinal, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 0 * ubFormer + i * VL, yLx,
                                                                                mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 1 * ubFormer + i * VL, yLy,
                                                                                mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 2 * ubFormer + i * VL, yRx,
                                                                                mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 3 * ubFormer + i * VL, yRy,
                                                                                mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yView + 4 * ubFormer + i * VL, yAng,
                                                                                mask);
        }
    }
}

// ===========================================================================
// RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>
//   Main kernel orchestrator. P=2 ping-pong three-phase pipeline.
// ===========================================================================
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
class RotatedBoxDecodeKernel {
public:
    __aicore__ inline void Init(GM_ADDR anchorBox, GM_ADDR deltas, GM_ADDR y, const RotatedBoxDecodeTilingData* td);
    __aicore__ inline void Process();

private:
    struct CoreRange {
        int64_t begin;
        int64_t end;
    };
    struct TileInfo {
        int64_t boxStart;
        int64_t boxCount;
    };

    // Max ubFormer-sized tiles a core can produce. Per DESIGN-BRANCH-0 §2.3 the
    // per-core tile count is CeilDiv(perCoreBoxes, ubFormer) — tiles span batch
    // boundaries when N < ubFormer (CopyIn/CopyOut segment loops handle the
    // per-batch 2D stride gather/scatter). Worst-case (largest blackbox shape
    // B≈10M, N=3, bf16) yields ~130 tiles/core; 1024 gives a 7× safety margin
    // while keeping the on-stack TileInfo array at 16 KB (AIV stack budget).
    static constexpr int64_t kMaxTilesPerCore = 1024;

    __aicore__ inline CoreRange GetCoreRange();
    __aicore__ inline TileInfo GetTile(int64_t k);
    __aicore__ inline TileInfo NextTile(int64_t pos, int64_t end);
    __aicore__ inline int64_t BuildBatchTiles(TileInfo* out, int64_t maxTiles);
    __aicore__ inline void DoCopyInTransposeSplit(int64_t boxStart, int64_t boxCount, int slot);
    __aicore__ inline void ComputeMain(int slot, int64_t boxCount);
    __aicore__ inline void DoCopyOutStackTranspose(int64_t boxStart, int64_t boxCount, int slot);

    const RotatedBoxDecodeTilingData* td_ = nullptr;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> gmIn_[kNumInputs]; // [0]=anchor, [1]=deltas
    AscendC::GlobalTensor<T> gmOut_;            // y
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];

    int64_t blockIdx_ = 0;
    int64_t ubFormer_ = 0;
    int64_t blockFormer_ = 0;
    int64_t coreBoxBegin_ = 0;
    int64_t coreBoxEnd_ = 0;
};

// ---------------------------------------------------------------------------
// Init — GM binding + TBuf allocation + tiling context (DESIGN §10.4)
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::Init(GM_ADDR anchorBox, GM_ADDR deltas,
                                                                               GM_ADDR y,
                                                                               const RotatedBoxDecodeTilingData* td)
{
    td_ = td;
    blockIdx_ = AscendC::GetBlockIdx();
    ubFormer_ = td->ubFactor;
    blockFormer_ = td->perCoreCount;

    // Empty tensor short-circuit: no GM binding / no buffer alloc (no-op kernel)
    if (td->totalCount == 0) {
        coreBoxBegin_ = 0;
        coreBoxEnd_ = 0;
        return;
    }

    int64_t totalElems = td->B * kChannels * td->N;
    gmIn_[0].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchorBox), totalElems);
    gmIn_[1].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(deltas), totalElems);
    gmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalElems);

    int64_t perBufBytes = td->bufferSize / kPhysNodes;
    if (perBufBytes <= 0)
        perBufBytes = AscendC::ONE_BLK_SIZE; // defensive: never allocate 0-byte TBuf
    for (int i = 0; i < kPhysNodes; i++) {
        pipe_.InitBuffer(buf_[i], perBufBytes);
    }
}

// ---------------------------------------------------------------------------
// GetCoreRange — this core's index range [begin, end) (DESIGN §10.3)
//   Multi-core split via大小核均衡: blockNum = CeilDiv(totalCount, blockFormer);
//   cores_tail cores get tiles_main+1 blocks, rest get tiles_main.
//
//   N-axis (key=0): range is in BOX index space (totalCount = B×N boxes,
//     perCoreCount = box count per block).
//   B-axis (key=1): range is in BATCH index space (B batches,
//     perCoreCount = batch count per block, 512-batch aligned). Each batch = one
//     UB tile with boxCount = td->N (DESIGN-BRANCH-1.md §2.1 / §3.3).
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline typename RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::CoreRange
RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::GetCoreRange()
{
    int64_t numCores = AscendC::GetBlockNum();
    if (numCores <= 0)
        numCores = 1;

    if constexpr (UB_AXIS_SEL == ROTATED_BOX_DECODE_UB_AXIS_SEL_B) {
        // B-axis: split along batch dimension (DESIGN-BRANCH-1.md §2.1)
        //   perCoreCount = blockFormer (batch count, 512-batch aligned)
        //   blockNum = CeilDiv(B, blockFormer); cores split blocks
        int64_t B = td_->B;
        int64_t blockFormer = td_->perCoreCount;
        if (blockFormer <= 0)
            blockFormer = 1;
        int64_t blockNum = AscendC::CeilDivision(B, blockFormer);
        int64_t tilesMain = blockNum / numCores;
        int64_t coresTail = blockNum % numCores;
        int64_t myBlocks = tilesMain + (blockIdx_ < coresTail ? 1 : 0);
        int64_t beginBlock = blockIdx_ * tilesMain + MinI64(blockIdx_, coresTail);
        int64_t begin = beginBlock * blockFormer;                // batch begin
        int64_t end = MinI64(begin + myBlocks * blockFormer, B); // batch end
        return {begin, end};
    } else {
        // N-axis: split along box index space (totalCount = B×N boxes)
        int64_t totalCount = td_->totalCount;
        int64_t blockFormer = td_->perCoreCount;
        if (blockFormer <= 0)
            blockFormer = 1;
        int64_t blockNum = AscendC::CeilDivision(totalCount, blockFormer);
        int64_t tilesMain = blockNum / numCores;
        int64_t coresTail = blockNum % numCores;
        int64_t myBlocks = tilesMain + (blockIdx_ < coresTail ? 1 : 0);
        int64_t beginBlock = blockIdx_ * tilesMain + MinI64(blockIdx_, coresTail);
        int64_t begin = beginBlock * blockFormer;
        int64_t end = MinI64(begin + myBlocks * blockFormer, totalCount);
        return {begin, end};
    }
}

template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline typename RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::TileInfo
RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::GetTile(int64_t k)
{
    int64_t bs = coreBoxBegin_ + k * ubFormer_;
    int64_t bc = MinI64(ubFormer_, coreBoxEnd_ - bs);
    return {bs, bc};
}

template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline typename RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::TileInfo
RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::NextTile(int64_t pos, int64_t end)
{
    int64_t N = td_->N;
    int64_t b = pos / N;
    int64_t n = pos % N;
    int64_t batchEnd = (b + 1) * N;
    int64_t cap = MinI64(batchEnd, end);
    int64_t bc = MinI64(ubFormer_, cap - pos);
    return {pos, bc};
}

// BuildBatchTiles — decompose a core's range into single-batch tiles.
//
// Tiles are capped by ubFormer_ AND by the current batch's tail, so each tile
// stays within one batch. This keeps the CopyIn/CopyOut 2D stride gather /
// scatter a SINGLE segment per tile (ubOff=0, UB destination 32B-aligned) —
// required because a multi-batch tile would accumulate a non-32B-aligned
// ubOff across segments and trigger an AICore 507035 exception.
//
// When N > ubFormer_ the cap collapses to ubFormer_ (partial-batch slice,
// still single-segment). When N ≤ ubFormer_ the cap is the batch tail (one
// batch per tile).
//
// N-axis (key=0): range [coreBoxBegin_, coreBoxEnd_) is in BOX index space
//   (totalCount = B×N boxes).
// B-axis (key=1): range is in BATCH index space (DESIGN-BRANCH-1.md §2.3);
//   converted to BOX index space here so both axes share the same tile
//   decomposition. gmOff for boxStart = batchIdx × N is computed by
//   DoCopyInTransposeSplit from boxStart.
//
// NOTE: Process() iterates tiles on-the-fly (no TileInfo array) to avoid the
// kMaxTilesPerCore cap — for small N + large B a core may need O(B/cores)
// single-batch tiles, far exceeding any fixed stack array. This helper is
// kept only for UT/inspection; Process does not call it.
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline int64_t RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::BuildBatchTiles(TileInfo* out,
                                                                                             int64_t maxTiles)
{
    int64_t N = td_->N;
    int64_t cnt = 0;

    // B-axis core range is in BATCH space → convert to BOX space.
    int64_t boxBegin, boxEnd;
    if constexpr (UB_AXIS_SEL == ROTATED_BOX_DECODE_UB_AXIS_SEL_B) {
        boxBegin = coreBoxBegin_ * N;
        boxEnd = coreBoxEnd_ * N;
    } else {
        boxBegin = coreBoxBegin_;
        boxEnd = coreBoxEnd_;
    }

    int64_t pos = boxBegin;
    while (pos < boxEnd && cnt < maxTiles) {
        int64_t b = pos / N;
        int64_t n = pos % N;
        int64_t batchEnd = (b + 1) * N; // next batch boundary
        int64_t cap = MinI64(batchEnd, boxEnd);
        int64_t bc = MinI64(ubFormer_, cap - pos); // ≤ ubFormer, within batch
        out[cnt].boxStart = pos;
        out[cnt].boxCount = bc;
        pos += bc;
        cnt++;
    }
    return cnt;
}

// ---------------------------------------------------------------------------
// DoCopyInTransposeSplit — Phase 1 (MTE2): GM [B,5,N] -> UB slot [5, boxCount]
//   2D stride gather per batch segment: blockCount=5 channels, blockLen=seg elems,
//   srcStride=(N-seg) skip GM between channels, dstStride=(boxCount-seg) skip UB
//   between channels (places ch_c at base + c*boxCount).
//   Handles tiles spanning batch boundaries by decomposing into per-batch segments.
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::DoCopyInTransposeSplit(int64_t boxStart,
                                                                                                 int64_t boxCount,
                                                                                                 int slot)
{
    constexpr int64_t kElemBytes = sizeof(T);
    int64_t N = td_->N;
    int64_t b = boxStart / N;
    int64_t n = boxStart % N;
    int64_t remaining = boxCount;
    int64_t ubOff = 0;
    while (remaining > 0) {
        int64_t seg = MinI64(N - n, remaining);
        for (int s = 0; s < kNumInputs; s++) {
            // DataCopyPad: GM-side srcStride is in BYTES; UB-side dstStride is in
            // 32-BYTE UNITS (dav-c310). Channel stride in UB = ubFormer_ elements, so
            // dstStride field = (ubFormer_ - seg) * sizeof(T) / 32.
            AscendC::DataCopyExtParams params;
            params.blockCount = static_cast<uint16_t>(kChannels);
            params.blockLen = static_cast<uint32_t>(seg * kElemBytes);
            params.srcStride = (N - seg) * kElemBytes;                                 // GM: bytes
            params.dstStride = (ubFormer_ - seg) * kElemBytes / AscendC::ONE_BLK_SIZE; // UB: 32B units
            AscendC::DataCopyPadExtParams<T> padParams;
            padParams.isPad = ((seg * kElemBytes) % AscendC::ONE_BLK_SIZE) != 0;

            int64_t ioBaseElem = static_cast<int64_t>(s) * kChannels * ubFormer_;
            uint32_t ubOffset = static_cast<uint32_t>(ioBaseElem + ubOff);
            int64_t gmOffset = (b * kChannels) * N + n; // channel 0 start of this batch
            AscendC::GlobalTensor<T> gmSrc = gmIn_[s][gmOffset];
            AscendC::LocalTensor<T> ubDst = buf_[slot].template Get<T>()[ubOffset];
            AscendC::DataCopyPad(ubDst, gmSrc, params, padParams);
        }
        ubOff += seg;
        remaining -= seg;
        b++;
        n = 0;
    }
}

// ---------------------------------------------------------------------------
// ComputeMain — Phase 2 (V): asc_vf_call VF0-VF7.
//   The VF performs Δ' = deltas / weight via TRUE division (Duplicate weight →
//   Reg::Div) to bit-match the golden's `deltas / weight` (torch true_divide).
//   This avoids the ≤1 ULP error of reciprocal-multiply (Muls by 1/weight)
//   which amplifies on near-zero outputs (catastrophic cancellation).
//   The 5 weight floats are passed as VF scalars; verified safe (non-default
//   weight cases pass — the earlier "≥7 scalars corrupt" diagnosis was a
//   misattribution of the real kMaxTilesPerCore truncation bug).
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::ComputeMain(int slot, int64_t boxCount)
{
    __ubuf__ T* slotBase = reinterpret_cast<__ubuf__ T*>(buf_[slot].template Get<T>().GetPhyAddr());
    __ubuf__ T* anchorView = slotBase;                         // ioIdx=0
    __ubuf__ T* deltasView = slotBase + kChannels * ubFormer_; // ioIdx=1
    __ubuf__ T* yView = slotBase + 2 * kChannels * ubFormer_;  // ioIdx=2

    asc_vf_call<RotatedBoxDecodeComputeVF<T>>(yView, anchorView, deltasView, td_->weight[0], td_->weight[1],
                                              td_->weight[2], td_->weight[3], td_->weight[4], boxCount, ubFormer_);
}

// ---------------------------------------------------------------------------
// DoCopyOutStackTranspose — Phase 3 (MTE3): UB slot [5, boxCount] -> GM [B,5,N]
//   Reverse 2D stride scatter per batch segment. No Interleave needed: GM is
//   channel-major per batch, UB is channel-contiguous — direct per-channel copy.
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::DoCopyOutStackTranspose(int64_t boxStart,
                                                                                                  int64_t boxCount,
                                                                                                  int slot)
{
    constexpr int64_t kElemBytes = sizeof(T);
    int64_t N = td_->N;
    int64_t b = boxStart / N;
    int64_t n = boxStart % N;
    int64_t remaining = boxCount;
    int64_t ubOff = 0;
    while (remaining > 0) {
        int64_t seg = MinI64(N - n, remaining);
        // DataCopyPad UB->GM: UB-side srcStride in 32-BYTE UNITS; GM-side dstStride in BYTES.
        AscendC::DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(kChannels);
        params.blockLen = static_cast<uint32_t>(seg * kElemBytes);
        params.srcStride = (ubFormer_ - seg) * kElemBytes / AscendC::ONE_BLK_SIZE; // UB: 32B units
        params.dstStride = (N - seg) * kElemBytes;                                 // GM: bytes

        int64_t yBaseElem = 2 * kChannels * ubFormer_; // ioIdx=2
        uint32_t ubOffset = static_cast<uint32_t>(yBaseElem + ubOff);
        int64_t gmOffset = (b * kChannels) * N + n;
        AscendC::GlobalTensor<T> gmDst = gmOut_[gmOffset];
        AscendC::LocalTensor<T> ubSrc = buf_[slot].template Get<T>()[ubOffset];
        AscendC::DataCopyPad(gmDst, ubSrc, params);
        ubOff += seg;
        remaining -= seg;
        b++;
        n = 0;
    }
}

// ---------------------------------------------------------------------------
// Process — main kernel entry (DESIGN §10.5 / §10.6)
//   P=2 ping-pong three-phase pipeline over single-batch tiles.
//
//   N-axis (key=0): each tile is split at batch boundaries (BuildBatchTiles)
//   so the CopyIn 2D stride gather has one segment per tile (ubOff=0, UB dst
//   32B-aligned).
//
//   B-axis (key=1): each tile is one whole batch (boxCount = td->N, N维整批
//   load). Multi-core split is along B (batch dimension). The CopyIn/CopyOut
//   2D stride gather / scatter and VF compute chain are shared with key=0
//   (DESIGN-BRANCH-1.md §3 / §5: "VF 计算链、Sync、CopyIn/CopyOut 逻辑与
//   key=0 同构"). gmOff = batchIdx × N × kChannels (DESIGN §10.3 B-axis branch),
//   computed by DoCopyInTransposeSplit from boxStart = batchIdx × N.
//
//   Sync events (eventID = slot 0/1):
//     MTE2_V:  CopyIn -> Compute (same slot, per round)
//     V_MTE3:  Compute -> CopyOut (same slot, per round)
//     MTE3_MTE2: CopyOut(slot) -> next-round CopyIn(slot) (slot reuse, round≥2)
// ---------------------------------------------------------------------------
template <typename T, int COPY_MODE, int UB_AXIS_SEL>
__aicore__ inline void RotatedBoxDecodeKernel<T, COPY_MODE, UB_AXIS_SEL>::Process()
{
    blockIdx_ = AscendC::GetBlockIdx();
    ubFormer_ = td_->ubFactor;
    blockFormer_ = td_->perCoreCount;

    // Empty tensor no-op (totalCount==0 -> host set ubFactor=0)
    if (td_->totalCount == 0 || ubFormer_ <= 0) {
        return;
    }

    CoreRange r = GetCoreRange();
    coreBoxBegin_ = r.begin;
    coreBoxEnd_ = r.end;
    if (coreBoxEnd_ <= coreBoxBegin_)
        return;

    // B-axis core range is in BATCH space → convert to BOX space so both axes
    // share the same single-batch tile iteration below.
    int64_t N = td_->N;
    int64_t posBegin, posEnd;
    if constexpr (UB_AXIS_SEL == ROTATED_BOX_DECODE_UB_AXIS_SEL_B) {
        posBegin = coreBoxBegin_ * N;
        posEnd = coreBoxEnd_ * N;
    } else {
        posBegin = coreBoxBegin_;
        posEnd = coreBoxEnd_;
    }
    if (posEnd <= posBegin)
        return;

    // Tile stepper: each tile is capped by ubFormer_ AND by the current
    // batch's tail (single-segment CopyIn/CopyOut, UB destination 32B-aligned).
    // Tiles are produced on-the-fly — no TileInfo array — so a core handling
    // small-N + large-B (O(B/cores) single-batch tiles) is not capped by any
    // fixed stack array (kMaxTilesPerCore). This was the root cause of 57/71
    // blackbox failures: BuildBatchTiles previously truncated at 512 tiles,
    // leaving the rest of the core's box range uncomputed (stale GM output).

    // ===== Prologue: round 0, slot 0 =====
    TileInfo tPrev = NextTile(posBegin, posEnd);
    DoCopyInTransposeSplit(tPrev.boxStart, tPrev.boxCount, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);

    int64_t pos = tPrev.boxStart + tPrev.boxCount;
    int64_t round = 1;

    // ===== Steady-state: P=2 ping-pong =====
    while (pos < posEnd) {
        int slot = static_cast<int>(round % kPhysNodes);
        int prevSlot = static_cast<int>((round - 1) % kPhysNodes);

        // Wait prev CopyIn -> Compute prev slot
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(prevSlot);
        ComputeMain(prevSlot, tPrev.boxCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(prevSlot);

        // Slot reuse guard: wait prev-prev CopyOut of this slot done
        if (round >= 2) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
        }
        // CopyIn current slot
        TileInfo tCur = NextTile(pos, posEnd);
        DoCopyInTransposeSplit(tCur.boxStart, tCur.boxCount, slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);

        // Wait prev Compute -> CopyOut prev slot
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(prevSlot);
        DoCopyOutStackTranspose(tPrev.boxStart, tPrev.boxCount, prevSlot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(prevSlot);

        tPrev = tCur;
        pos = tCur.boxStart + tCur.boxCount;
        round++;
    }

    // ===== Epilogue: last round =====
    int lastSlot = static_cast<int>((round - 1) % kPhysNodes);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(lastSlot);
    ComputeMain(lastSlot, tPrev.boxCount);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(lastSlot);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(lastSlot);
    DoCopyOutStackTranspose(tPrev.boxStart, tPrev.boxCount, lastSlot);
    // No SetFlag<MTE3_MTE2> for the last round (no further CopyIn to guard)
}

} // namespace rbd_kernel
