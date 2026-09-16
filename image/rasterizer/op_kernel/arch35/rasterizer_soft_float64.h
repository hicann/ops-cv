/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RASTERIZER_SOFT_FLOAT64_H
#define RASTERIZER_SOFT_FLOAT64_H

#include <cstdint>

namespace RasterizerKernel {
namespace SoftFloat64 {

constexpr uint64_t kSignMask = 0x8000000000000000ULL;
constexpr uint64_t kFracMask = 0x000fffffffffffffULL;
constexpr uint64_t kImplicitBit = 0x0010000000000000ULL;
constexpr uint64_t kInfBits = 0x7ff0000000000000ULL;
constexpr uint64_t kQuietNanBits = 0x7ff8000000000000ULL;
constexpr uint64_t kAbsMask = 0x7fffffffffffffffULL;
constexpr uint64_t kHalfBits = 0x3fe0000000000000ULL;
constexpr uint64_t kOneBits = 0x3ff0000000000000ULL;

struct Decoded {
    uint32_t sign;
    int32_t exponent;
    uint64_t significand;
    uint32_t isZero;
    uint32_t isInf;
    uint32_t isNan;
};

__simt_callee__ __attribute__((noinline)) inline uint64_t ShiftRightJam(uint64_t value, uint32_t distance)
{
    if (distance == 0U) {
        return value;
    }
    if (distance < 64U) {
        return (value >> distance) | ((value << (64U - distance)) != 0U);
    }
    return value != 0U;
}

__simt_callee__ __attribute__((noinline)) inline void Decode(uint64_t bits, Decoded& decoded)
{
    decoded.sign = static_cast<uint32_t>(bits >> 63U);
    const uint32_t rawExponent = static_cast<uint32_t>((bits >> 52U) & 0x7ffU);
    uint64_t fraction = bits & kFracMask;
    decoded.isZero = (rawExponent == 0U && fraction == 0U);
    decoded.isInf = (rawExponent == 0x7ffU && fraction == 0U);
    decoded.isNan = (rawExponent == 0x7ffU && fraction != 0U);
    decoded.exponent = 0;
    decoded.significand = 0U;
    if (decoded.isZero || decoded.isInf || decoded.isNan) {
        return;
    }
    if (rawExponent != 0U) {
        decoded.exponent = static_cast<int32_t>(rawExponent) - 1023;
        decoded.significand = kImplicitBit | fraction;
        return;
    }
    int32_t shift = 0;
    bool normalizing = true;
    for (uint32_t i = 0U; i < 52U; ++i) {
        if (normalizing && (fraction & kImplicitBit) == 0U) {
            fraction <<= 1U;
            ++shift;
        } else {
            normalizing = false;
        }
    }
    decoded.exponent = -1022 - shift;
    decoded.significand = fraction;
}

__simt_callee__ __attribute__((noinline)) inline uint64_t Pack(uint32_t sign, int32_t exponent, uint64_t extended)
{
    if (extended == 0U) {
        return static_cast<uint64_t>(sign) << 63U;
    }
    bool normalizing = true;
    for (uint32_t i = 0U; i < 56U; ++i) {
        if (normalizing && (extended & (1ULL << 55U)) == 0U) {
            extended <<= 1U;
            --exponent;
        } else {
            normalizing = false;
        }
    }
    if (exponent < -1022) {
        const uint32_t shift = static_cast<uint32_t>(-1022 - exponent);
        extended = ShiftRightJam(extended, shift);
        exponent = -1022;
    }
    const uint64_t roundBits = extended & 7U;
    uint64_t significand = extended >> 3U;
    if (roundBits > 4U || (roundBits == 4U && (significand & 1U) != 0U)) {
        ++significand;
    }
    if (significand == (1ULL << 53U)) {
        significand >>= 1U;
        ++exponent;
    }
    if (exponent > 1023) {
        return (static_cast<uint64_t>(sign) << 63U) | kInfBits;
    }
    if (exponent == -1022 && significand < kImplicitBit) {
        return (static_cast<uint64_t>(sign) << 63U) | significand;
    }
    if (exponent < -1022 || significand == 0U) {
        return static_cast<uint64_t>(sign) << 63U;
    }
    return (static_cast<uint64_t>(sign) << 63U) | (static_cast<uint64_t>(exponent + 1023) << 52U) |
           (significand & kFracMask);
}

__simt_callee__ __attribute__((noinline)) inline uint64_t Add(uint64_t lhsBits, uint64_t rhsBits)
{
    Decoded lhs;
    Decoded rhs;
    Decode(lhsBits, lhs);
    Decode(rhsBits, rhs);
    if (lhs.isNan || rhs.isNan) {
        return kQuietNanBits;
    }
    if (lhs.isInf || rhs.isInf) {
        if (lhs.isInf && rhs.isInf && lhs.sign != rhs.sign) {
            return kQuietNanBits;
        }
        return lhs.isInf ? lhsBits : rhsBits;
    }
    if (lhs.isZero) {
        return rhsBits;
    }
    if (rhs.isZero) {
        return lhsBits;
    }
    if (lhs.exponent < rhs.exponent || (lhs.exponent == rhs.exponent && lhs.significand < rhs.significand)) {
        Decoded temp = lhs;
        lhs = rhs;
        rhs = temp;
    }
    uint64_t lhsExtended = lhs.significand << 3U;
    const uint64_t rhsExtended = ShiftRightJam(rhs.significand << 3U,
                                               static_cast<uint32_t>(lhs.exponent - rhs.exponent));
    int32_t exponent = lhs.exponent;
    uint64_t result;
    const uint32_t sign = lhs.sign;
    if (lhs.sign == rhs.sign) {
        result = lhsExtended + rhsExtended;
        if ((result & (1ULL << 56U)) != 0U) {
            result = ShiftRightJam(result, 1U);
            ++exponent;
        }
    } else {
        result = lhsExtended - rhsExtended;
        if (result == 0U) {
            return 0U;
        }
        bool normalizing = true;
        for (uint32_t i = 0U; i < 56U; ++i) {
            if (normalizing && (result & (1ULL << 55U)) == 0U) {
                result <<= 1U;
                --exponent;
            } else {
                normalizing = false;
            }
        }
    }
    return Pack(sign, exponent, result);
}

__simt_callee__ __attribute__((noinline)) inline uint64_t Sub(uint64_t lhsBits, uint64_t rhsBits)
{
    return Add(lhsBits, rhsBits ^ kSignMask);
}

__simt_callee__ __attribute__((noinline)) inline void Mul128(uint64_t lhs, uint64_t rhs, uint64_t& high, uint64_t& low)
{
    const uint64_t lhsLow = static_cast<uint32_t>(lhs);
    const uint64_t lhsHigh = lhs >> 32U;
    const uint64_t rhsLow = static_cast<uint32_t>(rhs);
    const uint64_t rhsHigh = rhs >> 32U;
    const uint64_t product0 = lhsLow * rhsLow;
    const uint64_t product1 = lhsLow * rhsHigh;
    const uint64_t product2 = lhsHigh * rhsLow;
    const uint64_t product3 = lhsHigh * rhsHigh;
    const uint64_t middle = (product0 >> 32U) + static_cast<uint32_t>(product1) + static_cast<uint32_t>(product2);
    high = product3 + (product1 >> 32U) + (product2 >> 32U) + (middle >> 32U);
    low = (middle << 32U) | static_cast<uint32_t>(product0);
}

__simt_callee__ __attribute__((noinline)) inline uint64_t ShiftRightJam128(uint64_t high, uint64_t low,
                                                                           uint32_t distance)
{
    if (distance == 0U) {
        return low;
    }
    if (distance < 64U) {
        uint64_t result = (high << (64U - distance)) | (low >> distance);
        if ((low << (64U - distance)) != 0U) {
            result |= 1U;
        }
        return result;
    }
    if (distance == 64U) {
        return high | (low != 0U);
    }
    const uint32_t highDistance = distance - 64U;
    if (highDistance < 64U) {
        return (high >> highDistance) | (((high << (64U - highDistance)) | low) != 0U);
    }
    return (high | low) != 0U;
}

__simt_callee__ __attribute__((noinline)) inline uint64_t Mul(uint64_t lhsBits, uint64_t rhsBits)
{
    Decoded lhs;
    Decoded rhs;
    Decode(lhsBits, lhs);
    Decode(rhsBits, rhs);
    if (lhs.isNan || rhs.isNan) {
        return kQuietNanBits;
    }
    const uint32_t sign = lhs.sign ^ rhs.sign;
    if ((lhs.isInf && rhs.isZero) || (rhs.isInf && lhs.isZero)) {
        return kQuietNanBits;
    }
    if (lhs.isInf || rhs.isInf) {
        return (static_cast<uint64_t>(sign) << 63U) | kInfBits;
    }
    if (lhs.isZero || rhs.isZero) {
        return static_cast<uint64_t>(sign) << 63U;
    }
    uint64_t high;
    uint64_t low;
    Mul128(lhs.significand, rhs.significand, high, low);
    const uint32_t top = (high & (1ULL << 41U)) != 0U;
    const uint32_t shift = top ? 50U : 49U;
    return Pack(sign, lhs.exponent + rhs.exponent + static_cast<int32_t>(top), ShiftRightJam128(high, low, shift));
}

__simt_callee__ __attribute__((noinline)) inline uint64_t Div(uint64_t lhsBits, uint64_t rhsBits)
{
    Decoded lhs;
    Decoded rhs;
    Decode(lhsBits, lhs);
    Decode(rhsBits, rhs);
    if (lhs.isNan || rhs.isNan || (lhs.isZero && rhs.isZero) || (lhs.isInf && rhs.isInf)) {
        return kQuietNanBits;
    }
    const uint32_t sign = lhs.sign ^ rhs.sign;
    if (lhs.isInf || rhs.isZero) {
        return (static_cast<uint64_t>(sign) << 63U) | kInfBits;
    }
    if (lhs.isZero || rhs.isInf) {
        return static_cast<uint64_t>(sign) << 63U;
    }
    int32_t exponent = lhs.exponent - rhs.exponent;
    uint64_t remainder = lhs.significand;
    const uint64_t denominator = rhs.significand;
    if (remainder < denominator) {
        remainder <<= 1U;
        --exponent;
    }
    uint64_t quotient = 0U;
    for (int32_t bit = 55; bit >= 0; --bit) {
        if (remainder >= denominator) {
            remainder -= denominator;
            quotient |= 1ULL << static_cast<uint32_t>(bit);
        }
        if (bit != 0) {
            remainder <<= 1U;
        }
    }
    if (remainder != 0U) {
        quotient |= 1U;
    }
    return Pack(sign, exponent, quotient);
}

__simt_callee__ __attribute__((noinline)) inline uint64_t Float32ToDoubleBits(uint32_t bits)
{
    const uint32_t sign = bits >> 31U;
    const uint32_t rawExponent = (bits >> 23U) & 0xffU;
    uint32_t fraction = bits & 0x7fffffU;
    if (rawExponent == 0xffU) {
        return (static_cast<uint64_t>(sign) << 63U) | (fraction != 0U ? kQuietNanBits : kInfBits);
    }
    if (rawExponent == 0U && fraction == 0U) {
        return static_cast<uint64_t>(sign) << 63U;
    }
    if (rawExponent != 0U) {
        return (static_cast<uint64_t>(sign) << 63U) |
               (static_cast<uint64_t>(static_cast<int32_t>(rawExponent) - 127 + 1023) << 52U) |
               (static_cast<uint64_t>(fraction) << 29U);
    }
    int32_t shift = 0;
    bool normalizing = true;
    for (uint32_t i = 0U; i < 23U; ++i) {
        if (normalizing && (fraction & 0x800000U) == 0U) {
            fraction <<= 1U;
            ++shift;
        } else {
            normalizing = false;
        }
    }
    const int32_t exponent = -126 - shift;
    return (static_cast<uint64_t>(sign) << 63U) | (static_cast<uint64_t>(exponent + 1023) << 52U) |
           (static_cast<uint64_t>(fraction & 0x7fffffU) << 29U);
}

__simt_callee__ __attribute__((noinline)) inline uint64_t UInt32ToDoubleBits(uint32_t value)
{
    if (value == 0U) {
        return 0U;
    }
    int32_t mostSignificant = 0;
    for (uint32_t bit = 0U; bit < 32U; ++bit) {
        if (((value >> bit) & 1U) != 0U) {
            mostSignificant = static_cast<int32_t>(bit);
        }
    }
    return (static_cast<uint64_t>(mostSignificant + 1023) << 52U) |
           (static_cast<uint64_t>(value - (1U << static_cast<uint32_t>(mostSignificant)))
            << static_cast<uint32_t>(52 - mostSignificant));
}

__simt_callee__ __attribute__((noinline)) inline bool IsZero(uint64_t bits) { return (bits & kAbsMask) == 0U; }

__simt_callee__ __attribute__((noinline)) inline bool InUnitInterval(uint64_t bits)
{
    const uint64_t absolute = bits & kAbsMask;
    if (absolute == 0U) {
        return true;
    }
    if ((bits & kSignMask) != 0U) {
        return false;
    }
    return absolute <= kOneBits;
}

__simt_callee__ __attribute__((noinline)) inline uint64_t TriangleArea(uint64_t ax, uint64_t ay, uint64_t bx,
                                                                       uint64_t by, uint64_t cx, uint64_t cy)
{
    return Sub(Mul(Sub(bx, ax), Sub(cy, ay)), Mul(Sub(by, ay), Sub(cx, ax)));
}

} // namespace SoftFloat64
} // namespace RasterizerKernel

#endif // RASTERIZER_SOFT_FLOAT64_H
