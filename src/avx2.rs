use std::simd::{Select, StdFloat, cmp::SimdPartialOrd, f32x8, num::SimdFloat, u32x8};

/// 2^x for x <= 0. ~2e-7 relative error.
#[inline(always)]
pub fn exp2(x: f32x8) -> f32x8 {
    let lo = f32x8::splat(-126.0);
    let x = x.simd_lt(lo).select(lo, x);
    // Adding 1.5·2^23 rounds x to the nearest integer n, which lands in the low mantissa bits
    // of t. This is ~5x faster than floor() + a float->int cast.
    const MAGIC: f32 = 12582912.0;
    let t = x + f32x8::splat(MAGIC);
    let n = t - f32x8::splat(MAGIC);
    let f = x - n; // [-0.5, 0.5]
    // Taylor series of 2^f; truncation error ~1.2e-7 on this range.
    let mut p = f32x8::splat(1.540353e-4);
    p = p.mul_add(f, f32x8::splat(1.333356e-3));
    p = p.mul_add(f, f32x8::splat(9.618129e-3));
    p = p.mul_add(f, f32x8::splat(5.550411e-2));
    p = p.mul_add(f, f32x8::splat(2.402265e-1));
    p = p.mul_add(f, f32x8::splat(6.931472e-1));
    p = p.mul_add(f, f32x8::splat(1.0));
    // 2^n: exponent field n + 127 (wrapping u32 math handles negative n).
    let bits = (t.to_bits() - u32x8::splat(0x4B40_0000 - 127)) << u32x8::splat(23);
    p * f32x8::from_bits(bits)
}
