// Bare-metal libm subset (see riscv_math.hpp). Built only under RISCV_BAREMETAL; on
// qemu-user the hosted glibc libm is used instead.
//
// Accuracy (checked under qemu-riscv64 against glibc over 1-4 million inputs each):
//   sqrtf, sqrt, fmaf, fma, fabsf, fabs, fminf, fmaxf, floorf, rintf, rint: exact
//     (one RISC-V F/D instruction, or fcvt round trip)
//   expf  <= 1 ulp            Cephes: x = n ln2 + r, e^r polynomial, 2^n from the exponent bits
//   cos, sin <= 1 ulp         fdlibm kernels, |x| < 1e6 (two-part pi/2 reduction)
//   cosf, sinf                the double versions, rounded to float
//   atanf <= 2 ulp, atan2f <= 3 ulp   Cephes atanf
//
// RISCV_MATH_TEST renames every function to km_<name> so a hosted test can compare them
// with glibc in the same program.

#if defined(RISCV_BAREMETAL) || defined(RISCV_MATH_TEST)

#include <cstdint>

#ifdef RISCV_MATH_TEST
#define RM(name) km_##name
#else
#include "riscv_math.hpp"
#define RM(name) name
#endif

namespace {

inline uint32_t bits(float x) { union { float f; uint32_t u; } v; v.f = x; return v.u; }

inline float from_bits(uint32_t u) { union { float f; uint32_t u; } v; v.u = u; return v.f; }

inline float f_abs(float x) { float r; asm("fabs.s %0, %1" : "=f"(r) : "f"(x)); return r; }
inline float f_fma(float a, float b, float c) {
    float r; asm("fmadd.s %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
/* Keeps the sign of x on a zero result, as rint/floor must (-0.3 -> -0.0). */
inline float f_sign_of(float r, float x) { float o; asm("fsgnj.s %0, %1, %2" : "=f"(o) : "f"(r), "f"(x)); return o; }

/* rint with the current rounding mode (dyn), floor with round-down (rdn). |x| < 2^23 only. */
inline float f_rint_small(float x) {
    int32_t i; float r;
    asm volatile("fcvt.w.s %0, %1, dyn" : "=r"(i) : "f"(x));
    asm("fcvt.s.w %0, %1" : "=f"(r) : "r"(i));
    return f_sign_of(r, x);
}
inline float f_floor_small(float x) {
    int32_t i; float r;
    asm("fcvt.w.s %0, %1, rdn" : "=r"(i) : "f"(x));
    asm("fcvt.s.w %0, %1" : "=f"(r) : "r"(i));
    return f_sign_of(r, x);
}
inline double d_rint_small(double x) {          /* |x| < 2^52 */
    int64_t i; double r, o;
    asm volatile("fcvt.l.d %0, %1, dyn" : "=r"(i) : "f"(x));
    asm("fcvt.d.l %0, %1" : "=f"(r) : "r"(i));
    asm("fsgnj.d %0, %1, %2" : "=f"(o) : "f"(r), "f"(x));
    return o;
}

/* fdlibm __kernel_sin / __kernel_cos, |x| <= pi/4 */
inline double k_sin(double x) {
    const double S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03,
                 S3 = -1.98412698298579493134e-04, S4 = 2.75573137070700676789e-06,
                 S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
    double z = x * x;
    return x + x * z * (S1 + z * (S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))));
}

inline double k_cos(double x) {
    const double C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03,
                 C3 = 2.48015872894767294178e-05, C4 = -2.75573143513906633035e-07,
                 C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
    double z = x * x;
    double r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    double hz = 0.5 * z, w = 1.0 - hz;
    return w + (((1.0 - w) - hz) + z * r);      /* fdlibm's form: keeps 1 - z/2 exact */
}
/* x = q (pi/2) + r, |r| <= pi/4. Two-part pi/2: good for |x| < 1e6. Returns q mod 4. */
inline int reduce_pio2(double x, double *r) {
    const double TWO_OVER_PI = 6.36619772367581382433e-01;
    const double PIO2_1 = 1.57079632673412561417e+00, PIO2_1T = 6.07710050650619224932e-11;
    double q = d_rint_small(x * TWO_OVER_PI);
    *r = (x - q * PIO2_1) - q * PIO2_1T;
    return (int)(int64_t)q & 3;
}

/* Cephes atanf for x >= 0 */
inline float atan_pos(float x) {
    float y = 0.0f;
    if (x > 2.414213562373095f) { y = 1.5707963267948966f; x = -1.0f / x; }
    else if (x > 0.4142135623730950f) { y = 0.7853981633974483f; x = (x - 1.0f) / (x + 1.0f); }
    float z = x * x;
    return y + ((((8.05374449538e-2f * z - 1.38776856032e-1f) * z + 1.99777106478e-1f) * z
                 - 3.33329491539e-1f) * z * x + x);
}

}  /* namespace */

extern "C" {

float RM(sqrtf)(float x) { float r; asm("fsqrt.s %0, %1" : "=f"(r) : "f"(x)); return r; }
double RM(sqrt)(double x) { double r; asm("fsqrt.d %0, %1" : "=f"(r) : "f"(x)); return r; }
float RM(fmaf)(float a, float b, float c) { return f_fma(a, b, c); }
double RM(fma)(double a, double b, double c) {
    double r; asm("fmadd.d %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
float RM(fabsf)(float x) { return f_abs(x); }
double RM(fabs)(double x) { double r; asm("fabs.d %0, %1" : "=f"(r) : "f"(x)); return r; }
float RM(fminf)(float a, float b) { float r; asm("fmin.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }
float RM(fmaxf)(float a, float b) { float r; asm("fmax.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b)); return r; }

float RM(rintf)(float x) {
    if (!(f_abs(x) < 8388608.0f)) return x;      /* >= 2^23 is already an integer; inf, NaN */
    return f_rint_small(x);
}
double RM(rint)(double x) {
    double a; asm("fabs.d %0, %1" : "=f"(a) : "f"(x));
    if (!(a < 4503599627370496.0)) return x;     /* 2^52 */
    return d_rint_small(x);
}
float RM(floorf)(float x) {
    if (!(f_abs(x) < 8388608.0f)) return x;
    return f_floor_small(x);
}

float RM(expf)(float x) {
    if (x != x) return x;
    if (x > 88.72283935546875f) return from_bits(0x7f800000u);   /* +inf */
    if (x < -103.97208404541015625f) return 0.0f;
    float nf = f_rint_small(x * 1.44269504088896341f);           /* n = round(x / ln2) */
    float r = f_fma(-nf, 0.693359375f, x);                        /* r = x - n ln2 (ln2 = hi + lo) */
    r = f_fma(-nf, -2.12194440e-4f, r);
    float p = 1.9875691500e-4f;
    p = f_fma(p, r, 1.3981999507e-3f);
    p = f_fma(p, r, 8.3334519073e-3f);
    p = f_fma(p, r, 4.1665795894e-2f);
    p = f_fma(p, r, 1.6666665459e-1f);
    p = f_fma(p, r, 5.0000001201e-1f);
    float y = f_fma(p, r * r, r) + 1.0f;                          /* e^r */
    int n = (int)nf;
    if (n > 127) { y *= 2.0f; n -= 1; }                           /* 2^128 is not a float */
    if (n < -126) {                                               /* result below FLT_MIN */
        y *= from_bits((uint32_t)(n + 126 + 127) << 23);
        return y * from_bits(1u << 23);                           /* * 2^-126 */
    }
    return y * from_bits((uint32_t)(n + 127) << 23);              /* * 2^n */
}

double RM(sin)(double x) {
    if (x != x) return x;
    double r;
    switch (reduce_pio2(x, &r)) {
        case 0:  return k_sin(r);
        case 1:  return k_cos(r);
        case 2:  return -k_sin(r);
        default: return -k_cos(r);
    }
}
double RM(cos)(double x) {
    if (x != x) return x;
    double r;
    switch (reduce_pio2(x, &r)) {
        case 0:  return k_cos(r);
        case 1:  return -k_sin(r);
        case 2:  return -k_cos(r);
        default: return k_sin(r);
    }
}
float RM(cosf)(float x) { return (float)RM(cos)((double)x); }
float RM(sinf)(float x) { return (float)RM(sin)((double)x); }

float RM(atanf)(float x) {
    if (x != x) return x;
    return (bits(x) >> 31) ? -atan_pos(-x) : atan_pos(x);
}
float RM(atan2f)(float y, float x) {
    const float PI = 3.14159265358979323846f, PIO2 = 1.57079632679489661923f;
    if (x != x || y != y) return x + y;
    const bool ny = bits(y) >> 31, nx = bits(x) >> 31;
    if (y == 0.0f) return nx ? (ny ? -PI : PI) : y;               /* +-0 or +-pi */
    if (x == 0.0f) return ny ? -PIO2 : PIO2;
    float a = RM(atanf)(y / x);
    if (!nx) return a;
    return ny ? a - PI : a + PI;
}

}  /* extern "C" */

#endif
