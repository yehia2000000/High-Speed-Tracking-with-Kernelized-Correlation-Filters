#include <riscv_vector.h>
#include <stddef.h>
#include "K2_rvv.hpp"

namespace
{
/* Gradients of vl pixels of one output row. c points at in[y+1][x+1] (the pixel itself);
   its 4 neighbours are c[-1], c[+1] (same row) and c[-P], c[+P] (rows above and below). */
inline void grad(const float *c, ptrdiff_t P, size_t vl, vfloat32m4_t &gx, vfloat32m4_t &gy)
{
    vfloat32m4_t l = __riscv_vle32_v_f32m4(c - 1, vl), r = __riscv_vle32_v_f32m4(c + 1, vl);
    vfloat32m4_t u = __riscv_vle32_v_f32m4(c - P, vl), d = __riscv_vle32_v_f32m4(c + P, vl);
    gx = __riscv_vfsub_vv_f32m4(r, l, vl);                  // right - left
    gy = __riscv_vfsub_vv_f32m4(d, u, vl);                  // down - up
}

/* ref::k2_mag = sqrtf(fmaf(gy, gy, gx*gx)): gx*gx rounded, gy*gy fused, then sqrt. */
inline vfloat32m4_t magnitude(vfloat32m4_t gx, vfloat32m4_t gy, size_t vl)
{
    vfloat32m4_t s = __riscv_vfmul_vv_f32m4(gx, gx, vl);   // gx*gx           (rounded)
    s = __riscv_vfmacc_vv_f32m4(s, gy, gy, vl);            // gy*gy + s       (fused)
    return __riscv_vfsqrt_v_f32m4(s, vl);                  // IEEE sqrt, as fsqrt.s
}


inline vuint8m1_t bin_dot(vfloat32m4_t gx, vfloat32m4_t gy, size_t vl)
{
    using ref::K2_COS;
    using ref::K2_SIN;
    /* best = projection on k = 0 (cos 1, sin 0), the scalar's starting value */
    vfloat32m4_t best = __riscv_vfmul_vf_f32m4(gy, K2_SIN[0], vl);
    best = __riscv_vfmacc_vf_f32m4(best, K2_COS[0], gx, vl);
    vuint32m4_t b = __riscv_vmv_v_x_u32m4(0, vl);     // 32-bit like the floats: one vtype in the loop

    for (int k = 0; k < 9; k++)
    {
        vfloat32m4_t d = __riscv_vfmul_vf_f32m4(gy, K2_SIN[k], vl);   // SIN*gy      (rounded)
        d = __riscv_vfmacc_vf_f32m4(d, K2_COS[k], gx, vl);             // COS*gx + .  (fused)
        vfloat32m4_t nd = __riscv_vfneg_v_f32m4(d, vl);                // -d

        vbool8_t up = __riscv_vmflt_vv_f32m4_b8(best, d, vl);          // d > best
        vbool8_t dn = __riscv_vmflt_vv_f32m4_b8(best, nd, vl);         // -d > best (old best)
        dn = __riscv_vmandn_mm_b8(dn, up, vl);                         // "else if": not when up

        best = __riscv_vmerge_vvm_f32m4(best, d, up, vl);
        best = __riscv_vmerge_vvm_f32m4(best, nd, dn, vl);
        b = __riscv_vmerge_vxm_u32m4(b, (uint32_t)k, up, vl);
        b = __riscv_vmerge_vxm_u32m4(b, (uint32_t)(k + 9), dn, vl);
    }
    vuint16m2_t b16 = __riscv_vncvt_x_x_w_u16m2(b, vl);   // 32 -> 16 bit, once per strip
    return __riscv_vncvt_x_x_w_u8m1(b16, vl);             // 16 -> 8 bit
}



/* One method for the whole image; the method is a template argument, so the row loop
   has no branch on it. */
template <ref::K2Method M>
void k2_rows(const float *in, int T, float *mag, uint8_t *bin)
{
    const int P = T + 2;                                    // row stride of the input, in floats
    for (int y = 0; y < T; y++)
    {
        const float *row = in + (y + 1) * P + 1;            // in[y+1][1] = pixel (0, y)
        float *mrow = mag + y * T;                          // output row y
        uint8_t *brow = bin + y * T;

        for (int x = 0; x < T; )                            // strip-mine the row
        {
            const size_t vl = __riscv_vsetvl_e32m4(T - x);
            vfloat32m4_t gx, gy;
            grad(row + x, P, vl, gx, gy);
            __riscv_vse32_v_f32m4(mrow + x, magnitude(gx, gy, vl), vl);
            __riscv_vse8_v_u8m1(brow + x, bin_dot(gx, gy, vl), vl);
            x += (int)vl;
        }
    }
}
}  // namespace

void vec::k2_rvv(const float *in, int T, float *mag, uint8_t *bin, ref::K2Method method)
{
    switch (method)
    {
    case ref::K2_DOT:  k2_rows<ref::K2_DOT>(in, T, mag, bin);  break;
    default:
        /* K2_POLY and K2_ATAN2 have no vector bins here: the scalar computes them */
        ref::k2_scalar(in, T, mag, bin, method);
        break;
    }
}
