#include <riscv_vector.h>
#include "K1_rvv.hpp"

namespace vec
{

/* Vector version of ref::k1_coord for j = 0..P-1, with the same rounding at every step:
   std::fma -> vfmadd, the two ifs -> vfmax/vfmin, (int) -> vfcvt.rtz.

   NOTE: i1 is no longer needed by the hot loop (the u32 gather fetches both
   bytes row[i] and row[i+1] in one go), so it is kept here only for
   interface compatibility and may be passed as nullptr.                */
void k1_rvv_table(float base, float s, int n, int P, uint32_t *i0, uint32_t *i1, float *fr)
{
    for (int j = 0; j < P; ) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(P - j));
        vuint32m4_t jv = __riscv_vid_v_u32m4(vl);
        jv = __riscv_vadd_vx_u32m4(jv, (uint32_t)j, vl);
        vfloat32m4_t x  = __riscv_vfcvt_f_xu_v_f32m4(jv, vl);
        vfloat32m4_t b  = __riscv_vfmv_v_f_f32m4(base, vl);
        x  = __riscv_vfmadd_vf_f32m4(x, s, b, vl);
        x  = __riscv_vfmax_vf_f32m4(x, 0.0f, vl);
        x  = __riscv_vfmin_vf_f32m4(x, (float)(n - 1), vl);
        vuint32m4_t xi  = __riscv_vfcvt_rtz_xu_f_v_u32m4(x, vl);
        vfloat32m4_t xf = __riscv_vfcvt_f_xu_v_f32m4(xi, vl);
        vfloat32m4_t f  = __riscv_vfsub_vv_f32m4(x, xf, vl);
        __riscv_vse32_v_u32m4(i0 + j, xi, vl);
        if (i1) {
            vuint32m4_t x1 = __riscv_vadd_vx_u32m4(xi, 1, vl);
            x1 = __riscv_vminu_vx_u32m4(x1, (uint32_t)(n - 1), vl);
            __riscv_vse32_v_u32m4(i1 + j, x1, vl);
        }
        __riscv_vse32_v_f32m4(fr + j, f, vl);
        j += (int)vl;
    }
}

/* vl pairs (row[idx[i]], row[idx[i]+1]) returned as two float vectors.
   One indexed u32 load replaces two indexed u8 loads.

   REQUIREMENT: reading 4 bytes at idx = n-1 touches row[n..n+2], so the
   caller must guarantee at least 3 bytes of padding after every row.
   With the standard k1_rvv_table indices (clamped to n-1) row padding of
   3 bytes is the intended usage.                                       */
static inline void k1_gather2(const uint8_t *row, vuint32m2_t idx, size_t vl,
                              vfloat32m2_t &lo, vfloat32m2_t &hi)
{
    vuint32m2_t w = __riscv_vluxei32_v_u32m2((const uint32_t *)row, idx, vl);
    vuint32m2_t a = __riscv_vand_vx_u32m2(w, 0xFF, vl);                 /* row[idx]   */
    vuint32m2_t b = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(w, 8, vl), 0xFF, vl); /* row[idx+1] */
    lo = __riscv_vfcvt_f_xu_v_f32m2(a, vl);
    hi = __riscv_vfcvt_f_xu_v_f32m2(b, vl);
}

void k1_rvv(const Image<uint8_t, ImageType::GRAY>& in, Roi r, int T, float *out,
            uint32_t *i0, uint32_t *i1, float *fx)
{
    K1Map m;
    ref::k1_map(r, T, m);
    const int P = m.P;
    const int W = in.Width();
    const int H = in.Height();

    // Precompute X table (once)
    k1_rvv_table(m.x_base, m.sx, W, P, i0, nullptr, fx);

    // Precompute Y table as well (instead of scalar k1_coord per row)
    // We reuse i1 for y0, and allocate temporary for y1 + fy if needed.
    // For simplicity and to keep the existing interface, we use stack arrays
    // when P is not too large. For production you may want to pass extra buffers.
    uint32_t iy0[P];          // or better: pass extra buffers from caller
    uint32_t iy1[P];
    float    fy_tbl[P];

    k1_rvv_table(m.y_base, m.sy, H, P, iy0, iy1, fy_tbl);

    const float inv255 = 1.0f / 255.0f;

    for (int jy = 0; jy < P; jy++) {
        const uint8_t *r0 = in.GetPtr(0, (int)iy0[jy]);
        const uint8_t *r1 = in.GetPtr(0, (int)iy1[jy]);
        const float     fy = fy_tbl[jy];

        float *o = out + (size_t)jy * P;

        for (int j = 0; j < P; ) {
            size_t vl = __riscv_vsetvl_e32m2((size_t)(P - j));

            vuint32m2_t  a0  = __riscv_vle32_v_u32m2(i0 + j, vl);
            vfloat32m2_t vfx = __riscv_vle32_v_f32m2(fx + j, vl);

            vfloat32m2_t a, b, c, d;
            k1_gather2(r0, a0, vl, a, b);   // top-left / top-right
            k1_gather2(r1, a0, vl, c, d);   // bottom-left / bottom-right

            vfloat32m2_t dab = __riscv_vfsub_vv_f32m2(b, a, vl);
            vfloat32m2_t dcd = __riscv_vfsub_vv_f32m2(d, c, vl);

            vfloat32m2_t top = __riscv_vfmacc_vv_f32m2(a, vfx, dab, vl); // a + fx*(b-a)
            vfloat32m2_t bot = __riscv_vfmacc_vv_f32m2(c, vfx, dcd, vl); // c + fx*(d-c)

            vfloat32m2_t dtb = __riscv_vfsub_vv_f32m2(bot, top, vl);
            vfloat32m2_t v   = __riscv_vfmacc_vf_f32m2(top, fy, dtb, vl); // top + fy*(bot-top)

            v = __riscv_vfmul_vf_f32m2(v, inv255, vl);

            __riscv_vse32_v_f32m2(o + j, v, vl);
            j += (int)vl;
        }
    }
}

}  // namespace vec