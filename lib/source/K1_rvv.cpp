#include <riscv_vector.h>
#include "K1_rvv.hpp"

namespace vec
{

/* Vector version of ref::k1_coord for j = 0..P-1, with the same rounding at every step:
   std::fma -> vfmadd, the two ifs -> vfmax/vfmin, (int) -> vfcvt.rtz.                     */
void k1_rvv_table(float base, float s, int n, int P, uint32_t *i0, uint32_t *i1, float *fr)
{
    for (int j = 0; j < P; ) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(P - j));
        vuint32m1_t jv = __riscv_vid_v_u32m1(vl);                    /* 0, 1, 2, ...           */
        jv = __riscv_vadd_vx_u32m1(jv, (uint32_t)j, vl);             /* j, j+1, ...            */
        vfloat32m1_t x = __riscv_vfcvt_f_xu_v_f32m1(jv, vl);         /* (float)j               */
        vfloat32m1_t b = __riscv_vfmv_v_f_f32m1(base, vl);
        x = __riscv_vfmadd_vf_f32m1(x, s, b, vl);                    /* j*s + base, one rounding */
        x = __riscv_vfmax_vf_f32m1(x, 0.0f, vl);                     /* clamp low              */
        x = __riscv_vfmin_vf_f32m1(x, (float)(n - 1), vl);           /* clamp high             */
        vuint32m1_t xi = __riscv_vfcvt_rtz_xu_f_v_u32m1(x, vl);      /* truncate = floor, x >= 0 */
        vfloat32m1_t xf = __riscv_vfcvt_f_xu_v_f32m1(xi, vl);
        vfloat32m1_t f  = __riscv_vfsub_vv_f32m1(x, xf, vl);         /* fraction               */
        vuint32m1_t x1  = __riscv_vadd_vx_u32m1(xi, 1, vl);
        x1 = __riscv_vminu_vx_u32m1(x1, (uint32_t)(n - 1), vl);      /* right neighbour, inside the row */
        __riscv_vse32_v_u32m1(i0 + j, xi, vl);
        __riscv_vse32_v_u32m1(i1 + j, x1, vl);
        __riscv_vse32_v_f32m1(fr + j, f, vl);
        j += (int)vl;
    }
}

/* vl bytes from row[idx[i]], returned as floats. */
static inline vfloat32m1_t k1_gather(const uint8_t *row, vuint32m1_t idx, size_t vl)
{
    vuint8mf4_t b = __riscv_vluxei32_v_u8mf4(row, idx, vl);          /* indexed load (bytes)   */
    vuint32m1_t w = __riscv_vzext_vf4_u32m1(b, vl);                  /* u8 -> u32              */
    return __riscv_vfcvt_f_xu_v_f32m1(w, vl);                        /* u32 -> f32             */
}

void k1_rvv(const Image<uint8_t, ImageType::GRAY>& in, Roi r, int T, float *out,
            uint32_t *i0, uint32_t *i1, float *fx)
{
    K1Map m;
    ref::k1_map(r, T, m);                                            /* the scalar's own mapping */
    const int P = m.P;
    const int W = in.Width();
    const int H = in.Height();
    k1_rvv_table(m.x_base, m.sx, W, P, i0, i1, fx);                  /* once per call          */
    for (int jy = 0; jy < P; jy++) {
        int y0, y1; float fy;
        ref::k1_coord(m.y_base, m.sy, jy, H, y0, y1, fy);            /* scalar, once per row   */
        const uint8_t *r0 = in.GetPtr(0, y0);                        /* top source row         */
        const uint8_t *r1 = in.GetPtr(0, y1);                        /* bottom source row      */
        float *o = out + (size_t)jy * P;
        for (int j = 0; j < P; ) {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(P - j));
            vuint32m1_t a0 = __riscv_vle32_v_u32m1(i0 + j, vl);
            vuint32m1_t a1 = __riscv_vle32_v_u32m1(i1 + j, vl);
            vfloat32m1_t vfx = __riscv_vle32_v_f32m1(fx + j, vl);
            vfloat32m1_t a = k1_gather(r0, a0, vl);                  /* top-left               */
            vfloat32m1_t b = k1_gather(r0, a1, vl);                  /* top-right              */
            vfloat32m1_t c = k1_gather(r1, a0, vl);                  /* bottom-left            */
            vfloat32m1_t d = k1_gather(r1, a1, vl);                  /* bottom-right           */
            vfloat32m1_t dab = __riscv_vfsub_vv_f32m1(b, a, vl);
            vfloat32m1_t dcd = __riscv_vfsub_vv_f32m1(d, c, vl);
            vfloat32m1_t top = __riscv_vfmacc_vv_f32m1(a, vfx, dab, vl); /* a + fx*(b-a)       */
            vfloat32m1_t bot = __riscv_vfmacc_vv_f32m1(c, vfx, dcd, vl); /* c + fx*(d-c)       */
            vfloat32m1_t dtb = __riscv_vfsub_vv_f32m1(bot, top, vl);
            vfloat32m1_t v   = __riscv_vfmacc_vf_f32m1(top, fy, dtb, vl); /* top + fy*(bot-top) */
            v = __riscv_vfmul_vf_f32m1(v, 1.0f / 255.0f, vl);
            __riscv_vse32_v_f32m1(o + j, v, vl);
            j += (int)vl;
        }
    }
}

}  // namespace vec