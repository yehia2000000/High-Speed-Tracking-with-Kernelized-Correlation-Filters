#include "image_hist_rvv.hpp"
#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>



void vec::image_hist_rvv(const uint8_t* src, size_t n, uint32_t* hist)
{
    const size_t max_vl = __riscv_vsetvlmax_e32m4();

    // Private histograms: one 256-bin table per vector lane
    uint32_t local_tables[max_vl * 256];

    
    size_t total = max_vl * 256;
    uint32_t* p = local_tables;
    while (total > 0) {
        size_t vl = __riscv_vsetvl_e32m4(total);
        vuint32m4_t zero = __riscv_vmv_v_x_u32m4(0, vl);
        __riscv_vse32_v_u32m4(p, zero, vl);
        p     += vl;
        total -= vl;
    }


    // ---- 2. Main histogram loop ----
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m1(n);

        vuint8m1_t  v_src  = __riscv_vle8_v_u8m1(src, vl);
        vuint32m4_t v_idx  = __riscv_vzext_vf4_u32m4(v_src, vl);
        vuint32m4_t v_lane = __riscv_vid_v_u32m4(vl);

        // offset = (lane * 256 + pixel) * 4
        vuint32m4_t v_off = __riscv_vmul_vx_u32m4(v_lane, 256, vl);
        v_off = __riscv_vadd_vv_u32m4(v_off, v_idx, vl);
        v_off = __riscv_vsll_vx_u32m4(v_off, 2, vl);

        vuint32m4_t v_cnt = __riscv_vluxei32_v_u32m4(local_tables, v_off, vl);
        v_cnt = __riscv_vadd_vx_u32m4(v_cnt, 1, vl);
        __riscv_vsuxei32_v_u32m4(local_tables, v_off, v_cnt, vl);

        src += vl;
        n   -= vl;
    }

    for (int bin = 0; bin < 256; ++bin) {
        uint32_t sum = 0;
        for (size_t lane = 0; lane < max_vl; ++lane) {
            sum += local_tables[lane * 256 + bin];
        }
        hist[bin] = sum;
    }
}
