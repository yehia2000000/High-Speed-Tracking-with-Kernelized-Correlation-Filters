#ifndef IMAGE_MEDIAN_HIST_RVV_H_
#define IMAGE_MEDIAN_HIST_RVV_H_

#include "image.hpp"
#include <riscv_vector.h>
#include <cassert>
#include <cstddef>
#include <cstdint>


namespace vec
{
    namespace hist_detail
    {
        constexpr int BINS   = 256;
        constexpr int BANDS  = 16;   // coarse bins
        constexpr int BAND_W = 16;   // fine bins per coarse bin

        static inline void fill_zero(uint8_t* p, int n)
        {
            for (int i = 0; i < n;)
            {
                const size_t vl = __riscv_vsetvl_e8m1(n - i);
                __riscv_vse8_v_u8m1(p + i, __riscv_vmv_v_x_u8m1(0, vl), vl);
                i += static_cast<int>(vl);
            }
        }

            static inline void column_update(const uint8_t* row, int Wp, uint8_t* colF, uint8_t* colC,
                                            uint8_t delta)
            {
                for (int x = 0; x < Wp;)
                {
                    const size_t vl = __riscv_vsetvl_e8m1(Wp - x);   // e8m1 and e32m4 share VLMAX
                    const vuint32m4_t v   = __riscv_vzext_vf4_u32m4(__riscv_vle8_v_u8m1(row + x, vl), vl);
                    const vuint32m4_t col = __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(vl),static_cast<uint32_t>(x), vl);

                    // fine:   byte offset col * 256 + v
                    const vuint32m4_t fi = __riscv_vadd_vv_u32m4(__riscv_vsll_vx_u32m4(col, 8, vl), v, vl);
                    const vuint8m1_t  fc = __riscv_vluxei32_v_u8m1(colF, fi, vl);
                    __riscv_vsuxei32_v_u8m1(colF, fi, __riscv_vadd_vx_u8m1(fc, delta, vl), vl);

                    // coarse: byte offset col * 16 + (v >> 4)
                    const vuint32m4_t ci = __riscv_vadd_vv_u32m4(__riscv_vsll_vx_u32m4(col, 4, vl),
                                                                __riscv_vsrl_vx_u32m4(v, 4, vl), vl);
                    const vuint8m1_t  cc = __riscv_vluxei32_v_u8m1(colC, ci, vl);
                    __riscv_vsuxei32_v_u8m1(colC, ci, __riscv_vadd_vx_u8m1(cc, delta, vl), vl);

                    x += static_cast<int>(vl);
                }
            }

        // inclusive prefix sum over 16 u8 lanes: log2(16) = 4 slide-and-add steps
        static inline vuint8m1_t prefix16(vuint8m1_t v, vuint8m1_t zero, size_t vl)
        {
            v = __riscv_vadd_vv_u8m1(v, __riscv_vslideup_vx_u8m1(zero, v, 1, vl), vl);
            v = __riscv_vadd_vv_u8m1(v, __riscv_vslideup_vx_u8m1(zero, v, 2, vl), vl);
            v = __riscv_vadd_vv_u8m1(v, __riscv_vslideup_vx_u8m1(zero, v, 4, vl), vl);
            v = __riscv_vadd_vv_u8m1(v, __riscv_vslideup_vx_u8m1(zero, v, 8, vl), vl);
            return v;
        }

        // first lane whose prefix sum reaches target (always exists: the last lane holds the total)
        static inline int first_reaching(vuint8m1_t prefix, int target, size_t vl)
        {
            const vbool8_t ge = __riscv_vmsgeu_vx_u8m1_b8(prefix, static_cast<uint8_t>(target), vl);
            return static_cast<int>(__riscv_vfirst_m_b8(ge, vl));
        }
    }

    // pad: input already bordered by R = K/2 on every side. out: W x H.
    template <int K= 5>
    inline void median_hist_rvv_rows(const Image<uint8_t, ImageType::GRAY>& pad,
                                     Image<uint8_t, ImageType::GRAY>& out)
    {
        using namespace hist_detail;
        constexpr int N = K * K;
        constexpr int T = N / 2 + 1;                          // the median is the T-th smallest of N

        const int W  = out.Width();
        const int H  = out.Height();
        const int Wp = pad.Width();                           // GRAY: row stride in bytes == width

        Image<uint8_t, ImageType::GRAY> fine_buf(BINS, Wp);     // row x: fine histogram of column x
        Image<uint8_t, ImageType::GRAY> coarse_buf(BANDS, Wp);  // row x: coarse histogram of column x
        uint8_t* colF = fine_buf.GetPtr(0, 0);
        uint8_t* colC = coarse_buf.GetPtr(0, 0);
        fill_zero(colF, BINS * Wp);
        fill_zero(colC, BANDS * Wp);
        for (int r = 0; r < K; ++r)                           // column histograms for rows 0 .. K-1
            column_update(pad.GetPtr(0, r), Wp, colF, colC, 1);

        const size_t vl = __riscv_vsetvl_e8m1(BANDS);
        assert(vl == BANDS);                                  // V extension: VLEN >= 128
        const vuint8m1_t zero = __riscv_vmv_v_x_u8m1(0, vl);

        uint8_t kF[BINS];       // fine kernel histogram; band c is valid for window luc[c]
        int     luc[BANDS];     // last updated column of each band

        for (int y = 0; y < H; ++y)
        {
            if (y > 0)
            {
                column_update(pad.GetPtr(0, y - 1), Wp, colF, colC, 0xFF);   // row y-1 leaves
                column_update(pad.GetPtr(0, y + K - 1), Wp, colF, colC, 1);  // row y+K-1 enters
            }

            vuint8m1_t kC = zero;                             // coarse kernel histogram, window x = 0
            for (int j = 0; j < K; ++j)
                kC = __riscv_vadd_vv_u8m1(kC, __riscv_vle8_v_u8m1(colC + j * BANDS, vl), vl);
            for (int c = 0; c < BANDS; ++c)
                luc[c] = -K;                                  // every band stale -> rebuilt on first use

            uint8_t* dst = out.GetPtr(0, y);
            for (int x = 0; x < W; ++x)
            {
                // 1) coarse search
                const vuint8m1_t pc = prefix16(kC, zero, vl);
                const int c = first_reaching(pc, T, vl);
                const int below = c ? __riscv_vmv_x_s_u8m1_u8(__riscv_vslidedown_vx_u8m1(pc, c - 1, vl)) : 0;

                // 2) bring band c of the fine kernel histogram up to window x
                const uint8_t* band = colF + c * BAND_W;      // band c of column j is at band + j * BINS
                vuint8m1_t f;
                if (2 * (x - luc[c]) >= K)                    // far behind: rebuild from the K columns
                {
                    f = zero;
                    for (int j = x; j < x + K; ++j)
                        f = __riscv_vadd_vv_u8m1(f, __riscv_vle8_v_u8m1(band + j * BINS, vl), vl);
                }
                else                                          // close: replay the missed slides
                {
                    f = __riscv_vle8_v_u8m1(kF + c * BAND_W, vl);
                    for (int j = luc[c]; j < x; ++j)
                    {
                        f = __riscv_vadd_vv_u8m1(f, __riscv_vle8_v_u8m1(band + (j + K) * BINS, vl), vl);
                        f = __riscv_vsub_vv_u8m1(f, __riscv_vle8_v_u8m1(band + j * BINS, vl), vl);
                    }
                }
                __riscv_vse8_v_u8m1(kF + c * BAND_W, f, vl);
                luc[c] = x;

                // 3) fine search inside the band
                const int b = first_reaching(prefix16(f, zero, vl), T - below, vl);
                dst[x] = static_cast<uint8_t>(c * BAND_W + b);

                // 4) slide the coarse histogram to window x + 1
                if (x + 1 < W)
                {
                    kC = __riscv_vadd_vv_u8m1(kC, __riscv_vle8_v_u8m1(colC + (x + K) * BANDS, vl), vl);
                    kC = __riscv_vsub_vv_u8m1(kC, __riscv_vle8_v_u8m1(colC + x * BANDS, vl), vl);
                }
            }
        }
    }

}

#endif