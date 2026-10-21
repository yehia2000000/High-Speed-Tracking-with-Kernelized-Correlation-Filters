#ifndef IMAGE_MEDIAN_RVV_SIMPLE_H_
#define IMAGE_MEDIAN_RVV_SIMPLE_H_

#include "image.hpp"
#include <riscv_vector.h>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace vec
{
    // ---- primitives -------------------------------------------------------------------

    // a gets the smaller value, b the larger, lane by lane.
    static inline void cas(vuint8m1_t& a, vuint8m1_t& b, size_t vl)
    {
        const vuint8m1_t lo = __riscv_vminu_vv_u8m1(a, b, vl);
        b = __riscv_vmaxu_vv_u8m1(a, b, vl);
        a = lo;
    }

    static inline vuint8m1_t min3(vuint8m1_t a, vuint8m1_t b, vuint8m1_t c, size_t vl)
    {
        return __riscv_vminu_vv_u8m1(__riscv_vminu_vv_u8m1(a, b, vl), c, vl);
    }

    static inline vuint8m1_t max3(vuint8m1_t a, vuint8m1_t b, vuint8m1_t c, size_t vl)
    {
        return __riscv_vmaxu_vv_u8m1(__riscv_vmaxu_vv_u8m1(a, b, vl), c, vl);
    }

    // median of three = max(min(a,b), min(max(a,b), c)): 4 ops instead of a full sort3 (6).
    static inline vuint8m1_t med3(vuint8m1_t a, vuint8m1_t b, vuint8m1_t c, size_t vl)
    {
        const vuint8m1_t lo = __riscv_vminu_vv_u8m1(a, b, vl);
        const vuint8m1_t hi = __riscv_vmaxu_vv_u8m1(a, b, vl);
        return __riscv_vmaxu_vv_u8m1(lo, __riscv_vminu_vv_u8m1(hi, c, vl), vl);
    }

    // ---- sorting networks (each proven on every 0/1 input) ----------------------------
    // After each call: first argument <= second <= ... <= last, in every lane.

    static inline void sort3(vuint8m1_t& a0, vuint8m1_t& a1, vuint8m1_t& a2, size_t vl)
    {
        cas(a0, a2, vl); cas(a0, a1, vl); cas(a1, a2, vl);
    }

    static inline void sort4(vuint8m1_t& a0, vuint8m1_t& a1, vuint8m1_t& a2, vuint8m1_t& a3, size_t vl)
    {
        cas(a0, a2, vl); cas(a1, a3, vl);
        cas(a0, a1, vl); cas(a2, a3, vl);
        cas(a1, a2, vl);
    }

    static inline void sort5(vuint8m1_t& a0, vuint8m1_t& a1, vuint8m1_t& a2, vuint8m1_t& a3,
                             vuint8m1_t& a4, size_t vl)
    {
        cas(a0, a3, vl); cas(a1, a4, vl);
        cas(a0, a2, vl); cas(a1, a3, vl);
        cas(a0, a1, vl); cas(a2, a4, vl);
        cas(a1, a2, vl); cas(a3, a4, vl);
        cas(a2, a3, vl);
    }

    static inline void sort6(vuint8m1_t& a0, vuint8m1_t& a1, vuint8m1_t& a2, vuint8m1_t& a3,
                             vuint8m1_t& a4, vuint8m1_t& a5, size_t vl)
    {
        cas(a0, a5, vl); cas(a1, a3, vl); cas(a2, a4, vl);
        cas(a1, a2, vl); cas(a3, a4, vl);
        cas(a0, a3, vl); cas(a2, a5, vl);
        cas(a0, a1, vl); cas(a2, a3, vl); cas(a4, a5, vl);
        cas(a1, a2, vl); cas(a3, a4, vl);
    }

    static inline void sort7(vuint8m1_t& a0, vuint8m1_t& a1, vuint8m1_t& a2, vuint8m1_t& a3,
                             vuint8m1_t& a4, vuint8m1_t& a5, vuint8m1_t& a6, size_t vl)
    {
        cas(a0, a6, vl); cas(a2, a3, vl); cas(a4, a5, vl);
        cas(a0, a2, vl); cas(a1, a4, vl); cas(a3, a6, vl);
        cas(a0, a1, vl); cas(a2, a5, vl); cas(a3, a4, vl);
        cas(a1, a2, vl); cas(a4, a6, vl);
        cas(a2, a3, vl); cas(a4, a5, vl);
        cas(a1, a2, vl); cas(a3, a4, vl); cas(a5, a6, vl);
    }

    // Batcher odd-even merge sort for 11 values (37 comparators).
    static inline void sort11(vuint8m1_t& a0, vuint8m1_t& a1, vuint8m1_t& a2, vuint8m1_t& a3,
                              vuint8m1_t& a4, vuint8m1_t& a5, vuint8m1_t& a6, vuint8m1_t& a7,
                              vuint8m1_t& a8, vuint8m1_t& a9, vuint8m1_t& a10, size_t vl)
    {
        cas(a0, a1, vl); cas(a2, a3, vl); cas(a0, a2, vl); cas(a1, a3, vl); cas(a1, a2, vl);
        cas(a4, a5, vl); cas(a6, a7, vl); cas(a4, a6, vl); cas(a5, a7, vl); cas(a5, a6, vl);
        cas(a0, a4, vl); cas(a2, a6, vl); cas(a2, a4, vl); cas(a1, a5, vl); cas(a3, a7, vl);
        cas(a3, a5, vl); cas(a1, a2, vl); cas(a3, a4, vl); cas(a5, a6, vl);
        cas(a8, a9, vl); cas(a8, a10, vl); cas(a9, a10, vl);
        cas(a0, a8, vl); cas(a4, a8, vl); cas(a2, a10, vl); cas(a6, a10, vl);
        cas(a2, a4, vl); cas(a6, a8, vl); cas(a1, a9, vl); cas(a5, a9, vl);
        cas(a3, a5, vl); cas(a7, a9, vl);
        cas(a1, a2, vl); cas(a3, a4, vl); cas(a5, a6, vl); cas(a7, a8, vl); cas(a9, a10, vl);
    }

    // ---- stage 1: column sort -------------------------------------------------------
    // src: row y of the padded image. dst: the K-row buffer. After the call,
    // dst[r * dst_stride + x] is the r-th smallest of the K pixels in column x.
    template <int K>
    static inline void sort_columns(const uint8_t* src, int src_stride,
                                    uint8_t* dst, int dst_stride, int width)
    {
        static_assert(K == 3 || K == 5 || K == 7, "supported kernel sizes: 3, 5, 7");
        for (int x = 0; x < width;)
        {
            const size_t vl = __riscv_vsetvl_e8m1(width - x);
#define LDC(r)    __riscv_vle8_v_u8m1(src + (r) * src_stride + x, vl)
#define STC(r, v) __riscv_vse8_v_u8m1(dst + (r) * dst_stride + x, (v), vl)
            if constexpr (K == 3)
            {
                vuint8m1_t v0 = LDC(0), v1 = LDC(1), v2 = LDC(2);
                sort3(v0, v1, v2, vl);
                STC(0, v0); STC(1, v1); STC(2, v2);
            }
            else if constexpr (K == 5)
            {
                vuint8m1_t v0 = LDC(0), v1 = LDC(1), v2 = LDC(2), v3 = LDC(3), v4 = LDC(4);
                sort5(v0, v1, v2, v3, v4, vl);
                STC(0, v0); STC(1, v1); STC(2, v2); STC(3, v3); STC(4, v4);
            }
            else
            {
                vuint8m1_t v0 = LDC(0), v1 = LDC(1), v2 = LDC(2), v3 = LDC(3);
                vuint8m1_t v4 = LDC(4), v5 = LDC(5), v6 = LDC(6);
                sort7(v0, v1, v2, v3, v4, v5, v6, vl);
                STC(0, v0); STC(1, v1); STC(2, v2); STC(3, v3);
                STC(4, v4); STC(5, v5); STC(6, v6);
            }
#undef LDC
#undef STC
            x += static_cast<int>(vl);
        }
    }


    static inline vuint8m1_t median3x3_from_sorted_columns(const uint8_t* cs, int stride, int x, size_t vl)
    {
#define LD(r, c) __riscv_vle8_v_u8m1(cs + (r) * stride + x + (c), vl)
        const vuint8m1_t m02 = max3(LD(0,0), LD(0,1), LD(0,2), vl);
        const vuint8m1_t m11 = med3(LD(1,0), LD(1,1), LD(1,2), vl);
        const vuint8m1_t m20 = min3(LD(2,0), LD(2,1), LD(2,2), vl);
#undef LD
        return med3(m20, m11, m02, vl);   // middle of the anti-diagonal = median of 9
    }

    // 5x5: candidates (13 cells) are the anti-diagonals s = r + c = 3, 4, 5.
    // After sorting those diagonals, 7 survivors remain; the median of 25 is their 4th smallest.
    static inline vuint8m1_t median5x5_from_sorted_columns(const uint8_t* cs, int stride, int x, size_t vl)
    {
#define LD(r, c) __riscv_vle8_v_u8m1(cs + (r) * stride + x + (c), vl)
        vuint8m1_t m00 = LD(0,0), m01 = LD(0,1), m02 = LD(0,2), m03 = LD(0,3), m04 = LD(0,4);
        vuint8m1_t m10 = LD(1,0), m11 = LD(1,1), m12 = LD(1,2), m13 = LD(1,3), m14 = LD(1,4);
        vuint8m1_t m20 = LD(2,0), m21 = LD(2,1), m22 = LD(2,2), m23 = LD(2,3), m24 = LD(2,4);
        vuint8m1_t m30 = LD(3,0), m31 = LD(3,1), m32 = LD(3,2), m33 = LD(3,3), m34 = LD(3,4);
        vuint8m1_t m40 = LD(4,0), m41 = LD(4,1), m42 = LD(4,2), m43 = LD(4,3), m44 = LD(4,4);
#undef LD

        // row sort: each rank-row, left (small) to right (large). Columns stay sorted.
        sort5(m00, m01, m02, m03, m04, vl);
        sort5(m10, m11, m12, m13, m14, vl);
        sort5(m20, m21, m22, m23, m24, vl);
        sort5(m30, m31, m32, m33, m34, vl);
        sort5(m40, m41, m42, m43, m44, vl);

        // anti-diagonal sort, only the diagonals that hold candidates (s = 3, 4, 5).
        // (the 7x7 version sorts all of them; the compiler drops the dead ones anyway)
        sort4(m30, m21, m12, m03, vl);                               // s = 3 -> keep top 2
        sort5(m40, m31, m22, m13, m04, vl);                          // s = 4 -> keep middle 3
        sort4(m41, m32, m23, m14, vl);                               // s = 5 -> keep bottom 2

        // selection: the median of all 25 is the 4th smallest of the 7 survivors.
        sort7(m12, m03, m31, m22, m13, m41, m32, vl);
        return m22;   // 4th argument of sort7 = 4th smallest after sorting
    }

    // 7x7 (unchanged).
    static inline vuint8m1_t median7x7_from_sorted_columns(const uint8_t* cs, int stride, int x, size_t vl)
    {
        // m<r><c> = rank r of column (x + c). Vector types cannot live in arrays, hence 49 names.
#define LD(r, c) __riscv_vle8_v_u8m1(cs + (r) * stride + x + (c), vl)
        vuint8m1_t m00 = LD(0,0), m01 = LD(0,1), m02 = LD(0,2), m03 = LD(0,3), m04 = LD(0,4), m05 = LD(0,5), m06 = LD(0,6);
        vuint8m1_t m10 = LD(1,0), m11 = LD(1,1), m12 = LD(1,2), m13 = LD(1,3), m14 = LD(1,4), m15 = LD(1,5), m16 = LD(1,6);
        vuint8m1_t m20 = LD(2,0), m21 = LD(2,1), m22 = LD(2,2), m23 = LD(2,3), m24 = LD(2,4), m25 = LD(2,5), m26 = LD(2,6);
        vuint8m1_t m30 = LD(3,0), m31 = LD(3,1), m32 = LD(3,2), m33 = LD(3,3), m34 = LD(3,4), m35 = LD(3,5), m36 = LD(3,6);
        vuint8m1_t m40 = LD(4,0), m41 = LD(4,1), m42 = LD(4,2), m43 = LD(4,3), m44 = LD(4,4), m45 = LD(4,5), m46 = LD(4,6);
        vuint8m1_t m50 = LD(5,0), m51 = LD(5,1), m52 = LD(5,2), m53 = LD(5,3), m54 = LD(5,4), m55 = LD(5,5), m56 = LD(5,6);
        vuint8m1_t m60 = LD(6,0), m61 = LD(6,1), m62 = LD(6,2), m63 = LD(6,3), m64 = LD(6,4), m65 = LD(6,5), m66 = LD(6,6);
#undef LD

        // row sort: each rank-row, left (small) to right (large). Columns stay sorted.
        sort7(m00, m01, m02, m03, m04, m05, m06, vl);
        sort7(m10, m11, m12, m13, m14, m15, m16, vl);
        sort7(m20, m21, m22, m23, m24, m25, m26, vl);
        sort7(m30, m31, m32, m33, m34, m35, m36, vl);
        sort7(m40, m41, m42, m43, m44, m45, m46, vl);
        sort7(m50, m51, m52, m53, m54, m55, m56, vl);
        sort7(m60, m61, m62, m63, m64, m65, m66, vl);

        cas  (m10, m01, vl);                                         // s = 1
        sort3(m20, m11, m02, vl);                                    // s = 2
        sort4(m30, m21, m12, m03, vl);                               // s = 3
        sort5(m40, m31, m22, m13, m04, vl);                          // s = 4
        sort6(m50, m41, m32, m23, m14, m05, vl);                     // s = 5
        sort7(m60, m51, m42, m33, m24, m15, m06, vl);                // s = 6
        sort6(m61, m52, m43, m34, m25, m16, vl);                     // s = 7
        sort5(m62, m53, m44, m35, m26, vl);                          // s = 8
        sort4(m63, m54, m45, m36, vl);                               // s = 9
        sort3(m64, m55, m46, vl);                                    // s = 10
        cas  (m65, m56, vl);                                         // s = 11

        // selection: the median of all 49 is the 6th smallest of the 11 survivors.
        sort11(m04, m05, m14, m23, m24, m33, m42, m43, m52, m61, m62, vl);
        return m33;   // 6th argument of sort11 = 6th smallest after sorting
    }

    template <int K>
    static inline vuint8m1_t median_from_sorted_columns(const uint8_t* cs, int stride, int x, size_t vl)
    {
        if constexpr (K == 3)      return median3x3_from_sorted_columns(cs, stride, x, vl);
        else if constexpr (K == 5) return median5x5_from_sorted_columns(cs, stride, x, vl);
        else                       return median7x7_from_sorted_columns(cs, stride, x, vl);
    }

    // ---- driver ---------------------------------------------------------------------

    // pad: input already bordered by R = K/2 on every side, (W + 2R) x (H + 2R). out: W x H.
    template <int K = 7>
    inline void median_rvv_rows(const Image<uint8_t, ImageType::GRAY>& pad,
                                Image<uint8_t, ImageType::GRAY>& out)
    {
        static_assert(K == 3 || K == 5 || K == 7, "supported kernel sizes: 3, 5, 7");
        const int W  = out.Width();
        const int H  = out.Height();
        const int Wp = pad.Width();          // GRAY: row stride in bytes == width

        Image<uint8_t, ImageType::GRAY> colbuf(Wp, K);   // K rows x Wp: rank r of every column
        uint8_t* cs = colbuf.GetPtr(0, 0);
 
        for (int y = 0; y < H; ++y)
        {
            sort_columns<K>(pad.GetPtr(0, y), Wp, cs, Wp, Wp);   // stage 1 (shared)

            uint8_t* dst = out.GetPtr(0, y);
            for (int x = 0; x < W;)                              // stage 2 (per output)
            {
                const size_t vl = __riscv_vsetvl_e8m1(W - x);
                __riscv_vse8_v_u8m1(dst + x, median_from_sorted_columns<K>(cs, Wp, x, vl), vl);
                x += static_cast<int>(vl);
            }
        }
    }

}
#endif