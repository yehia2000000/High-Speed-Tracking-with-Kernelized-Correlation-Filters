#include <riscv_vector.h>
#include <stddef.h>          // size_t
#include "riscv_math.hpp"   // cos, sin (no <math.h> on bare metal)
#include "K5_fft_rvv.hpp"

namespace
{
float tw_re[512], tw_im[512];      // exp(-2 pi i k / n), k < n/2: the same table as ref::fft2
int   tw_n = 0;

void twiddles(int n)
{
    if (n == tw_n) return;
    for (int k = 0; k < n / 2; k++)
    {
        double a = -2.0 * 3.14159265358979323846 * k / n;
        tw_re[k] = (float)cos(a);
        tw_im[k] = (float)sin(a);
    }
    tw_n = n;
}

int reverse_bits(int x, int num_bits)
{
    int result = 0;
    for (int b = 0; b < num_bits; b++)
    {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

short swap_i[512], swap_j[512];
int   swap_count = 0, swap_n = 0;

void swaps(int n)
{
    if (n == swap_n) return;
    int num_bits = 0;
    while ((1 << num_bits) < n) num_bits++;
    swap_count = 0;
    for (int i = 0; i < n; i++)
    {
        const int j = reverse_bits(i, num_bits);
        if (i < j) { swap_i[swap_count] = (short)i; swap_j[swap_count] = (short)j; swap_count++; }
    }
    swap_n = n;
}
//fft_lanes<true>(t_re, t_im, cols, rows, rows, 1, inverse);
// lstride = distance between lanes, in floats, not bytes
// istride = distance between elements of a lane, in floats, not bytes
template <bool Unit>
void fft_lanes(float *re, float *im, int len, int istride, int lanes, int lstride, int inverse)
{
    twiddles(len);
    swaps(len);
    const int bstride = lstride * (int)sizeof(float);               // bytes between lanes

    auto load = [&](const float *p, size_t vl) {
        if constexpr (Unit) return __riscv_vle32_v_f32m4(p, vl);
        else                return __riscv_vlse32_v_f32m4(p, bstride, vl);
    };
    auto store = [&](float *p, vfloat32m4_t v, size_t vl) {
        if constexpr (Unit) __riscv_vse32_v_f32m4(p, v, vl);
        else                __riscv_vsse32_v_f32m4(p, bstride, v, vl);
    };

    for (int l0 = 0; l0 < lanes; )                     // strip-mine the lanes (VLEN-agnostic)
    {
        const size_t vl = __riscv_vsetvl_e32m4(lanes - l0);
        float *r0 = re + l0 * lstride, *i0 = im + l0 * lstride;

        /* 1. bit reversal: only the pairs that move, from the table */
        for (int p = 0; p < swap_count; p++)
        {
            const int i = swap_i[p], j = swap_j[p];
            vfloat32m4_t a = load(r0 + i * istride, vl), b = load(r0 + j * istride, vl);
            store(r0 + i * istride, b, vl); store(r0 + j * istride, a, vl);
            a = load(i0 + i * istride, vl); b = load(i0 + j * istride, vl);
            store(i0 + i * istride, b, vl); store(i0 + j * istride, a, vl);
        }

        for (int size = 2; size <= len; size *= 2)
        {
            const int half = size / 2, step = len / size;
            for (int start = 0; start < len; start += size)
                for (int k = 0; k < half; k++)
                {
                    const float wr = tw_re[k * step];
                    const float wi = inverse ? -tw_im[k * step] : tw_im[k * step];
                    float *rt = r0 + (start + k) * istride,        *it = i0 + (start + k) * istride;
                    float *rb = r0 + (start + k + half) * istride, *ib = i0 + (start + k + half) * istride;

                    vfloat32m4_t br = load(rb, vl), bi = load(ib, vl);
        

                    vfloat32m4_t tr = __riscv_vfmul_vf_f32m4(bi, wi, vl);
                    vfloat32m4_t ti = __riscv_vfmul_vf_f32m4(bi, wr, vl);
                    tr = __riscv_vfmsac_vf_f32m4(tr, wr, br, vl);      // wr*br - bi*wi
                    ti = __riscv_vfmacc_vf_f32m4(ti, wi, br, vl);      // wi*br + bi*wr

                    vfloat32m4_t ar = load(rt, vl), ai = load(it, vl);
                    store(rb, __riscv_vfsub_vv_f32m4(ar, tr, vl), vl);  // bottom = top - t
                    store(ib, __riscv_vfsub_vv_f32m4(ai, ti, vl), vl);
                    store(rt, __riscv_vfadd_vv_f32m4(ar, tr, vl), vl);  // top = top + t
                    store(it, __riscv_vfadd_vv_f32m4(ai, ti, vl), vl);
                }
        }
        l0 += (int)vl;
    }
}

/* scratch for the transposed matrix: KCF needs 32 x 32 = 1024 */
constexpr int T_MAX = 4096;
float t_re[T_MAX], t_im[T_MAX];

void transpose_in(const float *xr, const float *xi, float *tr, float *ti, int rows, int cols)
{
    const int bstride = cols * (int)sizeof(float);
    for (int r0 = 0; r0 < rows; )
    {
        const size_t vl = __riscv_vsetvl_e32m4(rows - r0);
        for (int c = 0; c < cols; c++)
        {
            __riscv_vse32_v_f32m4(tr + c * rows + r0, __riscv_vlse32_v_f32m4(xr + r0 * cols + c, bstride, vl), vl);
            __riscv_vse32_v_f32m4(ti + c * rows + r0, __riscv_vlse32_v_f32m4(xi + r0 * cols + c, bstride, vl), vl);
        }
        r0 += (int)vl;
    }
}

/* the way back: row c of t (contiguous) is stored as column c of x (strided) */
void transpose_out(const float *tr, const float *ti, float *xr, float *xi, int rows, int cols)
{
    const int bstride = cols * (int)sizeof(float);
    for (int r0 = 0; r0 < rows; )
    {
        const size_t vl = __riscv_vsetvl_e32m4(rows - r0);
        for (int c = 0; c < cols; c++)
        {
            __riscv_vsse32_v_f32m4(xr + r0 * cols + c, bstride, __riscv_vle32_v_f32m4(tr + c * rows + r0, vl), vl);
            __riscv_vsse32_v_f32m4(xi + r0 * cols + c, bstride, __riscv_vle32_v_f32m4(ti + c * rows + r0, vl), vl);
        }
        r0 += (int)vl;
    }
}


/* the 1/N of the inverse, as one long vector */
void scale_inverse(float *re, float *im, int rows, int cols)
{
    const float s = 1.0f / (float)(rows * cols);
    const int n = rows * cols;
    for (int i = 0; i < n; )
    {
        const size_t vl = __riscv_vsetvl_e32m4(n - i);
        __riscv_vse32_v_f32m4(re + i, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(re + i, vl), s, vl), vl);
        __riscv_vse32_v_f32m4(im + i, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(im + i, vl), s, vl), vl);
        i += (int)vl;
    }
}
}  // namespace

void vec::fft2(float *re, float *im, int rows, int cols, int inverse)
{
    if (rows * cols <= T_MAX)
    {
        transpose_in(re, im, t_re, t_im, rows, cols);
        fft_lanes<true>(t_re, t_im, cols, rows, rows, 1, inverse);
        transpose_out(t_re, t_im, re, im, rows, cols);
    }
    else
        fft_lanes<false>(re, im, cols, 1, rows, cols, inverse);    // too big for t: strided
    fft_lanes<true>(re, im, rows, cols, cols, 1, inverse);         // columns: unit stride
    if (inverse) scale_inverse(re, im, rows, cols);
}