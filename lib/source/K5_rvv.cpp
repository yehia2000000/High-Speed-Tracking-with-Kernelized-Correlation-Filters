#include <riscv_vector.h>
#include <stddef.h>
#include "K5_rvv.hpp"
#include "K5_fft_rvv.hpp"   // vec::fft2
#include "riscv_math.hpp"   // sqrtf, expf (k5_make_labels)

/* dst[i] = src[i],   i = 0 .. n-1 */
void vec::k5_copy(float *dst, const float *src, int n)
{
    for (int i = 0; i < n; )
    {
        const size_t vl = __riscv_vsetvl_e32m8(n - i);
        __riscv_vse32_v_f32m8(dst + i, __riscv_vle32_v_f32m8(src + i, vl), vl);
        i += (int)vl;
    }
}

/* dst[i] = 0,   i = 0 .. n-1 */
void vec::k5_zero(float *dst, int n)
{
    const vfloat32m8_t z = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());  // once
    for (int i = 0; i < n; )
    {
        const size_t vl = __riscv_vsetvl_e32m8(n - i);
        __riscv_vse32_v_f32m8(dst + i, z, vl);           // only the first vl lanes are written
        i += (int)vl;
    }
}

/* Detection response in the Fourier domain (paper Eq. 22), per frequency f:
       R^[f] = alpha^[f] (.) k^xz[f]
   with alpha^ = a + i b and k^xz = p + i q:
       Re R^ = a p - b q          rr = fma(-b, q, a*p)
       Im R^ = a q + b p          ri = fma( b, p, a*q) */
void vec::k5_detect_resp(const float *ar, const float *ai, const float *kr, const float *ki,
                         int N, float *rr, float *ri)
{
    for (int f = 0; f < N; )
    {
        const size_t vl = __riscv_vsetvl_e32m4(N - f);
        vfloat32m4_t a = __riscv_vle32_v_f32m4(ar + f, vl);
        vfloat32m4_t b = __riscv_vle32_v_f32m4(ai + f, vl);
        vfloat32m4_t p = __riscv_vle32_v_f32m4(kr + f, vl);
        vfloat32m4_t q = __riscv_vle32_v_f32m4(ki + f, vl);

        vfloat32m4_t re = __riscv_vfmul_vv_f32m4(a, p, vl);      // a*p          rounded
        re = __riscv_vfnmsac_vv_f32m4(re, b, q, vl);             // -(b*q) + re  fused
        vfloat32m4_t im = __riscv_vfmul_vv_f32m4(a, q, vl);      // a*q          rounded
        im = __riscv_vfmacc_vv_f32m4(im, b, p, vl);              // b*p + im     fused

        __riscv_vse32_v_f32m4(rr + f, re, vl);
        __riscv_vse32_v_f32m4(ri + f, im, vl);
        f += (int)vl;
    }
}

/* Model update (linear interpolation), i = 0 .. n-1:
       m[i] = (1 - eta) m[i] + eta nw[i]          m = fma(eta, nw, (1 - eta) * m)
   eta = 0 keeps m, eta = 1 gives nw. */
void vec::k5_update(float *m, const float *nw, int n, float eta)
{
    const float keep = 1.0f - eta;                        // same scalar rounding as ref
    for (int i = 0; i < n; )
    {
        const size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t x = __riscv_vle32_v_f32m8(m + i, vl);
        vfloat32m8_t y = __riscv_vle32_v_f32m8(nw + i, vl);
        x = __riscv_vfmul_vf_f32m8(x, keep, vl);          // keep*m          (rounded)
        x = __riscv_vfmacc_vf_f32m8(x, eta, y, vl);       // eta*nw + x      (fused)
        __riscv_vse32_v_f32m8(m + i, x, vl);
        i += (int)vl;
    }
}

/* Multi-channel cross-spectrum (the sum inside paper Eq. 31), per frequency f:
       S[f] = sum_{c=0}^{C-1} conj(X_c[f]) (.) Z_c[f]
   with X_c = a + i b and Z_c = p + i q:
       Re S = sum_c (a p + b q)
       Im S = sum_c (a q - b p)
   summed over c = 0, 1, ..., C-1 in this order (same rounding as ref). */
void vec::k5_spectral(const float *xr, const float *xi, const float *zr, const float *zi,
                      int C, int N, float *sr, float *si)
{
    for (int f = 0; f < N; )
    {
        const size_t vl = __riscv_vsetvl_e32m4(N - f);
        vfloat32m4_t re = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t im = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        const float *pa = xr + f, *pb = xi + f, *pp = zr + f, *pq = zi + f;   // channel 0
        for (int c = 0; c < C; c++)
        {
            vfloat32m4_t a = __riscv_vle32_v_f32m4(pa, vl);
            vfloat32m4_t b = __riscv_vle32_v_f32m4(pb, vl);
            vfloat32m4_t p = __riscv_vle32_v_f32m4(pp, vl);
            vfloat32m4_t q = __riscv_vle32_v_f32m4(pq, vl);
            re = __riscv_vfmacc_vv_f32m4(re, a, p, vl);       // re += a p
            re = __riscv_vfmacc_vv_f32m4(re, b, q, vl);       // re += b q
            im = __riscv_vfmacc_vv_f32m4(im, a, q, vl);       // im += a q
            im = __riscv_vfnmsac_vv_f32m4(im, b, p, vl);      // im -= b p
            pa += N; pb += N; pp += N; pq += N;               // same f, next channel
        }
        __riscv_vse32_v_f32m4(sr + f, re, vl);
        __riscv_vse32_v_f32m4(si + f, im, vl);
        f += (int)vl;
    }
}

/* Training in the Fourier domain (paper Eq. 17), per frequency f:
       alpha^[f] = y^[f] / (k^xx[f] + lambda)
   with w = k^xx + lambda = p + i q and y^ = a + i b:
       alpha^ = y^ conj(w) / |w|^2
       Re alpha^ = (a p + b q) / (p^2 + q^2)
       Im alpha^ = (b p - a q) / (p^2 + q^2)
   computed as inv = 1 / (p^2 + q^2), then each numerator * inv (as ref). */
void vec::k5_train_alpha(const float *yr, const float *yi, const float *kr, const float *ki,
                         float lambda, int N, float *ar, float *ai)
{
    for (int f = 0; f < N; )
    {
        const size_t vl = __riscv_vsetvl_e32m4(N - f);
        
        vfloat32m4_t p = __riscv_vfadd_vf_f32m4(__riscv_vle32_v_f32m4(kr + f, vl), lambda, vl); //(k^ + lambda)

        vfloat32m4_t q = __riscv_vle32_v_f32m4(ki + f, vl);
        vfloat32m4_t d = __riscv_vfmul_vv_f32m4(p, p, vl);           // p*p        (rounded)
        d = __riscv_vfmacc_vv_f32m4(d, q, q, vl);                    // + q*q      (fused)
        vfloat32m4_t inv = __riscv_vfrdiv_vf_f32m4(d, 1.0f, vl);     // 1.0f / d   

        vfloat32m4_t a = __riscv_vle32_v_f32m4(yr + f, vl);
        vfloat32m4_t b = __riscv_vle32_v_f32m4(yi + f, vl);
        vfloat32m4_t re = __riscv_vfmul_vv_f32m4(a, p, vl);          // a*p        (rounded)
        re = __riscv_vfmacc_vv_f32m4(re, b, q, vl);                  // + b*q      (fused)
        re = __riscv_vfmul_vv_f32m4(re, inv, vl);                    // * inv
        vfloat32m4_t im = __riscv_vfmul_vv_f32m4(b, p, vl);          // b*p        (rounded)
        im = __riscv_vfnmsac_vv_f32m4(im, a, q, vl);                 // - a*q      (fused)
        im = __riscv_vfmul_vv_f32m4(im, inv, vl);                    // * inv

        __riscv_vse32_v_f32m4(ar + f, re, vl);
        __riscv_vse32_v_f32m4(ai + f, im, vl);
        f += (int)vl;
    }
}

/* exp(x) on vl lanes (Cephes expf, as ref::k5_exp):
       exp(x) = 2^n * e^r
       n   = round(x / ln2),   r = x - n ln2,   |r| <= ln2 / 2,   ln2 = hi + lo
       e^r = 1 + r + r^2 P(r),   P = degree-5 polynomial (Horner)
       2^n = float with bits (n + 127) << 23
       exp(x) = 0 for x < K5_EXP_MIN (2^n would not be a normal float) */
static inline vfloat32m4_t exp_poly(vfloat32m4_t x, size_t vl)
{
    const vbool8_t under = __riscv_vmflt_vf_f32m4_b8(x, K5_EXP_MIN, vl);    // -> 0 at the end
    
    const vint32m4_t n = __riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(x, 1.44269504f, vl), vl);
    const vfloat32m4_t nf = __riscv_vfcvt_f_x_v_f32m4(n, vl);
    vfloat32m4_t r = __riscv_vfnmsac_vf_f32m4(x, 0.693359375f, nf, vl);     // x - nf*hi
    r = __riscv_vfnmsac_vf_f32m4(r, -2.12194440e-4f, nf, vl);               // r - nf*lo

    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(1.9875691500e-4f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.3981999507e-3f, vl), vl);  // p*r + c
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(8.3334519073e-3f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(4.1665795894e-2f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.6666665459e-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(5.0000001201e-1f, vl), vl);
    vfloat32m4_t y = __riscv_vfmacc_vv_f32m4(r, p, __riscv_vfmul_vv_f32m4(r, r, vl), vl); // p*r^2 + r
    y = __riscv_vfadd_vf_f32m4(y, 1.0f, vl);

    const vint32m4_t e = __riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(n, 127, vl), 23, vl);
    y = __riscv_vfmul_vv_f32m4(y, __riscv_vreinterpret_v_i32m4_f32m4(e), vl);  // y * 2^n
    return __riscv_vfmerge_vfm_f32m4(y, 0.0f, under, vl);                      // x < min: 0
}

/* Gaussian kernel of all cyclic shifts (paper Eq. 31), i = 0 .. N-1:
       d[i] = max(||x||^2 + ||z||^2 - 2 c[i], 0)        = ||z - shift_i(x)||^2
       k[i] = exp(-d[i] * inv_sd),   inv_sd = 1 / (sigma^2 * N * C)
   with c = F^-1(S) (k5_correlate) and xx = ||x||^2, zz = ||z||^2 (k5_norm).
   (The paper divides by sigma^2 only; the extra 1 / (N C) is KCFcpp's normalization.) */
void vec::k5_gauss(const float *c, int N, float xx, float zz, float inv_sd, int use_poly, float *k)
{
    if (!use_poly)                                   // libm expf: no vector version, stay scalar
    {
        ref::k5_gauss(c, N, xx, zz, inv_sd, use_poly, k);
        return;
    }
    const float s = xx + zz, neg_inv = -inv_sd;
    for (int i = 0; i < N; )
    {
        const size_t vl = __riscv_vsetvl_e32m4(N - i);
        vfloat32m4_t d = __riscv_vfmv_v_f_f32m4(s, vl);
        d = __riscv_vfmacc_vf_f32m4(d, -2.0f, __riscv_vle32_v_f32m4(c + i, vl), vl);  // -2c + s
        d = __riscv_vfmax_vf_f32m4(d, 0.0f, vl);
        const vfloat32m4_t t = __riscv_vfmul_vf_f32m4(d, neg_inv, vl);
        __riscv_vse32_v_f32m4(k + i, exp_poly(t, vl), vl);
        i += (int)vl;
    }
}

/* s = sum_{i=0}^{n-1} x[i]^2
   lane j sums x[j], x[j + VL], x[j + 2 VL], ...; then the VL partial sums are added in lane
   order, then the last n % VL squares: other rounding than ref (one serial sum). */
float vec::k5_sumsq(const float *x, int n)
{
    const int vlmax = (int)__riscv_vsetvlmax_e32m8();
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    int i = 0;
    for (; i + vlmax <= n; i += vlmax)                           // full strips: vl = VLMAX
    {
        const vfloat32m8_t v = __riscv_vle32_v_f32m8(x + i, vlmax);
        acc = __riscv_vfmacc_vv_f32m8(acc, v, v, vlmax);         // lane j += x[i + j]^2
    }
    vfloat32m1_t s = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    s = __riscv_vfredosum_vs_f32m8_f32m1(acc, s, vlmax);         // lane 0 + lane 1 + ...
    if (i < n)                                                   // last n - i < VLMAX
    {
        const size_t vl = __riscv_vsetvl_e32m8(n - i);           // = n - i exactly
        const vfloat32m8_t v = __riscv_vle32_v_f32m8(x + i, vl);
        s = __riscv_vfredosum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(v, v, vl), s, vl);
    }
    return __riscv_vfmv_f_s_f32m1_f32(s);
}

/* Squared norm of a multi-channel signal from its spectrum (Parseval):
       ||x||^2 = (1 / N) sum_{c=0}^{C-1} sum_{f=0}^{N-1} |X_c[f]|^2
               = (sumsq(re, C N) + sumsq(im, C N)) / N */
float vec::k5_norm(const float *re, const float *im, int C, int N)
{
    return (vec::k5_sumsq(re, C * N) + vec::k5_sumsq(im, C * N)) / (float)N;
}

/* Spectra of the C feature channels, c = 0 .. C-1:
       X_c = F(channel c of the features)     (imaginary part starts at 0)
   The argument F holds the features, planar. In place when xr == F (the layout step wrote
   the features into xr), as in the flow. */
void vec::k5_spectra(const float *F, int C, int rows, int cols, float *xr, float *xi)
{
    int N = rows * cols;
    for (int c = 0; c < C; c++)
    {
        float *re = xr + c * N, *im = xi + c * N;
        if (re != F + c * N) vec::k5_copy(re, F + c * N, N);   // skip it when K4 wrote into xr
        vec::k5_zero(im, N);
        vec::fft2(re, im, rows, cols, 0);
    }
}

/* Gaussian kernel correlation in the Fourier domain (paper Eq. 31):
       S     = sum_c conj(X_c) (.) Z_c                                      k5_spectral
       c     = Re F^-1(S)                                                   fft2, inverse
       k^xz  = F( exp(-max(||x||^2 + ||z||^2 - 2 c, 0) / (sigma^2 N C)) )   k5_gauss, fft2
   result in w->kr, w->ki. */
void vec::k5_correlate(const float *xr, const float *xi, float xx, const float *zr,
                       const float *zi, float zz, int rows, int cols, int C, float inv_sd,
                       int use_poly, K5Work *w)
{
    int N = rows * cols;
    vec::k5_spectral(xr, xi, zr, zi, C, N, w->sr, w->si);       // S = sum_c conj(X_c) Z_c
    vec::fft2(w->sr, w->si, rows, cols, 1);                     // c = IFFT(S); real part in sr
    vec::k5_gauss(w->sr, N, xx, zz, inv_sd, use_poly, w->kr);   // k, real, spatial
    vec::k5_zero(w->ki, N);
    vec::fft2(w->kr, w->ki, rows, cols, 0);                     // k^
}

/* Train on the template spectrum x^:
       xx      = ||x||^2
       k^xx    = kernel correlation of x with itself (Eq. 31)
       alpha^  = y^ / (k^xx + lambda)                               (Eq. 17)
       first frame:  model x^ = x^,  model alpha^ = alpha^
       later:        model = (1 - eta) model + eta new             (x^ and alpha^) */
void vec::k5_train(K5Model *m, const float *xr, const float *xi, int first, int use_poly, K5Work *w)
{
    int N = m->rows * m->cols, CN = m->ch * N;
    float xx = vec::k5_norm(xr, xi, m->ch, N);
    vec::k5_correlate(xr, xi, xx, xr, xi, xx, m->rows, m->cols, m->ch, m->inv_sd, use_poly, w);
    vec::k5_train_alpha(m->yr, m->yi, w->kr, w->ki, m->lambda, N, w->ar, w->ai);
    if (first)                                                  // nothing to blend
    {
        vec::k5_copy(m->xr, xr, CN);
        vec::k5_copy(m->xi, xi, CN);
        vec::k5_copy(m->ar, w->ar, N);
        vec::k5_copy(m->ai, w->ai, N);
    }
    else
    {
        vec::k5_update(m->xr, xr, CN, m->eta);
        vec::k5_update(m->xi, xi, CN, m->eta);
        vec::k5_update(m->ar, w->ar, N, m->eta);
        vec::k5_update(m->ai, w->ai, N, m->eta);
    }
}

/* Detect on the patch spectrum z^ with the model (x^, alpha^):
       xx    = ||x||^2,   zz = ||z||^2
       k^xz  = kernel correlation of x and z (Eq. 31)
       resp  = Re F^-1( alpha^ (.) k^xz )                           (Eq. 22)
   resp: N floats, spatial, peak at (0, 0) = no motion (top-left convention). */
void vec::k5_detect(const K5Model *m, const float *zr, const float *zi, int use_poly, float *resp,
                    K5Work *w)
{
    int N = m->rows * m->cols;
    float xx = vec::k5_norm(m->xr, m->xi, m->ch, N);
    float zz = vec::k5_norm(zr, zi, m->ch, N);
    vec::k5_correlate(m->xr, m->xi, xx, zr, zi, zz, m->rows, m->cols, m->ch, m->inv_sd,
                      use_poly, w);
    vec::k5_detect_resp(m->ar, m->ai, w->kr, w->ki, N, w->ar, w->ai);   // R^ = alpha^ k^
    vec::fft2(w->ar, w->ai, m->rows, m->cols, 1);                       // response
    vec::k5_copy(resp, w->ar, N);
}

/* Regression target, once at setup:
       s       = sqrt(rows * cols) / padding * out_sigma_factor        (in cells)
       y[i][j] = exp(-(di^2 + dj^2) / (2 s^2))
       di, dj  = cyclic distance of row i / column j to 0 (peak 1 at (0, 0), wrapping
                 to the other 3 corners)
       y^      = F(y)                                                   (forward, unscaled)
   The label loop stays scalar (expf, N values); the DFT is vec::fft2. */
void vec::k5_make_labels(int rows, int cols, float padding, float out_sigma_factor, float *yr,
                         float *yi)
{
    float s = sqrtf((float)(rows * cols)) / padding * out_sigma_factor;   // cells
    float mult = -0.5f / (s * s);
    for (int i = 0; i < rows; i++)
    {
        int di = (i <= rows / 2) ? i : i - rows;          // cyclic distance to row 0
        for (int j = 0; j < cols; j++)
        {
            int dj = (j <= cols / 2) ? j : j - cols;      // cyclic distance to col 0
            yr[i * cols + j] = expf(mult * (float)(di * di + dj * dj));
            yi[i * cols + j] = 0.0f;
        }
    }
    vec::fft2(yr, yi, rows, cols, 0);
}