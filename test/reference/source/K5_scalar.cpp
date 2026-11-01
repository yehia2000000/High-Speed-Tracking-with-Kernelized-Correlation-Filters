#include "riscv_math.hpp"   // <math.h> is not available on bare metal
#include <stdint.h>
#include "K5_scalar.hpp"


static void k5_copy(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

static void k5_zero(float *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = 0.0f;
}

static float tw_re[512], tw_im[512];      /* exp(-2 pi i k / n), k < n/2, for the current n */
static int   tw_n = 0;

static void twiddles(int n) {
    if (n == tw_n) return;
    for (int k = 0; k < n / 2; k++) {
        double a = -2.0 * 3.14159265358979323846 * k / n;
        tw_re[k] = (float)cos(a);
        tw_im[k] = (float)sin(a);
    }
    tw_n = n;
}

static int reverse_bits(int x, int num_bits) {
    int result = 0;
    for (int b = 0; b < num_bits; b++) {
        result = (result << 1) | (x & 1);  
        x >>= 1;
    }
    return result;
}

static void fft1(float *re, float *im, int n, int stride, int inverse) {
    twiddles(n);
    int num_bits = 0;
    while ((1 << num_bits) < n) num_bits++;

    for (int i = 0; i < n; i++) {
        int j = reverse_bits(i, num_bits);
        if (i < j) {
            float t;
            t = re[i * stride]; re[i * stride] = re[j * stride]; re[j * stride] = t;
            t = im[i * stride]; im[i * stride] = im[j * stride]; im[j * stride] = t;
        }
    }

    for (int size = 2; size <= n; size *= 2) {
        int half = size / 2, step = n / size;          /* twiddle k * step in the table of n */
        for (int start = 0; start < n; start += size) {
            for (int k = 0; k < half; k++) {
                float wr = tw_re[k * step];
                float wi = inverse ? -tw_im[k * step] : tw_im[k * step];
                int top = (start + k) * stride;
                int bot = (start + k + half) * stride;
                float tr = re[bot] * wr - im[bot] * wi;
                float ti = re[bot] * wi + im[bot] * wr;
                re[bot] = re[top] - tr;   im[bot] = im[top] - ti;
                re[top] = re[top] + tr;   im[top] = im[top] + ti;
            }
        }
    }
}

void ref::fft2(float *re, float *im, int rows, int cols, int inverse) {
    for (int r = 0; r < rows; r++) fft1(re + r * cols, im + r * cols, cols, 1, inverse);
    for (int c = 0; c < cols; c++) fft1(re + c, im + c, rows, cols, inverse);
    if (inverse) {
        float s = 1.0f / (float)(rows * cols);
        for (int i = 0; i < rows * cols; i++) { re[i] *= s; im[i] *= s; }
    }
}

void ref::k5_spectra(const float *F, int C, int rows, int cols, float *xr, float *xi) {
    int N = rows * cols;
    for (int c = 0; c < C; c++) {
        float *re = xr + c * N, *im = xi + c * N;
        if (re != F + c * N) k5_copy(re, F + c * N, N);   /* skip the copy when K4 wrote into xr */
        k5_zero(im, N);
        ref::fft2(re, im, rows, cols, 0);
    }
}

/* S[f] = sum over channels of conj(X_c[f]) * Z_c[f], for every frequency f. */
void ref::k5_spectral(const float *xr, const float *xi, const float *zr, const float *zi,
                 int C, int N, float *sr, float *si) {
    for (int f = 0; f < N; f++) {
        float re = 0.0f, im = 0.0f;
        for (int c = 0; c < C; c++) {
            float a = xr[c * N + f], b = xi[c * N + f];       /* X = a + ib, conjugated */
            float p = zr[c * N + f], q = zi[c * N + f];       /* Z = p + iq             */
            re = fmaf(a, p, re);                              /* re += a p              */
            re = fmaf(b, q, re);                              /* re += b q              */
            im = fmaf(a, q, im);                              /* im += a q              */
            im = fmaf(-b, p, im);                             /* im -= b p              */
        }
        sr[f] = re;
        si[f] = im;
    }
}

float ref::k5_sumsq(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s = fmaf(x[i], x[i], s);    /* in order: x0, x1, x2, ... */
    return s;
}

/* ||x||^2 from the spectrum (Parseval): sum of |X|^2 over all channels and frequencies / N. */
float ref::k5_norm(const float *re, const float *im, int C, int N) {
    return (ref::k5_sumsq(re, C * N) + ref::k5_sumsq(im, C * N)) / (float)N;
}

/* exp(x) for x <= 0 from vector-friendly steps (Cephes expf): x = n ln2 + r, exp(x) = 2^n e^r. */
float ref::k5_exp(float x) {
    if (x < K5_EXP_MIN) return 0.0f;                 /* 2^n would not fit a normal float */
    float nf = rintf(x * 1.44269504f);               /* n = round(x / ln2)               */
    float r  = fmaf(-nf, 0.693359375f, x);           /* r = x - n ln2, ln2 = hi + lo     */
    r = fmaf(-nf, -2.12194440e-4f, r);               /* |r| <= ln2 / 2                   */
    float p = 1.9875691500e-4f;                      /* e^r = 1 + r + r^2 P(r)           */
    p = fmaf(p, r, 1.3981999507e-3f);
    p = fmaf(p, r, 8.3334519073e-3f);
    p = fmaf(p, r, 4.1665795894e-2f);
    p = fmaf(p, r, 1.6666665459e-1f);
    p = fmaf(p, r, 5.0000001201e-1f);
    float y = fmaf(p, r * r, r) + 1.0f;
    union { int32_t i; float f; } s;                 /* bit cast without memcpy          */
    s.i = ((int32_t)nf + 127) << 23;                 /* 2^n: exponent field, mantissa 0  */
    return y * s.f;
}

/* k[i] = exp(-max(xx + zz - 2 c[i], 0) * inv_sd): the Gaussian of ||z - shift_i(x)||^2. */
void ref::k5_gauss(const float *c, int N, float xx, float zz, float inv_sd, int use_poly, float *k) {
    float s = xx + zz;
    for (int i = 0; i < N; i++) {
        float d = fmaf(-2.0f, c[i], s);              /* squared distance at shift i      */
        d = fmaxf(d, 0.0f);                          /* rounding can make it < 0         */
        float t = -d * inv_sd;
        k[i] = use_poly ? ref::k5_exp(t) : expf(t);
    }
}

/* alpha^ = y^ / (k^ + lambda), complex division per frequency. */
void ref::k5_train_alpha(const float *yr, const float *yi, const float *kr, const float *ki,
                         float lambda, int N, float *ar, float *ai) {
    for (int f = 0; f < N; f++) {
        float p = kr[f] + lambda, q = ki[f];         /* w = k^ + lambda = p + iq         */
        float inv = 1.0f / fmaf(q, q, p * p);        /* 1 / |w|^2                        */
        float a = yr[f], b = yi[f];                  /* y^ = a + ib                      */
        ar[f] = fmaf(b, q, a * p) * inv;             /* (a p + b q) / |w|^2              */
        ai[f] = fmaf(-a, q, b * p) * inv;            /* (b p - a q) / |w|^2              */
    }
}

/* R^ = alpha^ * k^, complex multiply per frequency (no conjugate). */
void ref::k5_detect_resp(const float *ar, const float *ai, const float *kr, const float *ki,
                    int N, float *rr, float *ri) {
    for (int f = 0; f < N; f++) {
        float a = ar[f], b = ai[f], p = kr[f], q = ki[f];
        rr[f] = fmaf(-b, q, a * p);                  /* a p - b q */
        ri[f] = fmaf(b, p, a * q);                   /* a q + b p */
    }
}

/* m = (1 - eta) m + eta new. eta = 0 keeps m exactly, eta = 1 gives new exactly. */
void ref::k5_update(float *m, const float *nw, int n, float eta) {
    float keep = 1.0f - eta;
    for (int i = 0; i < n; i++) m[i] = fmaf(eta, nw[i], keep * m[i]);
}

/* k^ (into w->kr, w->ki) for templates x (conjugated side) and z, given their norms. */
void ref::k5_scalar_correlate(const float *xr, const float *xi, float xx,
                         const float *zr, const float *zi, float zz,
                         int rows, int cols, int C, float inv_sd, int use_poly, K5Work *w) {
    int N = rows * cols;
    ref::k5_spectral(xr, xi, zr, zi, C, N, w->sr, w->si);     /* S = sum_c conj(X_c) Z_c       */
    fft2(w->sr, w->si, rows, cols, 1);                    /* c = IFFT(S); real part in sr  */
    ref::k5_gauss(w->sr, N, xx, zz, inv_sd, use_poly, w->kr);  /* k, real, spatial              */
    k5_zero(w->ki, N);
    fft2(w->kr, w->ki, rows, cols, 0);                    /* k^                            */
}

/* Train on template spectrum x^. First frame: the model is x^ and alpha^. Later: blend. */
void ref::k5_scalar_train(K5Model *m, const float *xr, const float *xi, int first, int use_poly,
                     K5Work *w) {
    int N = m->rows * m->cols, CN = m->ch * N;
    float xx = ref::k5_norm(xr, xi, m->ch, N);
    ref::k5_scalar_correlate(xr, xi, xx, xr, xi, xx, m->rows, m->cols, m->ch, m->inv_sd,
                        use_poly, w);                                  /* k^xx            */
    ref::k5_train_alpha(m->yr, m->yi, w->kr, w->ki, m->lambda, N, w->ar, w->ai);
    if (first) {                                                       /* nothing to blend */
        k5_copy(m->xr, xr, CN);
        k5_copy(m->xi, xi, CN);
        k5_copy(m->ar, w->ar, N);
        k5_copy(m->ai, w->ai, N);
    } else {
        ref::k5_update(m->xr, xr, CN, m->eta);
        ref::k5_update(m->xi, xi, CN, m->eta);
        ref::k5_update(m->ar, w->ar, N, m->eta);
        ref::k5_update(m->ai, w->ai, N, m->eta);
    }
}

/* Detect: response map (N floats, spatial) of the model at the patch spectrum z^. */
void ref::k5_scalar_detect(const K5Model *m, const float *zr, const float *zi, int use_poly,
                      float *resp, K5Work *w) {
    int N = m->rows * m->cols;
    float xx = ref::k5_norm(m->xr, m->xi, m->ch, N);
    float zz = ref::k5_norm(zr, zi, m->ch, N);
    ref::k5_scalar_correlate(m->xr, m->xi, xx, zr, zi, zz, m->rows, m->cols, m->ch, m->inv_sd,
                        use_poly, w);                                  /* k^xz            */
    ref::k5_detect_resp(m->ar, m->ai, w->kr, w->ki, N, w->ar, w->ai);       /* R^ = alpha^ k^  */
    ref::fft2(w->ar, w->ai, m->rows, m->cols, 1);                           /* response        */
    k5_copy(resp, w->ar, N);
}
/* y^ for a rows x cols grid: Gaussian label with peak 1 at (0, 0), wrapping to the other
   3 corners (top-left convention), then FFT, forward and unscaled. Writes N floats into
   yr and yi. Called once at init; the caller then points the model at these arrays. */
void ref::k5_make_labels(int rows, int cols, float padding, float out_sigma_factor,
                         float *yr, float *yi) {
    float s = sqrtf((float)(rows * cols)) / padding * out_sigma_factor;   /* cells */
    float mult = -0.5f / (s * s);
    for (int i = 0; i < rows; i++) {
        int di = (i <= rows / 2) ? i : i - rows;          /* cyclic distance to row 0 */
        for (int j = 0; j < cols; j++) {
            int dj = (j <= cols / 2) ? j : j - cols;      /* cyclic distance to col 0 */
            yr[i * cols + j] = expf(mult * (float)(di * di + dj * dj));
            yi[i * cols + j] = 0.0f;
        }
    }
    ref::fft2(yr, yi, rows, cols, 0);
}