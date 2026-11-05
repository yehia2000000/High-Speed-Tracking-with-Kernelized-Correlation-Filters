#include "gtest_lite.hpp"

#include "K5_scalar.hpp"
#include "K5_rvv.hpp"

/* vec::k5_* against ref::k5_*, bit for bit, one test per function (added as they are written).
   Sizes: below, at and above VL (16 on VLEN 128, 32 on VLEN 256 at e32 m4) and the KCF size
   N = 32 x 32 = 1024. */

namespace
{
constexpr int K5_MAX_N = 1024;
const int K5_SIZES[] = {1, 7, 16, 17, 33, 100, 1024};
const float K5_ETAS[] = {0.0f, 0.02f, 0.5f, 1.0f};     // 0.02 = KCF_DEFAULT_PARAMS

float g_a[K5_MAX_N], g_b[K5_MAX_N], g_c[K5_MAX_N], g_d[K5_MAX_N];
constexpr int K5_MAX_C = 31;                     // HOG channels in the project
const int K5_CHANNELS[] = {1, 2, 31};
float g_x[4][K5_MAX_C * K5_MAX_N];               // xr, xi, zr, zi for the spectral test
float g_out[K5_MAX_N + 1];                       // + 1 guard: nothing written past n
float g_ref0[K5_MAX_N], g_ref1[K5_MAX_N], g_vec0[K5_MAX_N], g_vec1[K5_MAX_N];
uint32_t g_seed = 2024;

float Random()                                   // in [-8, 8), with a fraction
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float)((g_seed >> 8) & 0xffff) / 4096.0f - 8.0f;
}

void Fill(float *x, int n)
{
    for (int i = 0; i < n; i++) x[i] = Random();
}

/* number of floats whose bits differ (no memcmp on bare metal) */
int Mismatches(const float *x, const float *y, int n)
{
    int bad = 0;
    for (int i = 0; i < n; i++)
    {
        union { float f; uint32_t u; } a, b;
        a.f = x[i];
        b.f = y[i];
        if (a.u != b.u) bad++;
    }
    return bad;
}
}  // namespace

TEST(K5Test, DetectRespMatchesReference)
{
    for (int N : K5_SIZES)
    {
        Fill(g_a, N); Fill(g_b, N); Fill(g_c, N); Fill(g_d, N);
        ref::k5_detect_resp(g_a, g_b, g_c, g_d, N, g_ref0, g_ref1);
        vec::k5_detect_resp(g_a, g_b, g_c, g_d, N, g_vec0, g_vec1);
        EXPECT_EQ(0, Mismatches(g_ref0, g_vec0, N));
        EXPECT_EQ(0, Mismatches(g_ref1, g_vec1, N));
    }
}

TEST(K5Test, UpdateMatchesReference)
{
    for (int N : K5_SIZES)
        for (float eta : K5_ETAS)
        {
            Fill(g_a, N); Fill(g_b, N);
            for (int i = 0; i < N; i++) g_ref0[i] = g_vec0[i] = g_a[i];    // same model m
            ref::k5_update(g_ref0, g_b, N, eta);
            vec::k5_update(g_vec0, g_b, N, eta);
            EXPECT_EQ(0, Mismatches(g_ref0, g_vec0, N));
        }
}

TEST(K5Test, CopyAndZero)
{
    for (int N : K5_SIZES)
    {
        Fill(g_a, N);
        for (int i = 0; i <= N; i++) g_out[i] = -1.0f;
        vec::k5_copy(g_out, g_a, N);
        EXPECT_EQ(0, Mismatches(g_out, g_a, N));
        EXPECT_EQ(-1.0f, g_out[N]);                                      // guard untouched

        for (int i = 0; i < N; i++) g_ref0[i] = 0.0f;
        for (int i = 0; i <= N; i++) g_out[i] = -1.0f;
        vec::k5_zero(g_out, N);
        EXPECT_EQ(0, Mismatches(g_out, g_ref0, N));                      // +0.0f bits
        EXPECT_EQ(-1.0f, g_out[N]);
    }
}

TEST(K5Test, SpectralMatchesReference)
{
    for (int C : K5_CHANNELS)
        for (int N : K5_SIZES)
        {
            for (int k = 0; k < 4; k++) Fill(g_x[k], C * N);
            ref::k5_spectral(g_x[0], g_x[1], g_x[2], g_x[3], C, N, g_ref0, g_ref1);
            vec::k5_spectral(g_x[0], g_x[1], g_x[2], g_x[3], C, N, g_vec0, g_vec1);
            EXPECT_EQ(0, Mismatches(g_ref0, g_vec0, N));
            EXPECT_EQ(0, Mismatches(g_ref1, g_vec1, N));
        }
}

TEST(K5Test, TrainAlphaMatchesReference)
{
    const float lambdas[] = {1e-4f, 0.5f};                           // 1e-4 = KCF_DEFAULT_PARAMS
    for (int N : K5_SIZES)
        for (float lambda : lambdas)
        {
            Fill(g_a, N); Fill(g_b, N); Fill(g_c, N); Fill(g_d, N);
            g_c[0] = 0.0f; g_d[0] = 0.0f;                            // k^ = 0: |w|^2 = lambda^2
            ref::k5_train_alpha(g_a, g_b, g_c, g_d, lambda, N, g_ref0, g_ref1);
            vec::k5_train_alpha(g_a, g_b, g_c, g_d, lambda, N, g_vec0, g_vec1);
            EXPECT_EQ(0, Mismatches(g_ref0, g_vec0, N));
            EXPECT_EQ(0, Mismatches(g_ref1, g_vec1, N));
        }
}

/* With xx = zz = 0 and inv_sd = 1, c = -u / 2 gives t = -max(u, 0) exactly, so u picks the
   exp argument: a sweep over [-1, 100) in steps of 0.1 (crosses K5_EXP_MIN = -87.33 and
   t = -0), the floats right at the cutoff, then random c with KCF-like values. */
TEST(K5Test, GaussMatchesReference)
{
    const int N = K5_MAX_N;
    for (int i = 0; i < N; i++) g_a[i] = -(0.1f * (float)i - 1.0f) * 0.5f;
    g_a[0] = 0.0f;                                                    // t = -0
    g_a[1] = 87.33f * 0.5f;                                           // t = K5_EXP_MIN
    g_a[2] = 87.330009f * 0.5f;                                       // just below it
    g_a[3] = 87.329994f * 0.5f;                                       // just above it
    ref::k5_gauss(g_a, N, 0.0f, 0.0f, 1.0f, 1, g_ref0);
    vec::k5_gauss(g_a, N, 0.0f, 0.0f, 1.0f, 1, g_vec0);
    EXPECT_EQ(0, Mismatches(g_ref0, g_vec0, N));

    for (int n : K5_SIZES)
    {
        Fill(g_a, n);
        ref::k5_gauss(g_a, n, 20.0f, 30.0f, 0.05f, 1, g_ref0);
        vec::k5_gauss(g_a, n, 20.0f, 30.0f, 0.05f, 1, g_vec0);
        EXPECT_EQ(0, Mismatches(g_ref0, g_vec0, n));
    }
}

/* k5_sumsq / k5_norm change the rounding order (see K5_rvv.cpp), so they are checked against
   the exact sum (double) instead of bit for bit: relative error <= 1e-5. The scalar itself is
   at about 1.3e-5 on the KCF spectrum below; the vector version at 1e-7 .. 6e-7. */
namespace
{
double ExactSumsq(const float *x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)x[i] * (double)x[i];
    return s;
}

bool Close(float got, double exact)
{
    const double err = (double)got - exact;
    return (err < 0 ? -err : err) <= 1e-5 * exact;
}
}  // namespace

TEST(K5Test, SumsqAndNormCloseToExact)
{
    for (int n : K5_SIZES)                                           // random in [-8, 8)
    {
        Fill(g_a, n);
        EXPECT_TRUE(Close(vec::k5_sumsq(g_a, n), ExactSumsq(g_a, n)));
    }

    /* the KCF case: 31 channels of HOG-like features in [0, 0.3), each FFT'd like k5_spectra */
    const int C = K5_MAX_C, N = K5_MAX_N;
    float *re = g_x[0], *im = g_x[1];
    for (int i = 0; i < C * N; i++) { re[i] = (Random() + 8.0f) * (0.3f / 16.0f); im[i] = 0.0f; }
    for (int c = 0; c < C; c++) ref::fft2(re + c * N, im + c * N, 32, 32, 0);
    const double exact = (ExactSumsq(re, C * N) + ExactSumsq(im, C * N)) / N;
    EXPECT_TRUE(Close(vec::k5_norm(re, im, C, N), exact));
    EXPECT_TRUE(Close(vec::k5_sumsq(re, C * N), ExactSumsq(re, C * N)));
}

/* The K5 stages on the KCF size (32 x 32 cells, 31 channels). k5_spectra and k5_make_labels are
   bit-identical to ref::. k5_train / k5_detect use vec::k5_norm, so after a first train, a
   second train and a detect, the response is compared with ref's by peak and by distance. */
namespace
{
constexpr int SN = K5_MAX_N, SC = K5_MAX_C, SCN = SC * SN;
float g_yr[SN], g_yi[SN], g_vyr[SN], g_vyi[SN];                  // label spectra
struct ModelMem
{
    float xr[SCN], xi[SCN], ar[SN], ai[SN], work[6 * SN], resp[SN];
};
ModelMem g_mref, g_mvec;                                         // .bss: about 290 KB each

void InitModel(K5Model &m, K5Work &w, ModelMem &mem)
{
    m.rows = 32; m.cols = 32; m.ch = SC;
    m.inv_sd = 1.0f / (0.5f * 0.5f * (float)SCN);                // sigma 0.5 as in the flow
    m.lambda = 1e-4f; m.eta = 0.02f;
    m.xr = mem.xr; m.xi = mem.xi; m.ar = mem.ar; m.ai = mem.ai;
    m.yr = g_yr;   m.yi = g_yi;
    w.sr = mem.work;          w.si = mem.work + SN;
    w.kr = mem.work + 2 * SN; w.ki = mem.work + 3 * SN;
    w.ar = mem.work + 4 * SN; w.ai = mem.work + 5 * SN;
}

void FillFeatures(float *f)                                      // HOG-like, in [0, 0.3)
{
    for (int i = 0; i < SCN; i++) f[i] = (Random() + 8.0f) * (0.3f / 16.0f);
}

int ArgMax(const float *x, int n)
{
    int k = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[k]) k = i;
    return k;
}
}  // namespace

TEST(K5Test, StagesCloseToReference)
{
    ref::k5_make_labels(32, 32, 2.5f, 0.1f, g_yr, g_yi);
    vec::k5_make_labels(32, 32, 2.5f, 0.1f, g_vyr, g_vyi);
    EXPECT_EQ(0, Mismatches(g_yr, g_vyr, SN));
    EXPECT_EQ(0, Mismatches(g_yi, g_vyi, SN));

    float *xr = g_x[0], *xi = g_x[1], *zr = g_x[2], *zi = g_x[3];
    FillFeatures(xr);                                            // spectra in place, as the flow
    for (int i = 0; i < SCN; i++) zr[i] = xr[i];
    ref::k5_spectra(xr, SC, 32, 32, xr, xi);
    vec::k5_spectra(zr, SC, 32, 32, zr, zi);
    EXPECT_EQ(0, Mismatches(xr, zr, SCN));
    EXPECT_EQ(0, Mismatches(xi, zi, SCN));
    FillFeatures(zr);                                            // z: another patch
    ref::k5_spectra(zr, SC, 32, 32, zr, zi);

    K5Model mr, mv;
    K5Work wr, wv;
    InitModel(mr, wr, g_mref);
    InitModel(mv, wv, g_mvec);
    ref::k5_scalar_train(&mr, xr, xi, 1, 1, &wr);               // first train: model = x
    vec::k5_train(&mv, xr, xi, 1, 1, &wv);
    ref::k5_scalar_train(&mr, zr, zi, 0, 1, &wr);               // second train: blend with z
    vec::k5_train(&mv, zr, zi, 0, 1, &wv);
    ref::k5_scalar_detect(&mr, xr, xi, 1, g_mref.resp, &wr);
    vec::k5_detect(&mv, xr, xi, 1, g_mvec.resp, &wv);

    const int peak = ArgMax(g_mref.resp, SN);
    EXPECT_EQ(peak, ArgMax(g_mvec.resp, SN));
    float worst = 0.0f;
    for (int i = 0; i < SN; i++)
    {
        const float d = g_mvec.resp[i] - g_mref.resp[i];
        if ((d < 0 ? -d : d) > worst) worst = d < 0 ? -d : d;
    }
    EXPECT_TRUE(worst <= 1e-3f * g_mref.resp[peak]);             // measured: about 5e-5 of the peak
}
