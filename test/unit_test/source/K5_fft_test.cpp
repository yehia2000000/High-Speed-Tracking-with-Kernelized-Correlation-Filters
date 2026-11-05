#include "gtest_lite.hpp"

#include <tuple>

#include "K5_scalar.hpp"    // ref::fft2
#include "K5_fft_rvv.hpp"   // vec::fft2

/* vec::fft2 must give the same bits as ref::fft2, re and im, for every size, input and
   direction. Parameters: rows, cols, input (5 x 4 x 4 = 80 cases, under gtest_lite's 128 per
   fixture); each case runs forward and inverse.
   Sizes: 1 (no butterfly), 2, powers of 2 below and above VL (16 at e32 m4 on VLEN 128,
   32 on VLEN 256), the KCF size 32 x 32, non-square (row and column passes of different
   lengths), and rows * cols > 4096 (the transpose scratch is too small: strided row pass). */

namespace
{
constexpr int FFT_MAX = 128 * 64;

enum FftInput
{
    FFT_IN_RANDOM,    // re and im random in [-8, 8)
    FFT_IN_REAL,      // im = 0: the K5 case (features)
    FFT_IN_IMPULSE,   // 1 at (1, 1): the spectrum is products of twiddles, every one is used
    FFT_IN_SCALES     // random sign and mantissa, exponents from 2^-60 to 2^60
};

float g_re[FFT_MAX], g_im[FFT_MAX];                      // input
float g_rr[FFT_MAX], g_ri[FFT_MAX];                      // ref::fft2 in place
float g_vr[FFT_MAX], g_vi[FFT_MAX];                      // vec::fft2 in place
uint32_t g_seed = 4242;

uint32_t NextRandom()
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

float RandomFloat()                                      // [-8, 8) with a fraction
{
    return (float)(NextRandom() & 0xffff) / 4096.0f - 8.0f;
}

float RandomScaled()                                     // +-[1, 2) * 2^e, e in [-60, 60]
{
    union { uint32_t u; float f; } v;
    const uint32_t sign = NextRandom() & 1, e = 127 - 60 + NextRandom() % 121;
    v.u = (sign << 31) | (e << 23) | (NextRandom() & 0x7fffff);
    return v.f;
}

void Fill(FftInput input, int rows, int cols)
{
    const int n = rows * cols;
    for (int i = 0; i < n; i++)
    {
        switch (input)
        {
        case FFT_IN_RANDOM:  g_re[i] = RandomFloat();  g_im[i] = RandomFloat();  break;
        case FFT_IN_REAL:    g_re[i] = RandomFloat();  g_im[i] = 0.0f;           break;
        case FFT_IN_IMPULSE: g_re[i] = 0.0f;           g_im[i] = 0.0f;           break;
        case FFT_IN_SCALES:  g_re[i] = RandomScaled(); g_im[i] = RandomScaled(); break;
        }
    }
    if (input == FFT_IN_IMPULSE)
        g_re[(1 % rows) * cols + (1 % cols)] = 1.0f;
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

/* ref and vec on copies of the same input, then compare */
int RunBoth(int rows, int cols, int inverse)
{
    const int n = rows * cols;
    for (int i = 0; i < n; i++)
    {
        g_rr[i] = g_vr[i] = g_re[i];
        g_ri[i] = g_vi[i] = g_im[i];
    }
    ref::fft2(g_rr, g_ri, rows, cols, inverse);
    vec::fft2(g_vr, g_vi, rows, cols, inverse);
    return Mismatches(g_rr, g_vr, n) + Mismatches(g_ri, g_vi, n);
}

const char *InputName(FftInput in)
{
    return in == FFT_IN_RANDOM ? "random" : in == FFT_IN_REAL ? "real" :
           in == FFT_IN_IMPULSE ? "impulse" : "scales";
}
}  // namespace

class K5FftTest : public ::testing::TestWithParam<std::tuple<int, int, FftInput>>
{
};

TEST_P(K5FftTest, VectorizedMatchesReferenceBitForBit)
{
    const auto [rows, cols, input] = GetParam();
    printf("[ %-7s %12d x %d ]\n", "size:", rows, cols);
    printf("[ %-7s %16s ]\n", "input:", InputName(input));

    Fill(input, rows, cols);
    EXPECT_EQ(0, RunBoth(rows, cols, 0));                // forward
    EXPECT_EQ(0, RunBoth(rows, cols, 1));                // inverse
}

INSTANTIATE_TEST_SUITE_P(SizesInputs, K5FftTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 8, 32, 64),
        ::testing::Values(1, 4, 32, 128),
        ::testing::Values(FFT_IN_RANDOM, FFT_IN_REAL, FFT_IN_IMPULSE, FFT_IN_SCALES))
    );

/* Both versions cache their twiddle and swap tables for the last length. Alternate lengths so
   every call starts with tables made for another size. */
TEST(K5FftCacheTest, AlternatingSizes)
{
    const int sizes[][2] = {{32, 32}, {64, 16}, {32, 32}, {16, 64}, {8, 128}, {32, 32}, {2, 2}};
    for (const auto &s : sizes)
    {
        Fill(FFT_IN_RANDOM, s[0], s[1]);
        EXPECT_EQ(0, RunBoth(s[0], s[1], 0));
        EXPECT_EQ(0, RunBoth(s[0], s[1], 1));
    }
}

/* The KCF use: 31 channels of real features, 32 x 32, forward (k5_spectra), then the inverse
   brings the features back (a sanity check of the convention: forward unscaled, inverse / N). */
TEST(K5FftKcfTest, ChannelsAndRoundTrip)
{
    constexpr int R = 32, C = 32, N = R * C, CH = 31;
    for (int ch = 0; ch < CH; ch++)
    {
        Fill(FFT_IN_REAL, R, C);
        for (int i = 0; i < N; i++) g_re[i] = (g_re[i] + 8.0f) * (0.3f / 16.0f);   // [0, 0.3)
        EXPECT_EQ(0, RunBoth(R, C, 0));

        /* inverse of vec's spectrum: back to the features within float rounding */
        vec::fft2(g_vr, g_vi, R, C, 1);
        float worst = 0.0f;
        for (int i = 0; i < N; i++)
        {
            const float d = g_vr[i] - g_re[i], e = g_vi[i];
            const float a = d < 0 ? -d : d, b = e < 0 ? -e : e;
            if (a > worst) worst = a;
            if (b > worst) worst = b;
        }
        EXPECT_TRUE(worst < 1e-6f);
    }
}
