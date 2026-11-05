#include "gtest_lite.hpp"

#include <tuple>

#include "K2_scalar.hpp"
#include "K2_rvv.hpp"
#include "riscv_math.hpp"   // cos, sin (no <cmath> on bare metal)

/* vec::k2_rvv must give the same bits as ref::k2_scalar: every mag (as a 32-bit pattern) and
   every bin, for every method, size and input. Parameters: template size T, method, input. */

namespace
{
constexpr int K2_MAX_T = 136;                    // the KCF template
constexpr int K2_MAX_P = K2_MAX_T + 2;

enum K2Input
{
    K2_IN_RANDOM,   // random floats with fractions, like K1's bilinear output
    K2_IN_TIES,     // integers 0..3: many gx = 0, gy = 0, |gx| = |gy|, flat pixels
    K2_IN_BLOCKS,   // 200 / 10 blocks: pure horizontal and vertical edges
    K2_IN_RAMPS     // linear ramps on the 18 bin edges: gradients within an ulp of a boundary
};

float g_in[K2_MAX_P * K2_MAX_P];
float g_mag_ref[K2_MAX_T * K2_MAX_T], g_mag_vec[K2_MAX_T * K2_MAX_T];
uint8_t g_bin_ref[K2_MAX_T * K2_MAX_T], g_bin_vec[K2_MAX_T * K2_MAX_T];
uint32_t g_seed = 12345;

uint32_t NextRandom()
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

/* edge: which bin edge the ramp sits on (only for K2_IN_RAMPS) */
void Fill(K2Input input, int T, int edge)
{
    const int P = T + 2;
    const double th = (20.0 * edge + 10.0) * 3.14159265358979 / 180.0;
    const double cx = cos(th), cy = sin(th);
    for (int y = 0; y < P; y++)
        for (int x = 0; x < P; x++)
        {
            float v;
            if (input == K2_IN_RANDOM)
                v = (float)(NextRandom() & 0xffff) / 65536.0f * 255.0f;
            else if (input == K2_IN_TIES)
                v = (float)(NextRandom() % 4);
            else if (input == K2_IN_BLOCKS)
                v = ((x / 3 + y / 5) & 1) ? 200.0f : 10.0f;
            else
                v = (float)(100.0 + 0.37 * (cx * x + cy * y));
            g_in[y * P + x] = v;
        }
}

/* number of outputs whose bits differ */
int CountMismatches(int T)
{
    int bad = 0;
    for (int i = 0; i < T * T; i++)
    {
        union { float f; uint32_t u; } a, b;            // compare bits, no memcmp on bare metal
        a.f = g_mag_ref[i];
        b.f = g_mag_vec[i];
        if (a.u != b.u) bad++;
        if (g_bin_ref[i] != g_bin_vec[i]) bad++;
    }
    return bad;
}

const char *MethodName(ref::K2Method m)
{
    return m == ref::K2_DOT ? "DOT" : m == ref::K2_POLY ? "POLY" : "ATAN2";
}

const char *InputName(K2Input in)
{
    return in == K2_IN_RANDOM ? "random" : in == K2_IN_TIES ? "ties" : in == K2_IN_BLOCKS ? "blocks" : "ramps";
}
}  // namespace

class K2Test : public ::testing::TestWithParam<std::tuple<int, ref::K2Method, K2Input>>
{
};

TEST_P(K2Test, VectorizedMatchesReferenceBitForBit)
{
    const auto [T, method, input] = GetParam();
    printf("[ %-7s %16d ]\n", "T:", T);
    printf("[ %-7s %16s ]\n", "method:", MethodName(method));
    printf("[ %-7s %16s ]\n", "input:", InputName(input));

    /* 18 ramps per size: up to T = 64, so the gem5 run stays short */
    const int edges = (input == K2_IN_RAMPS) ? (T <= 64 ? 18 : 0) : 1;
    for (int edge = 0; edge < edges; edge++)
    {
        Fill(input, T, edge);
        ref::k2_scalar(g_in, T, g_mag_ref, g_bin_ref, method);
        vec::k2_rvv(g_in, T, g_mag_vec, g_bin_vec, method);
        EXPECT_EQ(0, CountMismatches(T));
    }
}

// VL at e32 m4 = VLEN/8: 16 on VLEN 128, 32 on VLEN 256. Sizes below, at and above it, and
// the KCF template (136 = 8 x 16 + 8: a tail on every row).
INSTANTIATE_TEST_SUITE_P(SizesMethodsInputs, K2Test,
    ::testing::Combine(
        ::testing::Values(1, 7, 16, 17, 33, 64, 136),
        ::testing::Values(ref::K2_DOT, ref::K2_POLY),
        ::testing::Values(K2_IN_RANDOM, K2_IN_TIES, K2_IN_BLOCKS, K2_IN_RAMPS))
    );

// K2_ATAN2 has no vector version: vec::k2_rvv calls the scalar. One check that it still matches.
TEST(K2AtanTest, ScalarFallbackMatches)
{
    Fill(K2_IN_RANDOM, 33, 0);
    ref::k2_scalar(g_in, 33, g_mag_ref, g_bin_ref, ref::K2_ATAN2);
    vec::k2_rvv(g_in, 33, g_mag_vec, g_bin_vec, ref::K2_ATAN2);
    EXPECT_EQ(0, CountMismatches(33));
}
