#include "gtest_lite.hpp"

#include <tuple>

#include "K3_scalar.hpp"
#include "K3_rvv.hpp"

/* vec::Histogram_k3 against ref::k3_scalar<4>. The two layouts differ:
     ref  interleaved, hist[cell * 27 + ch]
     vec  planar,      hist[ch * cells + cell]
   and the vector version is not bit-identical (weights m * (wy * wx) instead of (m * wy) * wx,
   another accumulation order, unsigned bins = signed sum at the end), so the check is a
   relative tolerance. Every bin is a sum of non-negative terms, so |ref - vec| <= tol * ref
   is a fair test. The test prints the largest relative error and how many bins differ in
   their bits. */

namespace
{
constexpr int K3_MAX_W = 256, K3_MAX_H = 136;
constexpr int K3_MAX_CELLS = (K3_MAX_W / 4) * (K3_MAX_H / 4);
constexpr float K3_REL_TOL = 1e-5f;

uint8_t g_bin[K3_MAX_W * K3_MAX_H];
float   g_mag[K3_MAX_W * K3_MAX_H];
float   g_ref[K3_MAX_CELLS * 27], g_vec[K3_MAX_CELLS * 27];
uint32_t g_seed = 4242;

uint32_t NextRandom()
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

enum K3Input
{
    K3_IN_RANDOM,   // random magnitudes 0..1.4, random bins 0..17
    K3_IN_ONE_BIN,  // every pixel in bin 5: signed and unsigned bins get the same sums
    K3_IN_ZERO      // magnitude 0 everywhere: every bin must be exactly 0
};

void Fill(K3Input input, int W, int H)
{
    for (int i = 0; i < W * H; i++)
    {
        g_bin[i] = (input == K3_IN_ONE_BIN) ? 5 : (uint8_t)(NextRandom() % 18);
        g_mag[i] = (input == K3_IN_ZERO) ? 0.0f : 1.4f * (float)(NextRandom() & 0xffff) / 65535.0f;
    }
}

const char *InputName(K3Input in)
{
    return in == K3_IN_RANDOM ? "random" : in == K3_IN_ONE_BIN ? "one bin" : "zero";
}
}  // namespace

class K3Test : public ::testing::TestWithParam<std::tuple<int, int, K3Input>>
{
};

TEST_P(K3Test, PlanarMatchesReferenceWithinTolerance)
{
    const auto [W, H, input] = GetParam();
    printf("[ %-7s %10d x %-5d ]\n", "image:", W, H);
    printf("[ %-7s %18s ]\n", "input:", InputName(input));

    Fill(input, W, H);
    const int cells = (W / 4) * (H / 4);
    ref::k3_scalar<4>(g_bin, g_mag, g_ref, W, H);
    vec::Histogram_k3(g_bin, g_mag, g_vec, W, H);

    int over = 0, bits = 0;
    float worst = 0.0f;
    for (int cell = 0; cell < cells; cell++)
        for (int ch = 0; ch < 27; ch++)
        {
            const float r = g_ref[cell * 27 + ch], v = g_vec[ch * cells + cell];
            const float d = (r > v) ? r - v : v - r;
            if (d != 0.0f) bits++;
            if (d > K3_REL_TOL * r) over++;
            if (r > 0.0f && d / r > worst) worst = d / r;
        }
    printf("[ %-7s %10d of %-5d ]\n", "differ:", bits, cells * 27);
    printf("[ %-7s %18.3e ]\n", "max rel:", (double)worst);
    EXPECT_EQ(0, over);
    if (input == K3_IN_ZERO) EXPECT_EQ(0, bits);
}

// grid_w = W / 4 from 2 to 64 (the ring buffer limit), grid_h from 2. VL at e32 m2 = 8 on
// VLEN 128: grids of 2, 3, 9, 17, 34 (KCF) and 64 cells per row.
INSTANTIATE_TEST_SUITE_P(SizesInputs, K3Test,
    ::testing::Combine(
        ::testing::Values(8, 12, 36, 68, 136, 256),
        ::testing::Values(8, 20, 136),
        ::testing::Values(K3_IN_RANDOM, K3_IN_ONE_BIN, K3_IN_ZERO))
    );
