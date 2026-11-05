#include "gtest_lite.hpp"

#include <tuple>

#include "K1_scalar.hpp"
#include "K1_rvv.hpp"

/* vec::k1_rvv must give the same bits as ref::k1_scalar for every output pixel.
   Parameters: source image width, height, template size T (output is (T+2) x (T+2)).
   Each case runs 8 windows: inside the image, half outside, fully outside (clamped), tiny,
   and bigger than the image. */

namespace
{
constexpr int K1_MAX_P = 136 + 2;

float g_ref[K1_MAX_P * K1_MAX_P], g_vec[K1_MAX_P * K1_MAX_P];
uint32_t g_i0[K1_MAX_P], g_i1[K1_MAX_P];
float g_fx[K1_MAX_P];
uint32_t g_seed = 777;

uint32_t NextRandom()
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

float Uniform(float lo, float hi) { return lo + (hi - lo) * (float)(NextRandom() & 0xffff) / 65535.0f; }

int CountBitMismatches(const float *a, const float *b, int n)
{
    int bad = 0;
    for (int i = 0; i < n; i++)
    {
        union { float f; uint32_t u; } x, y;
        x.f = a[i];
        y.f = b[i];
        if (x.u != y.u) bad++;
    }
    return bad;
}
}  // namespace

class K1Test : public ::testing::TestWithParam<std::tuple<int, int, int>>
{
};

TEST_P(K1Test, VectorizedMatchesReferenceBitForBit)
{
    const auto [W, H, T] = GetParam();
    printf("[ %-7s %10d x %-5d ]\n", "image:", W, H);
    printf("[ %-7s %18d ]\n", "T:", T);

    Image<uint8_t> img(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img.GetPtr(0, y)[x] = (uint8_t)(NextRandom() & 0xff);

    for (int k = 0; k < 8; k++)
    {
        ref::Roi r;
        if (k == 0)      { r.cx = 0.5f * W; r.cy = 0.5f * H; r.tw = 0.3f * W; r.th = 0.3f * H; }
        else if (k == 1) { r.cx = 0.0f; r.cy = (float)H; r.tw = 0.5f * W; r.th = 0.5f * H; }
        else if (k == 2) { r.cx = -2.0f * W; r.cy = 3.0f * H; r.tw = 5.0f; r.th = 5.0f; }
        else if (k == 3) { r.cx = Uniform(0, W); r.cy = Uniform(0, H); r.tw = 0.7f; r.th = 1.3f; }
        else if (k == 4) { r.cx = 0.5f * W; r.cy = 0.5f * H; r.tw = 2.0f * W; r.th = 2.0f * H; }
        else
        {
            r.cx = Uniform(-0.2f * W, 1.2f * W);
            r.cy = Uniform(-0.2f * H, 1.2f * H);
            r.tw = Uniform(1.0f, (float)W);
            r.th = Uniform(1.0f, (float)H);
        }
        ref::k1_scalar(img, r, T, g_ref);
        vec::k1_rvv(img, r, T, g_vec, g_i0, g_i1, g_fx);
        EXPECT_EQ(0, CountBitMismatches(g_ref, g_vec, (T + 2) * (T + 2)));
    }
}

// VL at e32 m1 = VLEN/32: 4 on VLEN 128. P = T + 2 = 3, 16, 17, 32, 138 (KCF).
INSTANTIATE_TEST_SUITE_P(ImagesAndSizes, K1Test,
    ::testing::Combine(
        ::testing::Values(1, 17, 320),
        ::testing::Values(1, 9, 240),
        ::testing::Values(1, 14, 15, 30, 136))
    );
