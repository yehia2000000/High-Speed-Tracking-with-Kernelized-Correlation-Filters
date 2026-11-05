#include "gtest_lite.hpp"

#include <tuple>

#include "K6_scalar.hpp"
#include "K6_rvv.hpp"

/* vec::k6_detection must return the same location_x / location_y bits as ref::k6_detection.
   The response follows K5's convention: zero shift at (0, 0), periodic, so a peak near the
   right or bottom edge is a negative shift. Each case runs 12 responses: a Gaussian bump at
   random places (edges and corners included), a flat response, and two equal maxima. */

namespace
{
constexpr int K6_MAX = 128;

float g_res[K6_MAX * K6_MAX];
uint32_t g_seed = 31337;

uint32_t NextRandom()
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

float Uniform(float hi) { return hi * (float)(NextRandom() & 0xffff) / 65535.0f; }

/* a periodic bump at (px, py) on noise */
void Bump(int W, int H, int px, int py)
{
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
        {
            int dx = x - px, dy = y - py;
            if (dx > W / 2) dx -= W;
            if (dx < -W / 2) dx += W;
            if (dy > H / 2) dy -= H;
            if (dy < -H / 2) dy += H;
            const float d2 = (float)(dx * dx + dy * dy);
            g_res[y * W + x] = 1.0f / (1.0f + 0.3f * d2 + Uniform(0.05f)) + Uniform(0.01f);
        }
}

bool SameBits(float a, float b)
{
    union { float f; uint32_t u; } x, y;
    x.f = a;
    y.f = b;
    return x.u == y.u;
}
}  // namespace

class K6Test : public ::testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(K6Test, VectorizedMatchesReferenceBitForBit)
{
    const auto [W, H] = GetParam();
    printf("[ %-7s %10d x %-5d ]\n", "resp:", W, H);

    for (int k = 0; k < 12; k++)
    {
        if (k < 9)
        {
            /* corners and edges first, then random places */
            const int xs[3] = {0, W - 1, (int)(NextRandom() % W)};
            const int ys[3] = {0, H - 1, (int)(NextRandom() % H)};
            Bump(W, H, xs[k % 3], ys[k / 3]);
        }
        else if (k == 9)
        {
            for (int i = 0; i < W * H; i++) g_res[i] = 0.5f;      // flat: the first cell wins
        }
        else
        {
            /* two equal maxima, the later one in a lower column: row-major order decides */
            for (int i = 0; i < W * H; i++) g_res[i] = Uniform(0.5f);
            const int r1 = (int)(NextRandom() % H), c1 = (int)(NextRandom() % W);
            const int r2 = (k == 10) ? H - 1 : r1, c2 = (k == 10) ? 0 : W - 1;
            g_res[r1 * W + c1] = 2.0f;
            g_res[r2 * W + c2] = 2.0f;
        }

        float rx, ry, vx, vy;
        ref::k6_detection(g_res, W, H, 4, rx, ry);
        vec::k6_detection(g_res, W, H, 4, vx, vy);
        EXPECT_TRUE(SameBits(rx, vx));
        EXPECT_TRUE(SameBits(ry, vy));
    }
}

// VL at e32 m8 = 32 on VLEN 128, 256 on VLEN 1024: widths below, at and above it.
INSTANTIATE_TEST_SUITE_P(Sizes, K6Test,
    ::testing::Combine(
        ::testing::Values(1, 5, 32, 33, 100, 128),
        ::testing::Values(1, 3, 32, 64))
    );
