#include "gtest_lite.hpp"

#include <tuple>

#include "K4_scalar.hpp"
#include "K4_rvv.hpp"

/* vec::k4_normalization_hanning against ref::k4_normalization_hanning_scalar.
   Layouts:  ref  in hist[cell * 27 + b],  out fhog[cell * 31 + ch]   (interleaved)
             vec  in hist[b * cells + cell], out fhog[ch * out_cells + cell]   (planar)
   Not bit-identical (1/norm then multiply instead of a divide, other sum orders, the texture
   channels scaled by h * 0.2357 instead of (sum * 0.2357) * h), so the check is a relative
   tolerance on every output; the features are >= 0. */

namespace
{
constexpr int K4_MAX_G = 64;
constexpr int K4_MAX_CELLS = K4_MAX_G * K4_MAX_G;
constexpr float K4_ALPHA = 0.2f;
constexpr float K4_REL_TOL = 1e-5f;

float g_hist_i[K4_MAX_CELLS * 27], g_hist_p[K4_MAX_CELLS * 27];
float g_hann[K4_MAX_CELLS];
float g_ref[K4_MAX_CELLS * 31], g_vec[K4_MAX_CELLS * 31];
uint32_t g_seed = 99;

uint32_t NextRandom()
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

float Uniform(float hi) { return hi * (float)(NextRandom() & 0xffff) / 65535.0f; }

/* K3-like histograms: 18 signed bins, unsigned = the two signed halves added, about 1 cell in
   8 empty (norm = epsilon), and a few strong bins so the alpha truncation is used. */
void Fill(int gw, int gh)
{
    const int cells = gw * gh;
    for (int cell = 0; cell < cells; cell++)
    {
        float *h = g_hist_i + cell * 27;
        const bool empty = (NextRandom() % 8) == 0;
        for (int s = 0; s < 18; s++)
            h[9 + s] = empty ? 0.0f : ((NextRandom() % 6) == 0 ? Uniform(8.0f) : Uniform(1.0f));
        for (int u = 0; u < 9; u++) h[u] = h[9 + u] + h[18 + u];
        for (int b = 0; b < 27; b++) g_hist_p[b * cells + cell] = h[b];
    }
    const int out_cells = (gw - 2) * (gh - 2);
    for (int i = 0; i < out_cells; i++) g_hann[i] = Uniform(1.0f);
    g_hann[0] = 0.0f;
}
}  // namespace

class K4Test : public ::testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(K4Test, PlanarMatchesReferenceWithinTolerance)
{
    const auto [gw, gh] = GetParam();
    printf("[ %-7s %10d x %-5d ]\n", "grid:", gw, gh);

    Fill(gw, gh);
    const int out_cells = (gw - 2) * (gh - 2);
    ref::k4_normalization_hanning_scalar(g_hist_i, g_hann, g_ref, gw, gh, K4_ALPHA);
    vec::k4_normalization_hanning(g_hist_p, g_hann, g_vec, gw, gh, K4_ALPHA);

    int over = 0, bits = 0;
    float worst = 0.0f;
    for (int cell = 0; cell < out_cells; cell++)
        for (int ch = 0; ch < 31; ch++)
        {
            const float r = g_ref[cell * 31 + ch], v = g_vec[ch * out_cells + cell];
            const float d = (r > v) ? r - v : v - r;
            if (d != 0.0f) bits++;
            if (d > K4_REL_TOL * r) over++;
            if (r > 0.0f && d / r > worst) worst = d / r;
        }
    printf("[ %-7s %10d of %-5d ]\n", "differ:", bits, out_cells * 31);
    printf("[ %-7s %18.3e ]\n", "max rel:", (double)worst);
    EXPECT_EQ(0, over);
}

// VL at e32 m2 = 8 on VLEN 128: inner widths 1, 2, 8, 9, 17, 32 (KCF) and 62.
INSTANTIATE_TEST_SUITE_P(Grids, K4Test,
    ::testing::Combine(
        ::testing::Values(3, 4, 10, 11, 19, 34, 64),
        ::testing::Values(3, 7, 34, 64))
    );
