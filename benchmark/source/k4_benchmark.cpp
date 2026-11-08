/*
 * Kernel 4 benchmark: ref::k4_normalization_hanning_scalar vs vec::k4_normalization_hanning, on
 * the sizes and layouts of the KCF flow.
 *
 *   Size    in: the K3 grid, KCF_G x KCF_G cells (34 x 34) x 27 bins. out: KCF_NF x KCF_NF cells
 *           (32 x 32) x 31 features. alpha = KCF_DEFAULT_PARAMS.alpha (0.2). Hann = kcf_setup's.
 *   Layout  scalar: interleaved in (27 bins per cell) and out (31 features per cell).
 *           vector: planar in (one plane per bin) and out (one plane per feature).
 *           Both get the same histogram, each in its own layout. The scalar output is converted
 *           to planar (untimed) for the check.
 *   Input   K4_BENCH_REAL=1 (default when kcf_seq.hpp is found): K1 + K2 + scalar K3 on frame 2
 *           at the initial box, the input of the first timed K4 call in KCF_benchmark.
 *           K4_BENCH_REAL=0: a synthetic histogram (unsigned bin = s0 + s1, as K3 writes it).
 *   Timing  one untimed warm-up call of each, then K4_BENCH_ITERS timed calls.
 *
 * The speedup printed here is a guide only: the scalar code in this binary can compile to other
 * instructions than in the flow. Per-kernel speedups come from the KCF_benchmark table.
 */
#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"
#include "riscv_math.hpp"   // cosf
#include "K4_rvv.hpp"
#include "KCF_flow.hpp"     // KCF_G, KCF_NF, KCF_C, KCF_DEFAULT_PARAMS, K1 / K2 / K3 / K4 scalar

#ifndef K4_BENCH_ITERS
#define K4_BENCH_ITERS 3
#endif
#ifndef K4_BENCH_REAL
#if __has_include("kcf_seq.hpp")
#define K4_BENCH_REAL 1
#else
#define K4_BENCH_REAL 0
#endif
#endif

#if K4_BENCH_ITERS < 1
#error "K4_BENCH_ITERS must be 1 or more"
#endif

#if K4_BENCH_REAL
#include "kcf_seq.hpp"
#endif

#define IN_GRID   ref::KCF_G                 // 34
#define IN_CELLS  (IN_GRID * IN_GRID)        // 1156
#define IN_BINS   (27 * IN_CELLS)
#define OUT_GRID  ref::KCF_NF                // 32
#define OUT_CELLS (OUT_GRID * OUT_GRID)      // 1024
#define OUT_CH    ref::KCF_C                 // 31
#define OUT_BINS  (OUT_CH * OUT_CELLS)

static float g_hist_interleaved[IN_BINS];   // scalar in:  hist[cell * 27 + bin]
static float g_hist_planar[IN_BINS];        // vector in:  hist[bin * cells + cell]
static float g_hann[OUT_CELLS];
static float g_scalar_out[OUT_BINS];        // scalar out: out[cell * 31 + ch]
static float g_scalar_planar[OUT_BINS];     // scalar out converted to planar, for the check
static float g_vector_out[OUT_BINS];        // vector out: out[ch * cells + cell]

namespace
{
double ratio(uint64_t a, uint64_t b)
{
    return b ? (double)a / (double)b : 0.0;
}

/* Layout conversions, untimed here. */
inline void to_planar(const float *in, float *out, int cells, int channels)
{
    for (int cell = 0; cell < cells; cell++)
        for (int ch = 0; ch < channels; ch++)
            out[ch * cells + cell] = in[cell * channels + ch];
}

inline void to_interleaved(const float *in, float *out, int cells, int channels)
{
    for (int cell = 0; cell < cells; cell++)
        for (int ch = 0; ch < channels; ch++)
            out[cell * channels + ch] = in[ch * cells + cell];
}

/* kcf_setup's Hann: symmetric, edges exactly 0 (N - 1 denominator). */
void MakeHann()
{
    for (int i = 0; i < OUT_GRID; i++)
        for (int j = 0; j < OUT_GRID; j++)
        {
            float hy = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * i / (OUT_GRID - 1)));
            float hx = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * j / (OUT_GRID - 1)));
            g_hann[i * OUT_GRID + j] = hy * hx;
        }
}

#if K4_BENCH_REAL
float   g_k1[ref::KCF_P * ref::KCF_P];   /* K1 out with the 1-pixel apron */
float   g_mag[ref::KCF_T * ref::KCF_T];
uint8_t g_bin[ref::KCF_T * ref::KCF_T];

/* K1 + K2 + scalar K3 as features() in KCF_flow.cpp does them: frame 2 (frame 1 if the header
   has only one), box = the initial box, ROI = roi_for(). */
void MakeHistogram()
{
    const int f = kcf_seq_frames > 1 ? 1 : 0;
    Image<uint8_t> frame(kcf_seq_width, kcf_seq_height);
    const uint8_t *src = kcf_seq_data + (long)f * kcf_seq_width * kcf_seq_height;
    for (int y = 0; y < kcf_seq_height; y++)
    {
        uint8_t *row = frame.GetPtr(0, y);
        for (int x = 0; x < kcf_seq_width; x++)
            row[x] = src[y * kcf_seq_width + x];
    }

    const ref::KcfParams &p = ref::KCF_DEFAULT_PARAMS;
    const ref::KcfBox b = {kcf_seq_init[0] + kcf_seq_init[2] / 2, kcf_seq_init[1] + kcf_seq_init[3] / 2,
                           kcf_seq_init[2], kcf_seq_init[3]};
    const float grow = (float)ref::KCF_T / (float)(ref::KCF_T - 2 * ref::KCF_CELL);
    ref::Roi r;
    r.cx = b.cx;
    r.cy = b.cy;
    r.tw = b.w * grow * p.padding / 2.5f;   /* K1 multiplies by 2.5 itself */
    r.th = b.h * grow * p.padding / 2.5f;

    ref::k1_scalar(frame, r, ref::KCF_T, g_k1);
    ref::k2_scalar(g_k1, ref::KCF_T, g_mag, g_bin, p.k2);
    ref::k3_scalar(g_bin, g_mag, g_hist_interleaved, ref::KCF_T, ref::KCF_T);
    to_planar(g_hist_interleaved, g_hist_planar, IN_CELLS, 27);
}
#else
/* Synthetic, planar first: signed s0 in planes 9..17, s1 in 18..26, unsigned u = s0 + s1 in 0..8
   (the K3 order). Then the interleaved copy for the scalar. */
void MakeHistogram()
{
    for (int s = 0; s < 18; s++)
        for (int c = 0; c < IN_CELLS; c++)
        {
            const int idx = (9 + s) * IN_CELLS + c;
            g_hist_planar[idx] = 0.05f + (float)((idx * 17 + s * 11) % 50) * 0.02f;
        }
    for (int u = 0; u < 9; u++)
        for (int c = 0; c < IN_CELLS; c++)
            g_hist_planar[u * IN_CELLS + c] =
                g_hist_planar[(9 + u) * IN_CELLS + c] + g_hist_planar[(18 + u) * IN_CELLS + c];
    to_interleaved(g_hist_planar, g_hist_interleaved, IN_CELLS, 27);
}
#endif
}  // namespace

int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    const float alpha = ref::KCF_DEFAULT_PARAMS.alpha;
#if K4_BENCH_REAL
    const char *input = kcf_seq_name;
#else
    const char *input = "synthetic";
#endif

    // 1. Input
    MakeHann();
    MakeHistogram();

    printf("Kernel 4 benchmark: %dx%d cells x 27 bins in, %dx%d cells x %d features out, "
           "alpha %.2f, input %s, 1 warm-up + %d timed iterations\n",
           IN_GRID, IN_GRID, OUT_GRID, OUT_GRID, OUT_CH, (double)alpha, input, K4_BENCH_ITERS);

    // 2. Warm-up, not timed
    ref::k4_normalization_hanning_scalar(g_hist_interleaved, g_hann, g_scalar_out, IN_GRID, IN_GRID, alpha);
    vec::k4_normalization_hanning(g_hist_planar, g_hann, g_vector_out, IN_GRID, IN_GRID, alpha);

    // 3. Timed
    uint64_t ref_cycles = 0, ref_instrs = 0;
    uint64_t vec_cycles = 0, vec_instrs = 0;

    for (int it = 0; it < K4_BENCH_ITERS; ++it)
    {
        Timer t_ref, t_vec;

        t_ref.Start();
        ref::k4_normalization_hanning_scalar(g_hist_interleaved, g_hann, g_scalar_out, IN_GRID, IN_GRID, alpha);
        t_ref.Stop();

        t_vec.Start();
        vec::k4_normalization_hanning(g_hist_planar, g_hann, g_vector_out, IN_GRID, IN_GRID, alpha);
        t_vec.Stop();

        ref_cycles += t_ref.ElapsedCycles();
        ref_instrs += t_ref.ElapsedInstructions();
        vec_cycles += t_vec.ElapsedCycles();
        vec_instrs += t_vec.ElapsedInstructions();
    }

    const uint64_t n  = K4_BENCH_ITERS;
    const uint64_t rc = ref_cycles / n, ri = ref_instrs / n;
    const uint64_t vc = vec_cycles / n, vi = vec_instrs / n;

    printf("\n================ Performance Comparison ================\n");
    printf("%-26s %12s %12s\n", "per call", "cycles", "instrs");
    printf("%-26s %12llu %12llu\n", "Scalar K4 (interleaved)",
           (unsigned long long)rc, (unsigned long long)ri);
    printf("%-26s %12llu %12llu\n", "Vector K4 (planar)",
           (unsigned long long)vc, (unsigned long long)vi);
    printf("\nSpeedup K4:  %.3fx cycles, %.3fx instrs (guide only: the KCF_benchmark table is the reference)\n",
           ratio(rc, vc), ratio(ri, vi));
    printf("========================================================\n\n");

    // 4. Correctness: scalar (converted to planar) vs vector, per channel group.
    //    Output order (both): 0..17 signed, 18..26 unsigned, 27..30 texture energy.
    to_planar(g_scalar_out, g_scalar_planar, OUT_CELLS, OUT_CH);

    float max_diff_signed = 0.0f, max_diff_unsigned = 0.0f, max_diff_energy = 0.0f;
    float overall_max_diff = 0.0f;
    int mismatches = 0;

    for (int ch = 0; ch < OUT_CH; ch++) {
        const float *ref_ch = g_scalar_planar + ch * OUT_CELLS;
        const float *vec_ch = g_vector_out + ch * OUT_CELLS;

        for (int c = 0; c < OUT_CELLS; c++) {
            const float diff = (ref_ch[c] > vec_ch[c]) ? (ref_ch[c] - vec_ch[c]) : (vec_ch[c] - ref_ch[c]);
            if (diff > overall_max_diff) overall_max_diff = diff;

            if (ch < 18) {
                if (diff > max_diff_signed) max_diff_signed = diff;
            } else if (ch < 27) {
                if (diff > max_diff_unsigned) max_diff_unsigned = diff;
            } else {
                if (diff > max_diff_energy) max_diff_energy = diff;
            }

            if (diff > 1e-4f) {
                mismatches++;
                if (mismatches <= 5) {
                    printf("Mismatch at ch %d, cell %d: ref=%f, vec=%f, diff=%e\n",
                           ch, c, ref_ch[c], vec_ch[c], diff);
                }
            }
        }
    }

    printf("Verification Results (Tolerance 1e-4; outputs are at most ~0.85, alpha clamps them):\n");
    printf("  Signed Channels (0..17):     MaxDiff = %e\n", max_diff_signed);
    printf("  Unsigned Channels (18..26):  MaxDiff = %e\n", max_diff_unsigned);
    printf("  Energy Channels (27..30):    MaxDiff = %e\n", max_diff_energy);
    printf("  Overall MaxDiff:             %e\n", overall_max_diff);
    printf("  Total Mismatches:            %d / %d\n", mismatches, OUT_BINS);

    if (mismatches == 0) {
        printf("Output is correct.\n");
        return 0;
    } else {
        printf("Output is wrong!\n");
        return 1;
    }
}