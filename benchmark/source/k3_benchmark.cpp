/*
 * Kernel 3 benchmark: ref::k3_scalar vs vec::Histogram_k3, under the conditions K3 sees
 * inside the KCF flow (KCF_benchmark.cpp).
 *
 *   Size    K3_BENCH_SIZE x K3_BENCH_SIZE, default 136 = KCF_T (34 x 34 cells, as in the flow).
 *   Input   K3_BENCH_REAL=1 (default when kcf_seq.hpp is found): K1 + K2 on frame 2 at the
 *           initial box, the input of the first timed K3 call in KCF_benchmark.
 *           K3_BENCH_REAL=0: the old synthetic pattern.
 *   Timing  one untimed warm-up call of each kernel, then K3_BENCH_ITERS timed calls.
 *           gem5 is deterministic: the warm-up removes the cold start, more iterations do not
 *           average out noise.
 *   Layout  the scalar K3 writes interleaved, the vector K4 reads planar, so in the flow the
 *           scalar choice costs K3 + the layout stage. Both speedups are printed.
 *   Cache   K3_BENCH_POLLUTE_KB (default 1024, 0 = off): untimed, before every timed K3 call,
 *           write a buffer of that size, then the K3 inputs. In the flow, the stages between two
 *           K3 calls (K4, the 31 FFTs, detect / train, K6, K1) touch about 1 MB of KcfState, and
 *           K2 writes mag / bin just before K3. Not before the layout: in the flow it runs right
 *           after K3.
 *
 * Plane-stride check (vector output: 27 planes of cells * 4 bytes):
 *   -DK3_BENCH_SIZE=128 -> 1024 cells, 4096 B (power of 2)
 *   -DK3_BENCH_SIZE=132 -> 1089 cells, 4356 B
 *   -DK3_BENCH_SIZE=136 -> 1156 cells, 4624 B (the flow)
 */
#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"
#include "K3_rvv.hpp"

#ifndef K3_BENCH_SIZE
#define K3_BENCH_SIZE 136
#endif
#ifndef K3_BENCH_ITERS
#define K3_BENCH_ITERS 3
#endif
#ifndef K3_BENCH_POLLUTE_KB
#define K3_BENCH_POLLUTE_KB 1024
#endif
#ifndef K3_BENCH_REAL
#if __has_include("kcf_seq.hpp")
#define K3_BENCH_REAL 1
#else
#define K3_BENCH_REAL 0
#endif
#endif

#if K3_BENCH_SIZE % 4 != 0
#error "K3_BENCH_SIZE must be a multiple of 4 (the cell size)"
#endif
#if K3_BENCH_ITERS < 1
#error "K3_BENCH_ITERS must be 1 or more"
#endif
#if K3_BENCH_POLLUTE_KB < 0
#error "K3_BENCH_POLLUTE_KB must be 0 (off) or more"
#endif

#if K3_BENCH_REAL
#include "KCF_flow.hpp"   // ref::k1_scalar, ref::k2_scalar, ref::Roi, ref::KCF_DEFAULT_PARAMS
#include "kcf_seq.hpp"
#endif

#define BENCH_SIZE  K3_BENCH_SIZE
#define BENCH_GRID  (BENCH_SIZE / 4)
#define BENCH_CELLS (BENCH_GRID * BENCH_GRID)
#define BENCH_BINS  (27 * BENCH_CELLS)

static float   g_mag[BENCH_SIZE * BENCH_SIZE];
static uint8_t g_ori_bins[BENCH_SIZE * BENCH_SIZE];

static float g_scalar_interleaved[BENCH_BINS];   // scalar K3 out
static float g_scalar_planar[BENCH_BINS];        // scalar K3 out after the layout stage
static float g_vector_planar[BENCH_BINS];        // vector K3 out

namespace
{
/* The layout stage of KCF_flow.cpp: interleaved K3 -> planar for the vector K4. */
inline void to_planar(const float *in, float *out, int cells, int channels)
{
    for (int cell = 0; cell < cells; cell++)
        for (int ch = 0; ch < channels; ch++)
            out[ch * cells + cell] = in[cell * channels + ch];
}

double ratio(uint64_t a, uint64_t b)
{
    return b ? (double)a / (double)b : 0.0;
}

#if K3_BENCH_POLLUTE_KB > 0
/* Stands in for the KcfState the other stages touch between two K3 calls. */
float g_pollute[K3_BENCH_POLLUTE_KB * 1024 / sizeof(float)];
#endif

/* Untimed, before every timed K3 call: the caches as the flow leaves them for K3.
   1. One word written in every 64 B line of g_pollute: K3's output and inputs leave the caches.
   2. One word written in every line of mag and bin: K2 has just written them.
   volatile: the compiler keeps every store. */
void FlowLikeCache()
{
#if K3_BENCH_POLLUTE_KB > 0
    const int words = (int)(sizeof(g_pollute) / sizeof(float));
    volatile float *p = g_pollute;
    for (int i = 0; i < words; i += 16)
        p[i] = p[i] + 1.0f;

    volatile float *m = g_mag;
    for (int i = 0; i < BENCH_SIZE * BENCH_SIZE; i += 16)
        m[i] = m[i];
    volatile uint8_t *b = g_ori_bins;
    for (int i = 0; i < BENCH_SIZE * BENCH_SIZE; i += 64)
        b[i] = b[i];
#endif
}

#if K3_BENCH_REAL
float g_k1[(BENCH_SIZE + 2) * (BENCH_SIZE + 2)];   /* K1 out with the 1-pixel apron */

/* K1 + K2 as features() in KCF_flow.cpp does them: frame 2 (frame 1 if the header has only
   one), box = the initial box, ROI = roi_for() with this template size. Scalar K1 / K2: the
   input only has to be real, it is not timed. */
void MakeRealInput()
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
    const float grow = (float)BENCH_SIZE / (float)(BENCH_SIZE - 2 * ref::KCF_CELL);
    ref::Roi r;
    r.cx = b.cx;
    r.cy = b.cy;
    r.tw = b.w * grow * p.padding / 2.5f;   /* K1 multiplies by 2.5 itself */
    r.th = b.h * grow * p.padding / 2.5f;

    ref::k1_scalar(frame, r, BENCH_SIZE, g_k1);
    ref::k2_scalar(g_k1, BENCH_SIZE, g_mag, g_ori_bins, p.k2);
}
#endif
}  // namespace

int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    const int size = BENCH_SIZE;

    // 1. Input
#if K3_BENCH_REAL
    MakeRealInput();
    const char *input = kcf_seq_name;
#else
    for (int i = 0; i < size * size; i++) {
        g_mag[i] = 1.0f + static_cast<float>((i * 37) % 100) * 0.1f;
        g_ori_bins[i] = static_cast<uint8_t>((i * 13) % 18);
    }
    const char *input = "synthetic";
#endif

    printf("Kernel 3 benchmark: %dx%d, grid %dx%d (%d cells), plane stride %d B, input %s, "
           "1 warm-up + %d timed iterations, cache pollute %d KB before each K3%s\n",
           size, size, BENCH_GRID, BENCH_GRID, BENCH_CELLS, BENCH_CELLS * (int)sizeof(float),
           input, K3_BENCH_ITERS, K3_BENCH_POLLUTE_KB, K3_BENCH_POLLUTE_KB ? "" : " (off)");

    // 2. Warm-up, not timed: code, inputs and outputs in the caches, predictors trained,
    //    like the earlier calls of the flow leave them.
    ref::k3_scalar(g_ori_bins, g_mag, g_scalar_interleaved, size, size);
    to_planar(g_scalar_interleaved, g_scalar_planar, BENCH_CELLS, 27);
    vec::Histogram_k3(g_ori_bins, g_mag, g_vector_planar, size, size);

    // 3. Timed
    uint64_t ref_cycles = 0, ref_instrs = 0;
    uint64_t lay_cycles = 0, lay_instrs = 0;
    uint64_t vec_cycles = 0, vec_instrs = 0;

    for (int it = 0; it < K3_BENCH_ITERS; ++it)
    {
        Timer t_ref, t_lay, t_vec;

        FlowLikeCache();   // untimed
        t_ref.Start();
        ref::k3_scalar(g_ori_bins, g_mag, g_scalar_interleaved, size, size);
        t_ref.Stop();

        t_lay.Start();     // right after K3, as in the flow: no FlowLikeCache here
        to_planar(g_scalar_interleaved, g_scalar_planar, BENCH_CELLS, 27);
        t_lay.Stop();

        FlowLikeCache();   // untimed
        t_vec.Start();
        vec::Histogram_k3(g_ori_bins, g_mag, g_vector_planar, size, size);
        t_vec.Stop();

        ref_cycles += t_ref.ElapsedCycles();
        ref_instrs += t_ref.ElapsedInstructions();
        lay_cycles += t_lay.ElapsedCycles();
        lay_instrs += t_lay.ElapsedInstructions();
        vec_cycles += t_vec.ElapsedCycles();
        vec_instrs += t_vec.ElapsedInstructions();
    }

    const uint64_t n  = K3_BENCH_ITERS;
    const uint64_t rc = ref_cycles / n, ri = ref_instrs / n;
    const uint64_t lc = lay_cycles / n, li = lay_instrs / n;
    const uint64_t vc = vec_cycles / n, vi = vec_instrs / n;

    printf("\n================ Performance Comparison (%dx%d) ================\n", size, size);
    printf("%-26s %12s %12s\n", "per call", "cycles", "instrs");
    printf("%-26s %12llu %12llu\n", "Scalar K3 (interleaved)",
           (unsigned long long)rc, (unsigned long long)ri);
    printf("%-26s %12llu %12llu\n", "Layout to_planar",
           (unsigned long long)lc, (unsigned long long)li);
    printf("%-26s %12llu %12llu\n", "Scalar K3 + to_planar",
           (unsigned long long)(rc + lc), (unsigned long long)(ri + li));
    printf("%-26s %12llu %12llu\n", "Vector Histogram_k3",
           (unsigned long long)vc, (unsigned long long)vi);
    printf("\nSpeedup K3 only:      %.3fx cycles, %.3fx instrs\n", ratio(rc, vc), ratio(ri, vi));
    printf("Speedup K3 + layout:  %.3fx cycles, %.3fx instrs\n", ratio(rc + lc, vc), ratio(ri + li, vi));
    printf("=================================================================\n\n");

    // 4. Correctness: scalar (after the layout stage) vs vector, both planar.
    //    Relative tolerance: real magnitudes give bins in the thousands, where a different
    //    summation order alone moves the result by more than 1e-4 absolute.
    float max_rel = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < BENCH_BINS; i++) {
        const float s = g_scalar_planar[i], v = g_vector_planar[i];
        const float diff = (s > v) ? (s - v) : (v - s);
        const float as = (s > 0.0f) ? s : -s;
        const float rel = diff / (as > 1.0f ? as : 1.0f);
        if (rel > max_rel)
            max_rel = rel;
        if (rel > 1e-4f)
            mismatches++;
    }

    printf("Verification Results (relative tolerance 1e-4):\n");
    printf("  MaxRelDiff:       %e\n", max_rel);
    printf("  Total Mismatches: %d / %d\n", mismatches, BENCH_BINS);

    if (mismatches == 0) {
        printf("Output is correct.\n");
        return 0;
    } else {
        printf("Output is wrong!\n");
        return 1;
    }
}