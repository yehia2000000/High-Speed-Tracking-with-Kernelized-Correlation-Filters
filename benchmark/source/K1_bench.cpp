/*
 * K1 benchmark on real frames: ref::k1_scalar vs vec::k1_rvv, with the tracker's geometry.
 *
 * Frames: images/kcf_seq.hpp (the same file as KCF_benchmark), made on the host with
 *     python3 scripts/kcf_sequence.py SEQ_DIR --header images/kcf_seq.hpp --frames 3
 *
 * The flow calls K1 twice per frame: detect at the previous box, then train at the new box.
 * This benchmark makes the same two calls for every frame after the first, with T = KCF_T and
 * the window of the flow's roi_for(), so "K1 per frame" here is comparable with the
 * "K1 subwindow (x2)" row of KCF_benchmark. The boxes are the ground truth (the init box for
 * frame 1), so no tracker has to run: K1 does the same work for any box (P x P outputs).
 *
 * Before each timed pair, one untimed scalar call touches the same window, so both versions
 * start with the same cache state (otherwise the second one would find the rows already loaded).
 *
 * gem5: configure with -DGEM5_EXTRA_ARGS="--disable-difftest;--maxinsts=2000000000".
 * qemu-user: same code, a quick functional check; its cycle counters are host ticks.
 */
#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"
#include "K1_scalar.hpp"   // ref::k1_scalar, ref::Roi
#include "K1_rvv.hpp"      // vec::k1_rvv (riscv_cv.hpp does not include it)
#include "KCF_flow.hpp"    // KCF_T, KCF_CELL, KCF_P, KcfBox, KCF_DEFAULT_PARAMS

#if __has_include("kcf_seq.hpp")
#include "kcf_seq.hpp"
#define KCF_HAVE_SEQ 1
#endif

#ifdef KCF_HAVE_SEQ
namespace
{

constexpr int kP = ref::KCF_P;          // K1 output side, KCF_T + 2

/* Static storage, filled with loops: no heap, no memset. */
float    output_reference[kP * kP];
float    output_vectorized[kP * kP];
uint32_t table_i0[kP];                  // column table, filled by k1_rvv on every call
uint32_t table_i1[kP];
float    table_fx[kP];

/* Frame f of kcf_seq_data into an image of the same size. */
void LoadFrame(Image<uint8_t, ImageType::GRAY> &image, int f)
{
    const uint8_t *src = kcf_seq_data + (long)f * kcf_seq_width * kcf_seq_height;
    for (int y = 0; y < kcf_seq_height; y++)
    {
        uint8_t *row = image.GetPtr(0, y);
        for (int x = 0; x < kcf_seq_width; x++)
            row[x] = src[y * kcf_seq_width + x];
    }
}

/* Box of frame f (centre and size): the init box for frame 1, then the ground truth
   (x, y, w, h), or the init box again when the sequence has no ground truth. */
ref::KcfBox BoxOf(int f)
{
    const float *g = (f == 0 || !kcf_seq_has_gt) ? kcf_seq_init : kcf_seq_gt[f];
    return ref::KcfBox{g[0] + g[2] / 2, g[1] + g[3] / 2, g[2], g[3]};
}

/* The window K1 cuts around a box: the same formula as roi_for() in KCF_flow.cpp. */
ref::Roi RoiFor(const ref::KcfBox &b)
{
    const float grow = (float)ref::KCF_T / (float)(ref::KCF_T - 2 * ref::KCF_CELL);
    const float padding = ref::KCF_DEFAULT_PARAMS.padding;
    ref::Roi r;
    r.cx = b.cx;
    r.cy = b.cy;
    r.tw = b.w * grow * padding / 2.5f;   /* K1 multiplies by 2.5 itself */
    r.th = b.h * grow * padding / 2.5f;
    return r;
}

/* k1_rvv does the same FMA sequence as k1_scalar, so the outputs must be bit-exact. */
bool CheckCorrectnessK1(const float *reference, const float *vectorized, int frame, const char *call)
{
    for (int i = 0; i < kP * kP; ++i)
    {
        if (!(reference[i] == vectorized[i]))
        {
            printf("  frame %d, %s: mismatch at row %d, col %d\n", frame + 1, call, i / kP, i % kP);
            return false;
        }
    }
    return true;
}

/* One timed call of each version for one window; the counts are added to the totals. */
bool TimeK1(const Image<uint8_t, ImageType::GRAY> &frame, const ref::Roi &roi, int f, const char *call,
            uint64_t &reference_cycles, uint64_t &reference_instrs,
            uint64_t &vectorized_cycles, uint64_t &vectorized_instrs)
{
    ref::k1_scalar(frame, roi, ref::KCF_T, output_reference);       /* untimed: same cache state */

    Timer timer_scalar, timer_vectorized;

    timer_scalar.Start();
    ref::k1_scalar(frame, roi, ref::KCF_T, output_reference);
    timer_scalar.Stop();

    timer_vectorized.Start();
    vec::k1_rvv(frame, roi, ref::KCF_T, output_vectorized, table_i0, table_i1, table_fx);
    timer_vectorized.Stop();

    reference_cycles += timer_scalar.ElapsedCycles();
    reference_instrs += timer_scalar.ElapsedInstructions();
    vectorized_cycles += timer_vectorized.ElapsedCycles();
    vectorized_instrs += timer_vectorized.ElapsedInstructions();

    return CheckCorrectnessK1(output_reference, output_vectorized, f, call);
}

}  // namespace
#endif

int main()
{
#ifdef RISCV_BAREMETAL
    // Enable the vector unit before any auto-vectorized loop runs (nexus-am's
    // _start only enables the FPU).
    riscv_enable_vector();
#endif

#ifndef KCF_HAVE_SEQ
    printf("images/kcf_seq.hpp not found. Make it on the host with:\n"
           "  python3 scripts/kcf_sequence.py SEQ_DIR --header images/kcf_seq.hpp --frames 3\n");
    printf("Output is wrong!\n");
    return 1;
#else
    const int width = kcf_seq_width;
    const int height = kcf_seq_height;
    const int flows = kcf_seq_frames - 1;     // frame 1 only trains, as in the flow

    printf("K1 benchmark: %s, %dx%d, %d timed frames, T = %d (output %dx%d), 2 calls per frame\n",
           kcf_seq_name, width, height, flows, ref::KCF_T, kP, kP);
    if (flows < 1)
    {
        printf("kcf_seq.hpp needs at least 2 frames (--frames 2 or more)\n");
        printf("Output is wrong!\n");
        return 1;
    }

    Image<uint8_t, ImageType::GRAY> frame(width, height);

    // Frame 1: the flow's first train is one K1 call at the init box. Not timed.
    LoadFrame(frame, 0);
    ref::KcfBox previous = BoxOf(0);
    ref::k1_scalar(frame, RoiFor(previous), ref::KCF_T, output_reference);
    vec::k1_rvv(frame, RoiFor(previous), ref::KCF_T, output_vectorized, table_i0, table_i1, table_fx);

    // Frames 2..n: detect at the previous box, train at the new box, both versions timed.
    uint64_t reference_cycles = 0, vectorized_cycles = 0;
    uint64_t reference_instrs = 0, vectorized_instrs = 0;
    bool all_correct = true;

    for (int f = 1; f <= flows && all_correct; ++f)
    {
        LoadFrame(frame, f);
        const ref::KcfBox current = BoxOf(f);

        uint64_t rc = 0, ri = 0, vc = 0, vi = 0;
        all_correct = TimeK1(frame, RoiFor(previous), f, "detect", rc, ri, vc, vi)
                   && TimeK1(frame, RoiFor(current), f, "train", rc, ri, vc, vi);

        printf("frame %d: scalar %llu cycles, %llu instructions | vectorized %llu cycles, %llu instructions\n",
               f + 1, (unsigned long long)rc, (unsigned long long)ri,
               (unsigned long long)vc, (unsigned long long)vi);

        reference_cycles += rc;
        reference_instrs += ri;
        vectorized_cycles += vc;
        vectorized_instrs += vi;
        previous = current;
    }

    // Per frame = 2 calls, so num_pixels = 2 x P x P output pixels.
    PrintTime("K1 per frame (2 calls)", 2 * kP * kP,
              vectorized_cycles / flows, vectorized_instrs / flows,
              reference_cycles / flows, reference_instrs / flows);

    if (all_correct)
    {
        printf("Output is correct.\n");
        return 0;
    }
    printf("Output is wrong!\n");
    return 1;
#endif
}
