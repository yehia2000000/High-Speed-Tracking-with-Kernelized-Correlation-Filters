/*
 * KCF benchmark: cycles and instructions of the K1–K6 flow (ref::kcf_flow), per kernel and
 * in total, on real frames.
 *
 * Frames: images/kcf_seq.hpp, made on the host with
 *     python3 scripts/kcf_sequence.py SEQ_DIR --header images/kcf_seq.hpp --frames 3
 * Frame 1 trains the model (ref::kcf_flow_init, not timed). Every next frame runs one full
 * flow (detect + train) under the timer. Nothing else is measured: no box output, no drawing.
 *
 * gem5: init + 2 flows are about 78 M instructions, more than XS-GEM5's default --maxinsts
 * (40 M). Configure with -DGEM5_EXTRA_ARGS="--disable-difftest;--maxinsts=2000000000".
 * qemu-user: same code, a quick functional check; its cycle counters are host ticks.
 */
#include "benchmark.hpp"
#include "KCF_flow.hpp"
#include "riscv_math.hpp"   // sqrtf

#if __has_include("kcf_seq.hpp")
#include "kcf_seq.hpp"
#define KCF_HAVE_SEQ 1
#endif

#ifdef KCF_HAVE_SEQ
namespace
{
/* About 1 MB: static storage (.bss), not the 32 KB bare-metal stack. */
ref::KcfState g_state;

/* Frame f of kcf_seq_data into an image of the same size. */
void LoadFrame(Image<uint8_t> &image, int f)
{
    const uint8_t *src = kcf_seq_data + (long)f * kcf_seq_width * kcf_seq_height;
    for (int y = 0; y < kcf_seq_height; y++)
    {
        uint8_t *row = image.GetPtr(0, y);
        for (int x = 0; x < kcf_seq_width; x++)
            row[x] = src[y * kcf_seq_width + x];
    }
}

/* Distance between the box centre and the ground-truth centre (x, y, w, h). */
float CentreError(const ref::KcfBox &b, const float *gt)
{
    const float dx = b.cx - (gt[0] + gt[2] / 2), dy = b.cy - (gt[1] + gt[3] / 2);
    return sqrtf(dx * dx + dy * dy);
}

void PrintFlowTime(const ref::KcfFlowStats &stats, int flows,
                   uint64_t total_cycles, uint64_t total_instrs)
{
    printf("\n%-18s %14s %14s %6s %8s\n", "kernel", "cycles/frame", "instr/frame", "calls", "% flow");
    uint64_t sum_cycles = 0, sum_instrs = 0;
    for (int k = 0; k < ref::KCF_STAGES; k++)
    {
        const ref::KcfStageCount &c = stats.stage[k];
        sum_cycles += c.cycles;
        sum_instrs += c.instructions;
        printf("%-18s %14llu %14llu %6llu %7.1f%%\n", ref::kcf_stage_name(k),
               (unsigned long long)(c.cycles / flows), (unsigned long long)(c.instructions / flows),
               (unsigned long long)(c.calls / flows),
               total_cycles ? 100.0 * (double)c.cycles / (double)total_cycles : 0.0);
    }
    /* box move, timer reads and calls between the stages */
    const uint64_t other_cycles = total_cycles > sum_cycles ? total_cycles - sum_cycles : 0;
    const uint64_t other_instrs = total_instrs > sum_instrs ? total_instrs - sum_instrs : 0;
    printf("%-18s %14llu %14llu %6s %7.1f%%\n", "other",
           (unsigned long long)(other_cycles / flows), (unsigned long long)(other_instrs / flows), "",
           total_cycles ? 100.0 * (double)other_cycles / (double)total_cycles : 0.0);
    printf("%-18s %14llu %14llu\n", "flow total",
           (unsigned long long)(total_cycles / flows), (unsigned long long)(total_instrs / flows));
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
    const int flows = kcf_seq_frames - 1;     // frame 1 only trains

    printf("KCF benchmark: %s, %dx%d, %d timed frames, template %d px, %dx%d cells, %d channels\n",
           kcf_seq_name, width, height, flows, ref::KCF_T, ref::KCF_NF, ref::KCF_NF, ref::KCF_C);
    if (flows < 1)
    {
        printf("kcf_seq.hpp needs at least 2 frames (--frames 2 or more)\n");
        printf("Output is wrong!\n");
        return 1;
    }

    Image<uint8_t> frame(width, height);
    ref::kcf_setup(g_state, ref::KCF_DEFAULT_PARAMS);

    // Frame 1: the first train. Not timed.
    LoadFrame(frame, 0);
    ref::KcfBox box = {kcf_seq_init[0] + kcf_seq_init[2] / 2, kcf_seq_init[1] + kcf_seq_init[3] / 2,
                       kcf_seq_init[2], kcf_seq_init[3]};
    ref::kcf_flow_init(g_state, frame, box);

    // Frames 2..n: the flow under the timer, per kernel (stats) and in total (timer).
    ref::KcfFlowStats stats = {};
    uint64_t total_cycles = 0, total_instrs = 0;
    int within = 0;

    for (int f = 1; f <= flows; ++f)
    {
        LoadFrame(frame, f);

        Timer timer;
        timer.Start();
        box = ref::kcf_flow(g_state, frame, box, &stats);
        timer.Stop();

        total_cycles += timer.ElapsedCycles();
        total_instrs += timer.ElapsedInstructions();

        float err = kcf_seq_has_gt ? CentreError(box, kcf_seq_gt[f]) : 0.0f;
        if (err <= 20.0f)
            within++;
        printf("frame %d: centre %.2f %.2f, %llu cycles, %llu instructions",
               f + 1, box.cx, box.cy, (unsigned long long)timer.ElapsedCycles(),
               (unsigned long long)timer.ElapsedInstructions());
        if (kcf_seq_has_gt)
            printf(", %.2f px from ground truth", err);
        printf("\n");
    }

    PrintFlowTime(stats, flows, total_cycles, total_instrs);

    // Correct = every timed frame within 20 px of the ground truth (KCF's precision threshold).
    if (within == flows)
    {
        printf("Output is correct.\n");
        return 0;
    }
    printf("Output is wrong! (%d of %d frames within 20 px)\n", within, flows);
    return 1;
#endif
}
