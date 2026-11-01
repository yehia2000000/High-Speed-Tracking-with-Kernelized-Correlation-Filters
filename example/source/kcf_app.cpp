/*
 * kcf_app: the scalar KCF tracker (K1–K6, example/source/kcf_scalar.cpp) on a frame sequence.
 *
 * qemu-user (kcf_app_qemu): reads a PGM sequence folder made by scripts/kcf_sequence.py --pgm
 *     images.txt       one .pgm path per line
 *     region.txt       box of frame 1: x,y,w,h (or the 8-number VOT polygon)
 *     groundtruth.txt  optional, x,y,w,h per frame
 *   The folder is argv[1], else KCF_SEQ_DIR (set at configure time with -DKCF_SEQ_DIR=...).
 *   Writes <folder>/output_scalar.txt (x,y,w,h per frame) for eval.py and make_video.py.
 *
 * gem5 (kcf_app_gem5): no file I/O on bare metal, so the frames are compiled in from
 *   images/kcf_seq.hpp (scripts/kcf_sequence.py --header). Keep it to a few frames: one
 *   update() is tens of millions of instructions.
 *
 * Prints per frame the box, the centre error (with ground truth) and the cycles of update(),
 * then the cycles and instructions per kernel, and "Example passed." when precision@20px >= 80%.
 */
#include <cstdint>
#include <new>

#include "image.hpp"
#include "riscv_math.hpp"
#include "riscv_port.hpp"
#include "riscv_timer.hpp"
#include "kcf_scalar.hpp"
#include "kcf_prof.hpp"

#ifdef RISCV_QEMU
#include <cstring>
#ifndef KCF_SEQ_DIR
#define KCF_SEQ_DIR "."
#endif
#elif __has_include("kcf_seq.hpp")
#include "kcf_seq.hpp"
#define KCF_HAVE_SEQ 1
#endif

namespace {

constexpr int   MAX_FRAMES = 4096;
constexpr float PASS_PRECISION = 0.8f;      /* share of frames within 20 px */

float g_gt[MAX_FRAMES][4];                  /* ground truth, x,y,w,h */

/* The tracker is about 1 MB: static storage, not the 32 KB bare-metal stack. It is built with
   placement new because nexus-am never runs C++ static constructors. */
alignas(kcf::Tracker) unsigned char g_tracker_mem[sizeof(kcf::Tracker)];

kcf::Box box_from_xywh(const float *v) { return kcf::Box{v[0] + v[2] / 2, v[1] + v[3] / 2, v[2], v[3]}; }

float centre_error(const kcf::Box &b, const float *gt) {
    float dx = b.cx - (gt[0] + gt[2] / 2), dy = b.cy - (gt[1] + gt[3] / 2);
    return sqrtf(dx * dx + dy * dy);
}

#ifdef RISCV_QEMU
/* All numbers in a line, separated by commas and/or spaces. */
int parse_floats(const char *s, float *v, int max) {
    int n = 0;
    while (*s && n < max) {
        char *end;
        float x = strtof(s, &end);
        if (end == s) { s++; continue; }
        v[n++] = x;
        s = end;
    }
    return n;
}

/* 4 numbers: x,y,w,h. 8 numbers: polygon, take its bounding box. */
bool to_xywh(const float *v, int n, float *out) {
    if (n >= 8) {
        float x0 = v[0], x1 = v[0], y0 = v[1], y1 = v[1];
        for (int i = 2; i < 8; i += 2) {
            x0 = fminf(x0, v[i]); x1 = fmaxf(x1, v[i]);
            y0 = fminf(y0, v[i + 1]); y1 = fmaxf(y1, v[i + 1]);
        }
        out[0] = x0; out[1] = y0; out[2] = x1 - x0; out[3] = y1 - y0;
        return true;
    }
    if (n >= 4) { for (int i = 0; i < 4; i++) out[i] = v[i]; return true; }
    return false;
}

void chomp(char *s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) s[--n] = 0;
}
#endif

}  /* namespace */

#ifdef RISCV_QEMU
int main(int argc, char **argv)
#else
int main()
#endif
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif
    kcf::Params prm;                         /* paper parameters, K2_DOT (see kcf_scalar.hpp) */
    kcf::Tracker *tracker = new (g_tracker_mem) kcf::Tracker(prm);
    printf("KCF scalar tracker: template %d px, %d x %d cells, %d channels, %d FFT/frame\n",
           kcf::T, kcf::NF, kcf::NF, kcf::C, 67);

    /* ---------------- the sequence ---------------- */
    float init[4] = {0, 0, 0, 0};
    int n_gt = 0;
    Image<uint8_t> img;

#ifdef RISCV_QEMU
    const char *dir = (argc > 1) ? argv[1] : KCF_SEQ_DIR;
    char path[1024], line[1024];
    snprintf(path, sizeof path, "%s/images.txt", dir);
    FILE *list = fopen(path, "r");
    if (!list) {
        printf("cannot open %s\nMake the sequence with: python3 scripts/kcf_sequence.py SEQ_DIR --pgm %s\n", path, dir);
        printf("Example failed\n");
        return 1;
    }
    snprintf(path, sizeof path, "%s/region.txt", dir);
    FILE *rf = fopen(path, "r");
    float v[8];
    if (!rf || !fgets(line, sizeof line, rf) || !to_xywh(v, parse_floats(line, v, 8), init)) {
        printf("cannot read the first box from %s\nExample failed\n", path);
        return 1;
    }
    fclose(rf);
    snprintf(path, sizeof path, "%s/groundtruth.txt", dir);
    if (FILE *gf = fopen(path, "r")) {
        while (n_gt < MAX_FRAMES && fgets(line, sizeof line, gf))
            if (to_xywh(v, parse_floats(line, v, 8), g_gt[n_gt])) n_gt++;
        fclose(gf);
    }
    snprintf(path, sizeof path, "%s/output_scalar.txt", dir);
    FILE *out = fopen(path, "w");
    printf("sequence: %s (%d ground-truth boxes)\n", dir, n_gt);
#else
#ifndef KCF_HAVE_SEQ
    printf("images/kcf_seq.hpp not found. Make it on the host with:\n"
           "  python3 scripts/kcf_sequence.py SEQ_DIR --header images/kcf_seq.hpp --frames 3\n"
           "Example failed\n");
    return 1;
#else
    for (int i = 0; i < 4; i++) init[i] = kcf_seq_init[i];
    n_gt = kcf_seq_has_gt ? kcf_seq_frames : 0;
    for (int f = 0; f < n_gt && f < MAX_FRAMES; f++)
        for (int i = 0; i < 4; i++) g_gt[f][i] = kcf_seq_gt[f][i];
    printf("sequence: %s, %d frames of %d x %d compiled in\n", kcf_seq_name, kcf_seq_frames,
           kcf_seq_width, kcf_seq_height);
#endif
#endif

    /* ---------------- track ---------------- */
    int frames = 0, within = 0, scored = 0;
    float err_sum = 0;
    uint64_t upd_cycles = 0, upd_instret = 0;
    for (;;) {
#ifdef RISCV_QEMU
        if (!fgets(line, sizeof line, list)) break;
        chomp(line);
        if (!line[0]) continue;
        img.Read(line);
#else
        if (frames >= kcf_seq_frames) break;
        img.Read(kcf_seq_width, kcf_seq_height,
                 kcf_seq_data + (long)frames * kcf_seq_width * kcf_seq_height);
#endif
        kcf::Box b;
        uint64_t cyc = 0;
        if (frames == 0) {
            b = box_from_xywh(init);
            tracker->init(img, b);
#ifdef KCF_PROFILE
            kprof::reset();                   /* profile update() only */
#endif
        } else {
            CycleSample s0 = Sample();
            b = tracker->update(img);
            CycleSample d = Delta(Sample(), s0);
            cyc = d.cycles;
            upd_cycles += d.cycles;
            upd_instret += d.instructions;
        }
        float x = b.cx - b.w / 2, y = b.cy - b.h / 2;
#ifdef RISCV_QEMU
        if (out) fprintf(out, "%.3f,%.3f,%.3f,%.3f\n", x, y, b.w, b.h);
#endif
        if (frames < n_gt) {
            float e = centre_error(b, g_gt[frames]);
            err_sum += e;
            scored++;
            if (e <= 20.0f) within++;
            printf("frame %4d  box %7.2f %7.2f %6.2f %6.2f  err %6.2f px  update %llu cycles\n",
                   frames + 1, x, y, b.w, b.h, e, (unsigned long long)cyc);
        } else {
            printf("frame %4d  box %7.2f %7.2f %6.2f %6.2f  update %llu cycles\n",
                   frames + 1, x, y, b.w, b.h, (unsigned long long)cyc);
        }
        frames++;
    }
#ifdef RISCV_QEMU
    fclose(list);
    if (out) { fclose(out); printf("wrote %s/output_scalar.txt\n", dir); }
#endif

    /* ---------------- summary ---------------- */
    const int updates = frames > 1 ? frames - 1 : 1;
    printf("\nframes %d, update() mean %llu cycles, %llu instructions\n", frames,
           (unsigned long long)(upd_cycles / updates), (unsigned long long)(upd_instret / updates));
    float precision = scored ? (float)within / (float)scored : 1.0f;
    if (scored)
        printf("precision@20px %.1f %%, mean centre error %.2f px\n", 100.0f * precision, err_sum / scored);

#ifdef KCF_PROFILE
    printf("\n%-18s %14s %14s %8s %8s\n", "kernel", "cycles/frame", "instr/frame", "calls", "% upd");
    for (int r = 0; r < kprof::N_REGIONS; r++) {
        const kprof::Slot &s = kprof::table()[r];
        printf("%-18s %14llu %14llu %8llu %7.1f%%\n", kprof::name(r),
               (unsigned long long)(s.cycles / updates), (unsigned long long)(s.instret / updates),
               (unsigned long long)s.calls, upd_cycles ? 100.0 * (double)s.cycles / (double)upd_cycles : 0.0);
    }
#endif

    if (frames >= 2 && precision >= PASS_PRECISION) {
        printf("Example passed.\n");
        return 0;
    }
    printf("Example failed\n");
    return 1;
}
