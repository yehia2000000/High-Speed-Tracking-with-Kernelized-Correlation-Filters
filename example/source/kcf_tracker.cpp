/*
 * kcf_tracker (qemu-user only): the KCF tracker on a whole sequence, for accuracy and boxes.
 * The kernels are ref::kcf_flow (test/reference/source/KCF_flow.cpp); this file only reads the
 * frames, keeps the box, writes it out and scores it. Timing is in benchmark/source/KCF_benchmark.cpp.
 *
 * Input: a PGM folder made by scripts/kcf_sequence.py --pgm
 *     images.txt       one .pgm path per line
 *     region.txt       box of frame 1: x,y,w,h (or the 8-number VOT polygon)
 *     groundtruth.txt  optional, x,y,w,h per frame
 *   The folder is argv[1], else KCF_SEQ_DIR (configure with -DKCF_SEQ_DIR=...).
 *
 * Output: <folder>/output_scalar.txt (x,y,w,h per frame) for eval.py and make_video.py,
 *   the box and centre error per frame, precision@20px, and "Example passed." when >= 80 %.
 */
#ifndef RISCV_QEMU
#error "kcf_tracker reads files: it is the qemu-user application. On gem5 use KCF_benchmark."
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "image.hpp"
#include "riscv_math.hpp"
#include "KCF_flow.hpp"

#ifndef KCF_SEQ_DIR
#define KCF_SEQ_DIR "."
#endif

namespace
{
constexpr int   MAX_FRAMES = 4096;
constexpr float PASS_PRECISION = 0.8f;      // share of frames within 20 px

ref::KcfState g_state;                      // about 1 MB: static storage, not the stack
float g_gt[MAX_FRAMES][4];                  // ground truth, x,y,w,h

// All numbers in a line, separated by commas and/or spaces.
int ParseFloats(const char *s, float *v, int max)
{
    int n = 0;
    while (*s && n < max)
    {
        char *end;
        float x = strtof(s, &end);
        if (end == s) { s++; continue; }
        v[n++] = x;
        s = end;
    }
    return n;
}

// 4 numbers: x,y,w,h. 8 numbers: polygon, take its bounding box.
bool ToXywh(const float *v, int n, float *out)
{
    if (n >= 8)
    {
        float x0 = v[0], x1 = v[0], y0 = v[1], y1 = v[1];
        for (int i = 2; i < 8; i += 2)
        {
            x0 = fminf(x0, v[i]);     x1 = fmaxf(x1, v[i]);
            y0 = fminf(y0, v[i + 1]); y1 = fmaxf(y1, v[i + 1]);
        }
        out[0] = x0; out[1] = y0; out[2] = x1 - x0; out[3] = y1 - y0;
        return true;
    }
    if (n >= 4) { for (int i = 0; i < 4; i++) out[i] = v[i]; return true; }
    return false;
}

void Chomp(char *s)
{
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) s[--n] = 0;
}

float CentreError(const ref::KcfBox &b, const float *gt)
{
    const float dx = b.cx - (gt[0] + gt[2] / 2), dy = b.cy - (gt[1] + gt[3] / 2);
    return sqrtf(dx * dx + dy * dy);
}
}  // namespace

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : KCF_SEQ_DIR;
    char path[1024], line[1024];

    // ---------------- the sequence ----------------
    snprintf(path, sizeof path, "%s/images.txt", dir);
    FILE *list = fopen(path, "r");
    if (!list)
    {
        printf("cannot open %s\nMake the sequence with: python3 scripts/kcf_sequence.py SEQ_DIR --pgm %s\n", path, dir);
        printf("Example failed\n");
        return 1;
    }
    float init[4], v[8];
    snprintf(path, sizeof path, "%s/region.txt", dir);
    FILE *rf = fopen(path, "r");
    if (!rf || !fgets(line, sizeof line, rf) || !ToXywh(v, ParseFloats(line, v, 8), init))
    {
        printf("cannot read the first box from %s\nExample failed\n", path);
        return 1;
    }
    fclose(rf);
    int n_gt = 0;
    snprintf(path, sizeof path, "%s/groundtruth.txt", dir);
    if (FILE *gf = fopen(path, "r"))
    {
        while (n_gt < MAX_FRAMES && fgets(line, sizeof line, gf))
            if (ToXywh(v, ParseFloats(line, v, 8), g_gt[n_gt])) n_gt++;
        fclose(gf);
    }
    snprintf(path, sizeof path, "%s/output_scalar.txt", dir);
    FILE *out = fopen(path, "w");
    printf("KCF tracker (qemu): %s, %d ground-truth boxes\n", dir, n_gt);

    // ---------------- track ----------------
    ref::kcf_setup(g_state, ref::KCF_DEFAULT_PARAMS);
    ref::KcfBox box = {init[0] + init[2] / 2, init[1] + init[3] / 2, init[2], init[3]};
    Image<uint8_t> frame;
    int frames = 0, within = 0, scored = 0;
    float err_sum = 0;

    while (fgets(line, sizeof line, list))
    {
        Chomp(line);
        if (!line[0]) continue;
        frame.Read(line);

        if (frames == 0)
            ref::kcf_flow_init(g_state, frame, box);       // first frame: train only
        else
            box = ref::kcf_flow(g_state, frame, box);      // detect + train

        const float x = box.cx - box.w / 2, y = box.cy - box.h / 2;
        if (out) fprintf(out, "%.3f,%.3f,%.3f,%.3f\n", x, y, box.w, box.h);

        if (frames < n_gt)
        {
            const float e = CentreError(box, g_gt[frames]);
            err_sum += e;
            scored++;
            if (e <= 20.0f) within++;
            printf("frame %4d  box %7.2f %7.2f %6.2f %6.2f  err %6.2f px\n", frames + 1, x, y, box.w, box.h, e);
        }
        else
        {
            printf("frame %4d  box %7.2f %7.2f %6.2f %6.2f\n", frames + 1, x, y, box.w, box.h);
        }
        frames++;
    }
    fclose(list);
    if (out) { fclose(out); printf("wrote %s/output_scalar.txt\n", dir); }

    // ---------------- summary ----------------
    const float precision = scored ? (float)within / (float)scored : 1.0f;
    printf("\nframes %d\n", frames);
    if (scored)
        printf("precision@20px %.1f %%, mean centre error %.2f px\n", 100.0f * precision, err_sum / scored);
    if (frames >= 2 && precision >= PASS_PRECISION)
    {
        printf("Example passed.\n");
        return 0;
    }
    printf("Example failed\n");
    return 1;
}
