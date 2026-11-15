/*
 * kcf_video (qemu-user only): runs the KCF flow (ref::kcf_flow, with the RVV kernels selected
 * by the KCF_Kn_VEC macros in KCF_flow.cpp, all 1 by default) on a PGM sequence and draws the
 * detections into a video.
 *
 * The drawing is done here, on RISC-V: every frame is turned into RGB, the boxes and a few
 * lines of text are drawn on it, and the frames are appended to one raw rgb24 file. The host
 * only encodes that file (ffmpeg); scripts/kcf_video.sh does both steps.
 *
 * Usage (qemu-riscv64 -cpu max,vlen=128 kcf_video_qemu ...):
 *     kcf_video_qemu [SEQ_DIR] [--scales N] [--compare] [--out DIR] [--frames N] [--no-video]
 *   SEQ_DIR     PGM folder made by scripts/kcf_sequence.py --pgm (images.txt, region.txt,
 *               groundtruth.txt optional). Default KCF_SEQ_DIR.
 *   --scales N  scales per detect: 1 = the paper, 3 = multi-scale (default 3).
 *   --compare   also run a single-scale tracker on the same frames and draw it (yellow).
 *   --out DIR   where the outputs go (default SEQ_DIR).
 *   --frames N  stop after N frames.
 *   --no-video  boxes and scores only.
 *
 * Outputs, in DIR (<tag> = rvv_s<N>):
 *     boxes_<tag>.txt   x,y,w,h per frame (same format as groundtruth.txt)
 *     track_<tag>.csv   per frame: box, winning scale, K6 peak per scale, error, IoU
 *     video_<tag>.rgb   raw rgb24 frames, WIDTH x HEIGHT (printed at the end)
 * Video colours: red = ground truth, green = this tracker, yellow = single scale (--compare).
 *
 * "Example passed." when precision@20px >= 80 % (same rule as kcf_tracker).
 */
#ifndef RISCV_QEMU
#error "kcf_video reads and writes files: it is a qemu-user application."
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <riscv_vector.h>

#include "image.hpp"
#include "riscv_math.hpp"
#include "KCF_flow.hpp"

#ifndef KCF_SEQ_DIR
#define KCF_SEQ_DIR "."
#endif

namespace
{
constexpr int   MAX_FRAMES = 4096;
constexpr float PASS_PRECISION = 0.8f;

ref::KcfState g_main;      // the tracker asked for (--scales)
ref::KcfState g_single;    // --compare: the same flow with scales = 1
float g_gt[MAX_FRAMES][4];

// ---------------------------------------------------------------- parsing

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

bool Exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != nullptr;
}

/* images.txt may hold absolute paths from the machine that made it. If a path does not open,
   take its file name inside SEQ_DIR. */
bool ResolveFrame(const char *dir, const char *line, char *out, size_t size)
{
    snprintf(out, size, "%s", line);
    if (Exists(out)) return true;
    const char *base = strrchr(line, '/');
    base = base ? base + 1 : line;
    snprintf(out, size, "%s/%s", dir, base);
    return Exists(out);
}

// ---------------------------------------------------------------- scores

float CentreError(const ref::KcfBox &b, const float *gt)
{
    const float dx = b.cx - (gt[0] + gt[2] / 2), dy = b.cy - (gt[1] + gt[3] / 2);
    return sqrtf(dx * dx + dy * dy);
}

float Iou(const ref::KcfBox &b, const float *gt)
{
    const float ax0 = b.cx - b.w / 2, ay0 = b.cy - b.h / 2, ax1 = ax0 + b.w, ay1 = ay0 + b.h;
    const float bx0 = gt[0], by0 = gt[1], bx1 = gt[0] + gt[2], by1 = gt[1] + gt[3];
    const float iw = fminf(ax1, bx1) - fmaxf(ax0, bx0), ih = fminf(ay1, by1) - fmaxf(ay0, by0);
    if (iw <= 0 || ih <= 0) return 0.0f;
    const float inter = iw * ih;
    return inter / (b.w * b.h + gt[2] * gt[3] - inter);
}

struct Score
{
    int   scored = 0, within = 0, success = 0;
    float err_sum = 0, iou_sum = 0;
    void Add(float err, float iou)
    {
        scored++;
        err_sum += err;
        iou_sum += iou;
        if (err <= 20.0f) within++;
        if (iou > 0.5f) success++;
    }
    float Precision() const { return scored ? (float)within / scored : 1.0f; }
    void Print(const char *name) const
    {
        if (!scored) return;
        printf("%-16s precision@20px %5.1f %%   mean centre error %6.2f px   "
               "mean IoU %.3f   success (IoU > 0.5) %5.1f %%\n",
               name, 100.0f * Precision(), err_sum / scored, iou_sum / scored,
               100.0f * success / scored);
    }
};

// ---------------------------------------------------------------- drawing (rgb24)

struct Rgb { uint8_t r, g, b; };
constexpr Rgb RED = {235, 50, 40}, GREEN = {40, 230, 60}, YELLOW = {255, 210, 0},
              WHITE = {255, 255, 255}, BLACK = {0, 0, 0};

struct Canvas
{
    uint8_t *px = nullptr;
    int w = 0, h = 0;

    void Put(int x, int y, Rgb c)
    {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        uint8_t *p = px + 3 * (y * w + x);
        p[0] = c.r; p[1] = c.g; p[2] = c.b;
    }

    void Fill(int x0, int y0, int x1, int y1, Rgb c)   // [x0, x1) x [y0, y1)
    {
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++) Put(x, y, c);
    }

    void Rect(float fx, float fy, float fw, float fh, int t, Rgb c)
    {
        const int x0 = (int)lroundf(fx), y0 = (int)lroundf(fy);
        const int x1 = (int)lroundf(fx + fw), y1 = (int)lroundf(fy + fh);
        Fill(x0, y0, x1 + 1, y0 + t, c);           // top
        Fill(x0, y1 - t + 1, x1 + 1, y1 + 1, c);   // bottom
        Fill(x0, y0, x0 + t, y1 + 1, c);           // left
        Fill(x1 - t + 1, y0, x1 + 1, y1 + 1, c);   // right
    }

    void Cross(float fx, float fy, int r, Rgb c)
    {
        const int x = (int)lroundf(fx), y = (int)lroundf(fy);
        Fill(x - r, y, x + r + 1, y + 1, c);
        Fill(x, y - r, x + 1, y + r + 1, c);
    }
};

/* 5 x 7 glyphs, one byte per row (bit 4 = left column). Only the characters the overlay uses. */
const uint8_t *Glyph(char ch)
{
    static const uint8_t blank[7] = {0};
    static const struct { char c; uint8_t rows[7]; } font[] = {
        {'0', {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}}, {'1', {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}},
        {'2', {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}}, {'3', {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}},
        {'4', {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}}, {'5', {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}},
        {'6', {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}}, {'7', {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
        {'8', {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}}, {'9', {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}},
        {'.', {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}}, {'x', {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11}},
        {'-', {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}}, {'/', {0x01,0x01,0x02,0x04,0x08,0x10,0x10}},
        {'A', {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}}, {'C', {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}},
        {'E', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}}, {'F', {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
        {'G', {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}}, {'K', {0x11,0x12,0x14,0x18,0x14,0x12,0x11}},
        {'L', {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}}, {'M', {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}},
        {'N', {0x11,0x19,0x15,0x13,0x11,0x11,0x11}}, {'R', {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
        {'S', {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}}, {'T', {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
        {'V', {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}},
    };
    for (const auto &g : font)
        if (g.c == ch) return g.rows;
    return blank;
}

/* text with a black background strip; s = pixel size of one font dot */
void Text(Canvas &cv, int x, int y, const char *str, Rgb c, int s)
{
    const int n = (int)strlen(str);
    cv.Fill(x - s, y - s, x + n * 6 * s, y + 8 * s, BLACK);
    for (int i = 0; i < n; i++)
    {
        const uint8_t *rows = Glyph(str[i]);
        for (int r = 0; r < 7; r++)
            for (int col = 0; col < 5; col++)
                if (rows[r] & (0x10 >> col))
                    cv.Fill(x + (i * 6 + col) * s, y + r * s, x + (i * 6 + col + 1) * s, y + (r + 1) * s, c);
    }
}

void DrawBox(Canvas &cv, const ref::KcfBox &b, int t, Rgb c)
{
    cv.Rect(b.cx - b.w / 2, b.cy - b.h / 2, b.w, b.h, t, c);
    cv.Cross(b.cx, b.cy, 3, c);
}
}  // namespace

int main(int argc, char **argv)
{
    // ---------------- arguments ----------------
    const char *dir = KCF_SEQ_DIR, *out_dir = nullptr;
    int scales = 3, max_frames = MAX_FRAMES;
    bool compare = false, video = true;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--scales") && i + 1 < argc)      scales = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--compare"))                compare = true;
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)    out_dir = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-video"))               video = false;
        else if (argv[i][0] != '-')                            dir = argv[i];
        else { printf("unknown option %s\nExample failed\n", argv[i]); return 1; }
    }
    if (!out_dir) out_dir = dir;
    if (scales < 1 || scales > ref::KCF_MAX_SCALES || (scales & 1) == 0)
    {
        printf("--scales must be odd, 1..%d\nExample failed\n", ref::KCF_MAX_SCALES);
        return 1;
    }
    if (scales == 1) compare = false;   // it would be the same tracker twice

    // ---------------- the sequence ----------------
    char path[1024], line[1024], frame_path[1024];
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

    char tag[32];
    snprintf(tag, sizeof tag, "rvv_s%d", scales);
    snprintf(path, sizeof path, "%s/boxes_%s.txt", out_dir, tag);
    FILE *boxes = fopen(path, "w");
    snprintf(path, sizeof path, "%s/track_%s.csv", out_dir, tag);
    FILE *csv = fopen(path, "w");
    char video_path[1024];
    snprintf(video_path, sizeof video_path, "%s/video_%s.rgb", out_dir, tag);
    FILE *vid = video ? fopen(video_path, "wb") : nullptr;
    if (!boxes || !csv || (video && !vid))
    {
        printf("cannot write into %s\nExample failed\n", out_dir);
        return 1;
    }
    fprintf(csv, "frame,x,y,w,h,scale_index,size_change,size_vs_frame1");
    for (int i = 0; i < scales; i++) fprintf(csv, ",peak_s%d", i);
    fprintf(csv, ",err_px,iou\n");

    const int vlen = (int)__riscv_vsetvlmax_e8m1() * 8;
    printf("KCF video (qemu, VLEN %d): %s, %d ground-truth boxes, %d scale%s per detect%s\n",
           vlen, dir, n_gt, scales, scales > 1 ? "s" : "", compare ? ", + single scale" : "");
    printf("kernels:");
    int scalar_stages = 0;
    for (int k = 0; k < ref::KCF_STAGES; k++)
    {
        const char *ver = ref::kcf_stage_version(k);
        printf(" %s=%s", ref::kcf_stage_name(k), ver);
        if (!strcmp(ver, "scalar")) scalar_stages++;
    }
    printf("\n");
    if (scalar_stages)
        printf("note: %d stage(s) run the scalar kernel (KCF_Kn_VEC = 0 in KCF_flow.cpp)\n", scalar_stages);

    // ---------------- track ----------------
    ref::KcfParams p = ref::KCF_DEFAULT_PARAMS;
    p.scales = scales;
    ref::kcf_setup(g_main, p);
    if (compare) ref::kcf_setup(g_single, ref::KCF_DEFAULT_PARAMS);

    const ref::KcfBox first = {init[0] + init[2] / 2, init[1] + init[3] / 2, init[2], init[3]};
    ref::KcfBox box = first, box1 = first;
    Image<uint8_t> frame;
    Canvas cv;
    Score sc_main, sc_single;
    int frames = 0;
    const clock_t t0 = clock();

    while (frames < max_frames && fgets(line, sizeof line, list))
    {
        Chomp(line);
        if (!line[0]) continue;
        if (!ResolveFrame(dir, line, frame_path, sizeof frame_path))
        {
            printf("cannot open frame %s (also not in %s)\nExample failed\n", line, dir);
            return 1;
        }
        frame.Read(frame_path);

        if (frames == 0)
        {
            ref::kcf_flow_init(g_main, frame, box);
            if (compare) ref::kcf_flow_init(g_single, frame, box1);
        }
        else
        {
            box = ref::kcf_flow(g_main, frame, box);
            if (compare) box1 = ref::kcf_flow(g_single, frame, box1);
        }

        const float x = box.cx - box.w / 2, y = box.cy - box.h / 2;
        const float size_rel = box.w / first.w;
        const bool has_gt = frames < n_gt;
        float err = -1.0f, iou = -1.0f;
        if (has_gt)
        {
            err = CentreError(box, g_gt[frames]);
            iou = Iou(box, g_gt[frames]);
            sc_main.Add(err, iou);
            if (compare) sc_single.Add(CentreError(box1, g_gt[frames]), Iou(box1, g_gt[frames]));
        }

        fprintf(boxes, "%.3f,%.3f,%.3f,%.3f\n", x, y, box.w, box.h);
        fprintf(csv, "%d,%.3f,%.3f,%.3f,%.3f,%d,%.5f,%.4f", frames + 1, x, y, box.w, box.h,
                frames ? g_main.scale_index : 0, frames ? g_main.scale_factor : 1.0f, size_rel);
        for (int i = 0; i < scales; i++) fprintf(csv, ",%.5f", frames ? g_main.scale_peak[i] : 0.0f);
        fprintf(csv, ",%.3f,%.4f\n", err, iou);

        if (frames % 25 == 0 || frames < 3)
        {
            printf("frame %4d  box %7.2f %7.2f %6.2f %6.2f  size x%.3f", frames + 1, x, y, box.w, box.h, size_rel);
            if (has_gt) printf("  err %6.2f px  IoU %.3f", err, iou);
            printf("\n");
        }

        // ---- draw this frame ----
        if (vid)
        {
            const int W = frame.Width(), H = frame.Height();
            if (!cv.px)
            {
                cv.w = W;
                cv.h = H;
                cv.px = new uint8_t[(size_t)W * H * 3];
            }
            for (int yy = 0; yy < H; yy++)
            {
                const uint8_t *src = frame.GetPtr(0, yy);
                uint8_t *dst = cv.px + (size_t)3 * yy * W;
                for (int xx = 0; xx < W; xx++)
                    dst[3 * xx] = dst[3 * xx + 1] = dst[3 * xx + 2] = src[xx];
            }
            const int t = (W >= 640) ? 3 : 2, s = (W >= 640) ? 2 : 1;   // line width, font dot
            if (has_gt) cv.Rect(g_gt[frames][0], g_gt[frames][1], g_gt[frames][2], g_gt[frames][3], 1, RED);
            DrawBox(cv, box, t, GREEN);
            if (compare) DrawBox(cv, box1, 1, YELLOW);   // thinner and on top: visible where they overlap

            char txt[64];
            int ty = 2 * s;
            snprintf(txt, sizeof txt, "FRAME %d", frames + 1);
            Text(cv, 2 * s, ty, txt, WHITE, s);
            ty += 9 * s;
            snprintf(txt, sizeof txt, "RVV VLEN %d", vlen);
            Text(cv, 2 * s, ty, txt, WHITE, s);
            ty += 9 * s;
            snprintf(txt, sizeof txt, "%d SCALE%s x%.2f", scales, scales > 1 ? "S" : "", size_rel);
            Text(cv, 2 * s, ty, txt, GREEN, s);
            ty += 9 * s;
            if (compare)
            {
                snprintf(txt, sizeof txt, "1 SCALE x%.2f", box1.w / first.w);
                Text(cv, 2 * s, ty, txt, YELLOW, s);
                ty += 9 * s;
            }
            if (has_gt) Text(cv, 2 * s, ty, "GT", RED, s);

            fwrite(cv.px, 1, (size_t)W * H * 3, vid);
        }
        frames++;
    }
    fclose(list);
    fclose(boxes);
    fclose(csv);
    if (vid) fclose(vid);
    const double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;

    // ---------------- summary ----------------
    printf("\nframes %d, %.1f s under qemu (functional emulation, not hardware speed)\n", frames, secs);
    char name[32];
    snprintf(name, sizeof name, "%d scale%s", scales, scales > 1 ? "s" : "");
    sc_main.Print(name);
    if (compare) sc_single.Print("1 scale");
    printf("wrote %s/boxes_%s.txt and %s/track_%s.csv\n", out_dir, tag, out_dir, tag);
    if (vid)
    {
        printf("wrote %s: %d frames, rgb24, %dx%d\n", video_path, frames, cv.w, cv.h);
        printf("encode on the host:\n  ffmpeg -y -f rawvideo -pix_fmt rgb24 -s %dx%d -r 30 -i %s "
               "-vf \"scale=trunc(iw*2/2)*2:trunc(ih*2/2)*2:flags=neighbor\" -c:v libx264 -pix_fmt yuv420p -crf 18 %s/video_%s.mp4\n",
               cv.w, cv.h, video_path, out_dir, tag);
    }
    delete[] cv.px;

    if (frames >= 2 && sc_main.Precision() >= PASS_PRECISION)
    {
        printf("Example passed.\n");
        return 0;
    }
    printf("Example failed\n");
    return 1;
}
