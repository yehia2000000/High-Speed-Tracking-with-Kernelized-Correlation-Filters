#include "KCF_flow.hpp"
#include "riscv_math.hpp"    // cosf; <cmath> is not available on bare metal
#include "riscv_timer.hpp"   // Timer: cycle / instret CSRs

#ifndef KCF_K1_VEC
#define KCF_K1_VEC 1  // 0 ref::k1_scalar   1 vec::k1_rvv
#endif
#ifndef KCF_K2_VEC
#define KCF_K2_VEC 1   // 0 ref::k2_scalar   1 vec::k2_rvv (method from KcfParams::k2)
#endif
#ifndef KCF_K3_VEC
#define KCF_K3_VEC 1   // 0 ref::k3_scalar (interleaved out)   1 vec::Histogram_k3 (planar out)
#endif
#ifndef KCF_K4_VEC
#define KCF_K4_VEC 1   // 0 ref::k4_normalization_hanning_scalar (interleaved in / out)
                       // 1 vec::k4_normalization_hanning (planar in / out, writes s.fr: no layout)
#endif
#ifndef KCF_K5_VEC
#define KCF_K5_VEC 1   // 0 ref::k5_spectra / k5_scalar_detect / k5_scalar_train / k5_make_labels
                       // 1 vec::k5_spectra / k5_detect / k5_train / k5_make_labels
#endif
#ifndef KCF_K6_VEC
#define KCF_K6_VEC 1   // 0 ref::k6_detection   1 vec::k6_detection
#endif

#if (KCF_K1_VEC != 0 && KCF_K1_VEC != 1) || (KCF_K2_VEC != 0 && KCF_K2_VEC != 1) || \
    (KCF_K3_VEC != 0 && KCF_K3_VEC != 1) || (KCF_K4_VEC != 0 && KCF_K4_VEC != 1) || \
    (KCF_K5_VEC != 0 && KCF_K5_VEC != 1) || (KCF_K6_VEC != 0 && KCF_K6_VEC != 1)
#error "every KCF_Kn_VEC must be 0 or 1"
#endif

#if KCF_K1_VEC
#include "K1_rvv.hpp"        // vec::k1_rvv
#endif
#if KCF_K2_VEC
#include "K2_rvv.hpp"        // vec::k2_rvv
#endif
#if KCF_K3_VEC
#include "K3_rvv.hpp"        // vec::Histogram_k3
#endif
#if KCF_K4_VEC
#include "K4_rvv.hpp"        // vec::k4_normalization_hanning
#endif
#if KCF_K5_VEC
#include "K5_rvv.hpp"        // vec::k5_spectra, vec::k5_detect, vec::k5_train, vec::k5_make_labels
#endif
#if KCF_K6_VEC
#include "K6_rvv.hpp"        // vec::k6_detection
#endif

namespace
{
#if KCF_K1_VEC
/* scratch of vec::k1_rvv: column tables (one entry per output column) */
uint32_t g_k1_i0[ref::KCF_P], g_k1_i1[ref::KCF_P];
float    g_k1_fx[ref::KCF_P];
#endif

/* K3 and K4 have two layouts. Scalar: interleaved, the bins of a cell next to each other,
   a[cell * channels + ch]. Vector: planar, one plane per channel, a[ch * cells + cell].
   When K3 and K4 differ, the layout stage converts s.hist; s.fhog is the scratch (a planar
   K3 histogram, 34 * 34 * 27 floats, fits in it). */
static_assert(sizeof(ref::KcfState::fhog) >= sizeof(ref::KcfState::hist), "s.fhog holds a K3 histogram");

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

/* Runs one stage. With stats, it is timed like a kernel in Add_benchmark (Timer around the
   call) and added to its row; without stats it is just the call. */
template <typename Kernel>
inline void run_stage(ref::KcfFlowStats *stats, ref::KcfStage stage, Kernel &&kernel)
{
    if (stats == nullptr)
    {
        kernel();
        return;
    }
    Timer timer;
    timer.Start();
    kernel();
    timer.Stop();
    stats->stage[stage].cycles       += timer.ElapsedCycles();
    stats->stage[stage].instructions += timer.ElapsedInstructions();
    stats->stage[stage].calls++;
}

/* The K1 window for a box. K1 maps 2.5 x target onto T pixels and K4 keeps only the inner
   T - 2*CELL, so the window is widened by T / (T - 2*CELL): the features then cover exactly
   2.5 x target, as KCFcpp does with its + 8 pixels. */
ref::Roi roi_for(const ref::KcfParams &p, ref::KcfBox b)
{
    const float grow = (float)ref::KCF_T / (float)(ref::KCF_T - 2 * ref::KCF_CELL);
    ref::Roi r;
    r.cx = b.cx;
    r.cy = b.cy;
    r.tw = b.w * grow * p.padding / 2.5f;   /* K1 multiplies by 2.5 itself */
    r.th = b.h * grow * p.padding / 2.5f;
    return r;
}

/* K1 -> K2 -> K3 -> K4 -> layout -> K5 spectra, at box b. Leaves the spectra in s.fr / s.fi. */
void features(ref::KcfState &s, const Image<uint8_t, ImageType::GRAY> &frame, ref::KcfBox b,
              ref::KcfFlowStats *stats)
{
    using namespace ref;
    const Roi r = roi_for(s.prm, b);
    run_stage(stats, KCF_K1, [&] {
#if KCF_K1_VEC
        vec::k1_rvv(frame, r, KCF_T, s.k1, g_k1_i0, g_k1_i1, g_k1_fx);
#else
        k1_scalar(frame, r, KCF_T, s.k1);
#endif
    });
    run_stage(stats, KCF_K2, [&] {
#if KCF_K2_VEC
        vec::k2_rvv(s.k1, KCF_T, s.mag, s.bin, s.prm.k2);
#else
        k2_scalar(s.k1, KCF_T, s.mag, s.bin, s.prm.k2);
#endif
    });
    run_stage(stats, KCF_K3, [&] {
#if KCF_K3_VEC && KCF_K4_VEC
        vec::Histogram_k3(s.bin, s.mag, s.hist, KCF_T, KCF_T);          /* planar         */
#elif KCF_K3_VEC
        vec::Histogram_k3(s.bin, s.mag, s.fhog, KCF_T, KCF_T);          /* planar, scratch */
#else
        k3_scalar<KCF_CELL>(s.bin, s.mag, s.hist, KCF_T, KCF_T);        /* interleaved    */
#endif
    });
#if KCF_K3_VEC && !KCF_K4_VEC   /* planar K3 -> interleaved for the scalar K4 */
    run_stage(stats, KCF_LAYOUT, [&] { to_interleaved(s.fhog, s.hist, KCF_G * KCF_G, 27); });
#elif !KCF_K3_VEC && KCF_K4_VEC /* interleaved K3 -> planar for the vector K4 */
    run_stage(stats, KCF_LAYOUT, [&] { to_planar(s.hist, s.fhog, KCF_G * KCF_G, 27); });
#endif
    run_stage(stats, KCF_K4, [&] {
#if KCF_K4_VEC && KCF_K3_VEC
        vec::k4_normalization_hanning(s.hist, s.hann, s.fr, KCF_G, KCF_G, s.prm.alpha);
#elif KCF_K4_VEC
        vec::k4_normalization_hanning(s.fhog, s.hann, s.fr, KCF_G, KCF_G, s.prm.alpha);
#else
        k4_normalization_hanning_scalar(s.hist, s.hann, s.fhog, KCF_G, KCF_G, s.prm.alpha);
#endif
    });
#if !KCF_K4_VEC
    /* The scalar K4 writes the 31 features of a cell next to each other (interleaved); K5 needs
       one plane per channel. The same job as the .t() in KCFcpp's getFeatures. The vector K4
       writes the planes into s.fr itself, so there this stage does not run. */
    run_stage(stats, KCF_LAYOUT, [&] { to_planar(s.fhog, s.fr, KCF_N, KCF_C); });
#endif
    run_stage(stats, KCF_K5_FFT, [&] {   /* 31 forward FFTs, in place */
#if KCF_K5_VEC
        vec::k5_spectra(s.fr, KCF_C, KCF_NF, KCF_NF, s.fr, s.fi);
#else
        k5_spectra(s.fr, KCF_C, KCF_NF, KCF_NF, s.fr, s.fi);
#endif
    });
}
}  // namespace

const char *ref::kcf_stage_name(int stage)
{
    static const char *const names[KCF_STAGES] = {
        "K1_subwindow", "K2_gradient", "K3_histogram", "K4_norm_pca_hann", "layout_to_planar",
        "K5_spectra_31fft", "K5_detect", "K6_peak", "K5_train"};
    return (stage >= 0 && stage < KCF_STAGES) ? names[stage] : "?";
}

const char *ref::kcf_stage_version(int stage)
{
    static const char *const version[KCF_STAGES] = {
        KCF_K1_VEC ? "vec" : "scalar", KCF_K2_VEC ? "vec" : "scalar",
        KCF_K3_VEC ? "vec" : "scalar", KCF_K4_VEC ? "vec" : "scalar",
        (KCF_K3_VEC && KCF_K4_VEC) ? "none" : (KCF_K3_VEC && !KCF_K4_VEC) ? "scalar x2" : "scalar",
        KCF_K5_VEC ? "vec" : "scalar", KCF_K5_VEC ? "vec" : "scalar",
        KCF_K6_VEC ? "vec" : "scalar", KCF_K5_VEC ? "vec" : "scalar"};
    return (stage >= 0 && stage < KCF_STAGES) ? version[stage] : "?";
}

void ref::kcf_setup(KcfState &s, const KcfParams &p)
{
    s.prm = p;
    /* Hann on the K4 grid, symmetric, edges exactly 0 (N - 1 denominator, as KCFcpp) */
    for (int i = 0; i < KCF_NF; i++)
        for (int j = 0; j < KCF_NF; j++)
        {
            float hy = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * i / (KCF_NF - 1)));
            float hx = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * j / (KCF_NF - 1)));
            s.hann[i * KCF_NF + j] = hy * hx;
        }
#if KCF_K5_VEC
    vec::k5_make_labels(KCF_NF, KCF_NF, p.padding, p.out_sigma_factor, s.yr, s.yi);
#else
    k5_make_labels(KCF_NF, KCF_NF, p.padding, p.out_sigma_factor, s.yr, s.yi);
#endif

    s.model.rows = KCF_NF;
    s.model.cols = KCF_NF;
    s.model.ch   = KCF_C;
    s.model.inv_sd = 1.0f / (p.sigma * p.sigma * (float)KCF_N * (float)KCF_C);
    s.model.lambda = p.lambda;
    s.model.eta    = p.eta;
    s.model.xr = s.mxr; s.model.xi = s.mxi;
    s.model.ar = s.mar; s.model.ai = s.mai;
    s.model.yr = s.yr;  s.model.yi = s.yi;

    s.wk.sr = s.work;             s.wk.si = s.work + KCF_N;
    s.wk.kr = s.work + 2 * KCF_N; s.wk.ki = s.work + 3 * KCF_N;
    s.wk.ar = s.work + 4 * KCF_N; s.wk.ai = s.work + 5 * KCF_N;
    s.peak_x = 0.0f;
    s.peak_y = 0.0f;
}

void ref::kcf_flow_init(KcfState &s, const Image<uint8_t, ImageType::GRAY> &frame, KcfBox b,
                        KcfFlowStats *stats)
{
    features(s, frame, b, stats);
    run_stage(stats, KCF_K5_TRAIN, [&] {
#if KCF_K5_VEC
        vec::k5_train(&s.model, s.fr, s.fi, 1, s.prm.use_poly_exp, &s.wk);    /* first = 1 */
#else
        k5_scalar_train(&s.model, s.fr, s.fi, 1, s.prm.use_poly_exp, &s.wk);   /* first = 1 */
#endif
    });
}

ref::KcfBox ref::kcf_flow(KcfState &s, const Image<uint8_t, ImageType::GRAY> &frame, KcfBox b,
                          KcfFlowStats *stats)
{
    /* detect at the old centre */
    features(s, frame, b, stats);
    run_stage(stats, KCF_K5_DETECT, [&] {
#if KCF_K5_VEC
        vec::k5_detect(&s.model, s.fr, s.fi, s.prm.use_poly_exp, s.resp, &s.wk);
#else
        k5_scalar_detect(&s.model, s.fr, s.fi, s.prm.use_poly_exp, s.resp, &s.wk);
#endif
    });
    run_stage(stats, KCF_K6, [&] {   /* peak shift in template pixels */
#if KCF_K6_VEC
        vec::k6_detection(s.resp, KCF_NF, KCF_NF, KCF_CELL, s.peak_x, s.peak_y);
#else
        k6_detection(s.resp, KCF_NF, KCF_NF, KCF_CELL, s.peak_x, s.peak_y);
#endif
    });

    /* move the box: template pixels -> image pixels with K1's sample step, then keep the
       centre inside the frame */
    const Roi r = roi_for(s.prm, b);
    const float sx = 2.5f * r.tw / (float)KCF_T, sy = 2.5f * r.th / (float)KCF_T;
    b.cx += s.peak_x * sx;
    b.cy += s.peak_y * sy;
    if (b.cx < 0) b.cx = 0;
    if (b.cy < 0) b.cy = 0;
    if (b.cx > (float)(frame.Width() - 1))  b.cx = (float)(frame.Width() - 1);
    if (b.cy > (float)(frame.Height() - 1)) b.cy = (float)(frame.Height() - 1);

    /* train at the new centre */
    features(s, frame, b, stats);
    run_stage(stats, KCF_K5_TRAIN, [&] {
#if KCF_K5_VEC
        vec::k5_train(&s.model, s.fr, s.fi, 0, s.prm.use_poly_exp, &s.wk);
#else
        k5_scalar_train(&s.model, s.fr, s.fi, 0, s.prm.use_poly_exp, &s.wk);
#endif
    });
    return b;
}