#include "KCF_flow.hpp"
#include "riscv_math.hpp"    // cosf; <cmath> is not available on bare metal
#include "riscv_timer.hpp"   // Timer: cycle / instret CSRs

namespace
{
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
    run_stage(stats, KCF_K1, [&] { k1_scalar(frame, r, KCF_T, s.k1); });
    run_stage(stats, KCF_K2, [&] { k2_scalar(s.k1, KCF_T, s.mag, s.bin, s.prm.k2); });
    run_stage(stats, KCF_K3, [&] { k3_scalar<KCF_CELL>(s.bin, s.mag, s.hist, KCF_T, KCF_T); });
    run_stage(stats, KCF_K4, [&] {
        k4_normalization_hanning_scalar(s.hist, s.hann, s.fhog, KCF_G, KCF_G, s.prm.alpha);
    });
    /* K4 writes the 31 features of a cell next to each other (interleaved); K5 needs one plane
       per channel. The same job as the .t() in KCFcpp's getFeatures. */
    run_stage(stats, KCF_LAYOUT, [&] {
        for (int cell = 0; cell < KCF_N; cell++)
            for (int ch = 0; ch < KCF_C; ch++)
                s.fr[ch * KCF_N + cell] = s.fhog[cell * KCF_C + ch];
    });
    run_stage(stats, KCF_K5_FFT, [&] {   /* 31 forward FFTs, in place */
        k5_spectra(s.fr, KCF_C, KCF_NF, KCF_NF, s.fr, s.fi);
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
    k5_make_labels(KCF_NF, KCF_NF, p.padding, p.out_sigma_factor, s.yr, s.yi);

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
        k5_scalar_train(&s.model, s.fr, s.fi, 1, s.prm.use_poly_exp, &s.wk);   /* first = 1 */
    });
}

ref::KcfBox ref::kcf_flow(KcfState &s, const Image<uint8_t, ImageType::GRAY> &frame, KcfBox b,
                          KcfFlowStats *stats)
{
    /* detect at the old centre */
    features(s, frame, b, stats);
    run_stage(stats, KCF_K5_DETECT, [&] {
        k5_scalar_detect(&s.model, s.fr, s.fi, s.prm.use_poly_exp, s.resp, &s.wk);
    });
    run_stage(stats, KCF_K6, [&] {   /* peak shift in template pixels */
        k6_detection(s.resp, KCF_NF, KCF_NF, KCF_CELL, s.peak_x, s.peak_y);
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
        k5_scalar_train(&s.model, s.fr, s.fi, 0, s.prm.use_poly_exp, &s.wk);
    });
    return b;
}
