#include "kcf_scalar.hpp"
#include "kcf_prof.hpp"
#include "riscv_math.hpp"   // <cmath> is not available on bare metal

namespace kcf {

Tracker::Tracker(const Params &p) : prm_(p) {
    /* Hann on the K4 grid, symmetric, edges exactly 0 (N - 1 denominator, as KCFcpp) */
    for (int i = 0; i < NF; i++)
        for (int j = 0; j < NF; j++) {
            float hy = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * i / (NF - 1)));
            float hx = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * j / (NF - 1)));
            hann_[i * NF + j] = hy * hx;
        }
    ref::k5_make_labels(NF, NF, prm_.padding, prm_.out_sigma_factor, yr_, yi_);
    m_.rows = NF; m_.cols = NF; m_.ch = C;
    m_.inv_sd = 1.0f / (prm_.sigma * prm_.sigma * (float)N * (float)C);
    m_.lambda = prm_.lambda; m_.eta = prm_.eta;
    m_.xr = mxr_; m_.xi = mxi_; m_.ar = mar_; m_.ai = mai_;
    m_.yr = yr_;  m_.yi = yi_;
    w_.sr = work_;         w_.si = work_ + N;
    w_.kr = work_ + 2 * N; w_.ki = work_ + 3 * N;
    w_.ar = work_ + 4 * N; w_.ai = work_ + 5 * N;
}

/* K1 maps 2.5 x target onto T pixels. K4 keeps only the inner T - 2*CELL pixels, so the
   window is widened by T / (T - 2*CELL) to make the features cover exactly 2.5 x target,
   as KCFcpp does with its + 8 pixels. */
ref::Roi Tracker::roi_for(Box b) const {
    const float grow = (float)T / (float)(T - 2 * CELL);
    ref::Roi r;
    r.cx = b.cx; r.cy = b.cy;
    r.tw = b.w * grow * prm_.padding / 2.5f;   /* K1 multiplies by 2.5 itself */
    r.th = b.h * grow * prm_.padding / 2.5f;
    return r;
}

void Tracker::features(const Image<uint8_t, ImageType::GRAY> &img, Box b) {
    { PROF(K1); ref::k1_scalar(img, roi_for(b), T, k1_); }
    { PROF(K2); ref::k2_scalar(k1_, T, mag_, bin_, prm_.k2); }
    { PROF(K3); ref::k3_scalar<CELL>(bin_, mag_, hist_, T, T); }
    { PROF(K4); ref::k4_normalization_hanning_scalar(hist_, hann_, fhog_, G, G, prm_.alpha); }
    /* K4 writes 31 features per cell next to each other (interleaved); K5 needs one plane
       per channel. This is the same job as the .t() in KCFcpp's getFeatures. */
    { PROF(LAYOUT);
      for (int cell = 0; cell < N; cell++)
          for (int ch = 0; ch < C; ch++)
              fr_[ch * N + cell] = fhog_[cell * C + ch]; }
    { PROF(K5_FFT); ref::k5_spectra(fr_, C, NF, NF, fr_, fi_); }        /* 31 FFTs, in place */
}

void Tracker::init(const Image<uint8_t, ImageType::GRAY> &img, Box b) {
    box_ = b;
    features(img, box_);
    ref::k5_scalar_train(&m_, fr_, fi_, 1, prm_.use_poly_exp, &w_);
}

Box Tracker::update(const Image<uint8_t, ImageType::GRAY> &img) {
    /* detect at the old centre */
    features(img, box_);
    { PROF(K5_DETECT); ref::k5_scalar_detect(&m_, fr_, fi_, prm_.use_poly_exp, resp_, &w_); }
    { PROF(K6); ref::k6_detection(resp_, NF, NF, CELL, shift_x_, shift_y_); }   /* template pixels */

    /* template pixels -> image pixels: K1's sample step */
    ref::Roi r = roi_for(box_);
    float sx = 2.5f * r.tw / (float)T, sy = 2.5f * r.th / (float)T;
    box_.cx += shift_x_ * sx;
    box_.cy += shift_y_ * sy;
    if (box_.cx < 0) box_.cx = 0;
    if (box_.cy < 0) box_.cy = 0;
    if (box_.cx > (float)(img.Width() - 1))  box_.cx = (float)(img.Width() - 1);
    if (box_.cy > (float)(img.Height() - 1)) box_.cy = (float)(img.Height() - 1);

    /* train at the new centre */
    features(img, box_);
    { PROF(K5_TRAIN); ref::k5_scalar_train(&m_, fr_, fi_, 0, prm_.use_poly_exp, &w_); }
    return box_;
}

}  /* namespace kcf */
