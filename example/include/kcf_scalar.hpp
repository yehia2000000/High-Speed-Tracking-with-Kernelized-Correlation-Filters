/*
 * kcf_scalar.hpp: the scalar KCF tracker built from the K1–K6 kernels.
 *
 * Per frame (single scale):
 *   detect: K1 (window at the old centre) -> K2 -> K3 -> K4 -> planar -> K5 spectra -> K5 detect -> K6
 *   train:  K1 (window at the new centre) -> K2 -> K3 -> K4 -> planar -> K5 spectra -> K5 train
 *
 * All buffers are fixed-size members: no malloc, memcpy or memset (gem5 lite).
 * The object is large (about 1 MB), so make it static or global, not a local variable.
 */
#pragma once
#include <cstdint>
#include "image.hpp"
#include "K1_scalar.hpp"
#include "K2_scalar.hpp"
#include "K3_scalar.hpp"
#include "K4_scalar.hpp"
#include "K5_scalar.hpp"
#include "K6_scalar.hpp"

namespace kcf {

/* Geometry. K4 drops the outer ring of cells, so the K1 template is 2 cells bigger than the
   grid K5 works on: T = 136 -> 34 x 34 cells -> 32 x 32 after K4 (a power of 2 for the FFT). */
constexpr int CELL = 4;
constexpr int T    = 136;              /* K1 output without the apron                  */
constexpr int P    = T + 2;            /* K1 output with the 1-pixel apron             */
constexpr int G    = T / CELL;         /* cells per side out of K3 (34)                 */
constexpr int NF   = G - 2;            /* cells per side out of K4 (32)                 */
constexpr int N    = NF * NF;          /* frequencies per channel (1024)                */
constexpr int C    = 31;               /* feature channels                              */

struct Params {
    float padding          = 2.5f;     /* window = 2.5 x target                         */
    float sigma            = 0.5f;     /* Gaussian kernel bandwidth (paper Table 2)     */
    float lambda           = 1e-4f;    /* ridge regularizer                             */
    float eta              = 0.02f;    /* model learning rate                           */
    float out_sigma_factor = 0.1f;     /* label width / target size                     */
    float alpha            = 0.2f;     /* K4 truncation                                 */
    ref::K2Method k2       = ref::K2_DOT;    /* same bins as KCFcpp fhog */
    int   use_poly_exp     = 1;        /* K5: k5_exp (1) or expf (0)                    */
};

struct Box { float cx, cy, w, h; };    /* centre and size, image pixels                 */

class Tracker {
public:
    explicit Tracker(const Params &p = Params());
    void init(const Image<uint8_t, ImageType::GRAY> &img, Box b);
    Box  update(const Image<uint8_t, ImageType::GRAY> &img);
    Box  box() const { return box_; }
    float last_peak_shift_x() const { return shift_x_; }   /* template pixels, last detect */
    float last_peak_shift_y() const { return shift_y_; }

private:
    void features(const Image<uint8_t, ImageType::GRAY> &img, Box b);   /* -> fr_ (planar) */
    ref::Roi roi_for(Box b) const;

    Params prm_;
    Box    box_ {0, 0, 0, 0};
    float  shift_x_ = 0, shift_y_ = 0;

    float   k1_[P * P];                /* K1 out                                        */
    float   mag_[T * T];               /* K2 out                                        */
    uint8_t bin_[T * T];
    float   hist_[G * G * 27];         /* K3 out, interleaved: 27 bins per cell         */
    float   hann_[N];                  /* 2-D Hann on the 32 x 32 grid                  */
    float   fhog_[N * C];              /* K4 out, interleaved: 31 features per cell     */
    float   fr_[C * N], fi_[C * N];    /* planar features, then their spectra           */
    float   yr_[N], yi_[N];            /* label spectrum                                */
    float   mxr_[C * N], mxi_[C * N];  /* model x^                                      */
    float   mar_[N], mai_[N];          /* model alpha^                                  */
    float   work_[6 * N];              /* K5Work scratch                                */
    float   resp_[N];                  /* response, top-left convention                 */
    K5Model      m_;
    K5Work       w_;
};

}  /* namespace kcf */
