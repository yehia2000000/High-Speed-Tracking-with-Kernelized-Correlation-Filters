#ifndef K5_RVV_HPP_
#define K5_RVV_HPP_

#include "K5_scalar.hpp"   // K5Model, K5Work, K5_EXP_MIN and the ref:: versions

namespace vec
{

/* element-wise */
void  k5_copy(float *dst, const float *src, int n);
void  k5_zero(float *dst, int n);
void  k5_update(float *m, const float *nw, int n, float eta);

/* per frequency */
void  k5_detect_resp(const float *ar, const float *ai, const float *kr, const float *ki,
                     int N, float *rr, float *ri);
void  k5_spectral(const float *xr, const float *xi, const float *zr, const float *zi,
                  int C, int N, float *sr, float *si);
void  k5_train_alpha(const float *yr, const float *yi, const float *kr, const float *ki,
                     float lambda, int N, float *ar, float *ai);

/* Gaussian kernel (k5_exp for use_poly = 1, ref::k5_gauss for use_poly = 0) */
void  k5_gauss(const float *c, int N, float xx, float zz, float inv_sd, int use_poly, float *k);

/* reductions (not bit-identical to ref::) */
float k5_sumsq(const float *x, int n);
float k5_norm(const float *re, const float *im, int C, int N);

/* stages */
void  k5_spectra(const float *F, int C, int rows, int cols, float *xr, float *xi);
void  k5_correlate(const float *xr, const float *xi, float xx, const float *zr, const float *zi,
                   float zz, int rows, int cols, int C, float inv_sd, int use_poly, K5Work *w);
void  k5_train(K5Model *m, const float *xr, const float *xi, int first, int use_poly, K5Work *w);
void  k5_detect(const K5Model *m, const float *zr, const float *zi, int use_poly, float *resp,
                K5Work *w);
void  k5_make_labels(int rows, int cols, float padding, float out_sigma_factor, float *yr,
                     float *yi);

}  // namespace vec

#endif  // K5_RVV_HPP_