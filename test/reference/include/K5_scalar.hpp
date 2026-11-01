#ifndef K5_H
#define K5_H
#include <stdint.h>
#include <stddef.h>

/* K5 works on spectra of feature maps: ch channels of rows x cols = N frequencies.
   Planar: channel c starts at c * N. Complex numbers are split: one re array, one im array. */
typedef struct {
    int    rows, cols, ch;     /* 32, 32, 31 in the project                     */
    float  inv_sd;             /* 1 / (sigma^2 * D), D = rows * cols * ch       */
    float  lambda, eta;        /* ridge regularizer, model learning rate        */
    float *xr, *xi;            /* model template spectrum x^  (ch * N each)     */
    float *ar, *ai;            /* model alpha^                (N each)          */
    const float *yr, *yi;      /* label spectrum y^           (N each, fixed)   */
} K5Model;

typedef struct {               /* scratch, N floats each, from the caller       */
    float *sr, *si;            /* spectral product, then c (after the IFFT)     */
    float *kr, *ki;            /* kernel k, then its spectrum k^                */
    float *ar, *ai;            /* new alpha^ (train) or response spectrum (detect) */
} K5Work;

#define K5_EXP_MIN (-87.33f)   /* below this, exp(x) < FLT_MIN: the result is 0  */

/* scalar (k5_scalar.c) */
namespace ref
{

    void fft2(float *re, float *im, int rows, int cols, int inverse); //1

    void  k5_spectral(const float *xr, const float *xi, const float *zr, const float *zi,
                      int C, int N, float *sr, float *si);//2

    float k5_sumsq(const float *x, int n);//3

    float k5_exp(float x);//4

    void  k5_gauss(const float *c, int N, float xx, float zz, float inv_sd, int use_poly, float *k); //5 

    void  k5_train_alpha(const float *yr, const float *yi, const float *kr, const float *ki,
                         float lambda, int N, float *ar, float *ai); //6 

    void  k5_detect_resp(const float *ar, const float *ai, const float *kr, const float *ki,
                         int N, float *rr, float *ri); //7

    void  k5_update(float *m, const float *nw, int n, float eta); //8

    float k5_norm(const float *re, const float *im, int C, int N); //9

    void  k5_scalar_correlate(const float *xr, const float *xi, float xx,
                              const float *zr, const float *zi, float zz,
                              int rows, int cols, int C, float inv_sd, int use_poly, K5Work *w); //10

    void  k5_scalar_train(K5Model *m, const float *xr, const float *xi, int first, int use_poly,
                          K5Work *w); //11

    void  k5_scalar_detect(const K5Model *m, const float *zr, const float *zi, int use_poly,
                       float *resp, K5Work *w); //12 

    void  k5_make_labels(int rows, int cols, float padding, float out_sigma_factor,
                         float *yr, float *yi); //13

    void  k5_spectra(const float *F, int C, int rows, int cols, float *xr, float *xi); //14

 
}


#endif  /* K5_H */