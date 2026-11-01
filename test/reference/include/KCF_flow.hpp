#ifndef KCF_FLOW_H_
#define KCF_FLOW_H_
/*
 * KCF_flow: the K1–K6 reference kernels connected into one tracking step (single scale).
 *
 *   ref::kcf_flow_init  first frame:
 *       K1 -> K2 -> K3 -> K4 -> layout -> K5 spectra -> K5 train
 *
 *   ref::kcf_flow       every next frame:
 *       detect at the old centre:  K1 -> K2 -> K3 -> K4 -> layout -> K5 spectra -> K5 detect -> K6
 *       move the box by the K6 peak
 *       train at the new centre:   K1 -> K2 -> K3 -> K4 -> layout -> K5 spectra -> K5 train
 *
 * Kernels only: no file I/O, no printf, no drawing. The caller owns the frames and the boxes:
 *   benchmark/source/KCF_benchmark.cpp   times the flow on gem5 (cycles per kernel)
 *   example/source/kcf_tracker.cpp       runs a whole sequence on qemu and writes the boxes
 *
 * KcfState holds every buffer (about 1 MB). It is a plain struct with no constructor, so a
 * global KcfState lands in .bss and needs no static initialization (nexus-am runs none).
 * Do not put it on the stack: the bare-metal stack is 32 KB.
 */
#include <cstdint>
#include "image.hpp"
#include "K1_scalar.hpp"
#include "K2_scalar.hpp"
#include "K3_scalar.hpp"
#include "K4_scalar.hpp"
#include "K5_scalar.hpp"
#include "K6_scalar.hpp"

namespace ref
{
    /* Geometry. K4 drops the outer ring of cells, so the K1 template is 2 cells bigger than
       the grid K5 works on: T = 136 -> 34 x 34 cells -> 32 x 32 after K4 (a power of 2). */
    constexpr int KCF_CELL = 4;
    constexpr int KCF_T    = 136;               /* K1 output without the apron       */
    constexpr int KCF_P    = KCF_T + 2;         /* K1 output with the 1-pixel apron  */
    constexpr int KCF_G    = KCF_T / KCF_CELL;  /* cells per side out of K3 (34)      */
    constexpr int KCF_NF   = KCF_G - 2;         /* cells per side out of K4 (32)      */
    constexpr int KCF_N    = KCF_NF * KCF_NF;   /* frequencies per channel (1024)     */
    constexpr int KCF_C    = 31;                /* feature channels                   */

    typedef struct {
        float    padding;            /* window = 2.5 x target                         */
        float    sigma;              /* Gaussian kernel bandwidth (paper Table 2)     */
        float    lambda;             /* ridge regularizer                             */
        float    eta;                /* model learning rate                           */
        float    out_sigma_factor;   /* label width / target size                     */
        float    alpha;              /* K4 truncation                                 */
        K2Method k2;                 /* K2 binning (K2_DOT = same bins as KCFcpp fhog) */
        int      use_poly_exp;       /* K5: k5_exp (1) or expf (0)                    */
    } KcfParams;

    /* The paper's values. Copy and change a field: KcfParams p = KCF_DEFAULT_PARAMS; */
    constexpr KcfParams KCF_DEFAULT_PARAMS = {2.5f, 0.5f, 1e-4f, 0.02f, 0.1f, 0.2f, K2_DOT, 1};

    typedef struct { float cx, cy, w, h; } KcfBox;   /* centre and size, image pixels */

    /* The stages, in the order the flow runs them. K1 .. K5_FFT run twice per frame
       (detect and train), K5_DETECT and K6 once, K5_TRAIN once. */
    enum KcfStage { KCF_K1, KCF_K2, KCF_K3, KCF_K4, KCF_LAYOUT, KCF_K5_FFT,
                    KCF_K5_DETECT, KCF_K6, KCF_K5_TRAIN, KCF_STAGES };
    const char *kcf_stage_name(int stage);

    /* Filled by the flow when the caller passes one; sums over every call of a stage. */
    typedef struct { uint64_t cycles, instructions, calls; } KcfStageCount;
    typedef struct { KcfStageCount stage[KCF_STAGES]; } KcfFlowStats;

    typedef struct {
        KcfParams prm;
        float   k1[KCF_P * KCF_P];               /* K1 out                                 */
        float   mag[KCF_T * KCF_T];              /* K2 out                                 */
        uint8_t bin[KCF_T * KCF_T];
        float   hist[KCF_G * KCF_G * 27];        /* K3 out, 27 bins per cell, interleaved   */
        float   hann[KCF_N];                     /* 2-D Hann on the 32 x 32 grid            */
        float   fhog[KCF_N * KCF_C];             /* K4 out, 31 features per cell, interleaved */
        float   fr[KCF_C * KCF_N], fi[KCF_C * KCF_N];   /* planar features, then spectra    */
        float   yr[KCF_N], yi[KCF_N];            /* label spectrum                          */
        float   mxr[KCF_C * KCF_N], mxi[KCF_C * KCF_N]; /* model x^                         */
        float   mar[KCF_N], mai[KCF_N];          /* model alpha^                            */
        float   work[6 * KCF_N];                 /* K5Work scratch                          */
        float   resp[KCF_N];                     /* K5 response, top-left convention        */
        float   peak_x, peak_y;                  /* last K6 peak, template pixels           */
        K5Model model;
        K5Work  wk;
    } KcfState;

    /* Once, before the first frame: Hann window, label spectrum, model pointers. */
    void kcf_setup(KcfState &s, const KcfParams &p);

    /* First frame: features at box b, then the first train (the model starts from them). */
    void kcf_flow_init(KcfState &s, const Image<uint8_t, ImageType::GRAY> &frame, KcfBox b,
                       KcfFlowStats *stats = nullptr);

    /* One frame: detect around box b, move it to the peak, train there. Returns the new box.
       stats: nullptr = no timing (qemu); a KcfFlowStats = cycles/instructions per stage. */
    KcfBox kcf_flow(KcfState &s, const Image<uint8_t, ImageType::GRAY> &frame, KcfBox b,
                    KcfFlowStats *stats = nullptr);
}

#endif  // KCF_FLOW_H_
