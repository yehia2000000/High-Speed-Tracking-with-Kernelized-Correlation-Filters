#ifndef K3_H
#define K3_H
#include <stdint.h>
#include <stddef.h>
#include <cstring>

#define NUM_SECTORS 9   // 9 unsigned bins (0..8)
#define NUM_BINS    27  // 9 unsigned + 18 signed (0..26)

namespace ref
{
// =========================================================================
// 1. Generate Weights & Nearest Directions for any even k
// =========================================================================
inline void generate_k3_weights(uint8_t k, float *w_h, float *w_n, int *nearest) {
    for (int j = 0; j < k / 2; j++) {
        // Nearest direction: Left/Top (-1) for first half, Right/Bottom (+1) for second half
        nearest[j]             = -1;
        nearest[k - 1 - j]     =  1;

        // Distance-based weights
        float b_x = (k / 2.0f) + j + 0.5f;
        float home_weight = b_x / (float)k;
        float nbr_weight  = 1.0f - home_weight;

        // Left half (j)
        w_h[j] = home_weight;
        w_n[j] = nbr_weight;

        // Symmetrical Right half (k - 1 - j)
        int mirror = k - 1 - j;
        w_h[mirror] = home_weight;
        w_n[mirror] = nbr_weight;
    }
}

// =========================================================================
// 2. Core Accumulation Engine (Templated on k for auto-unrolling)
// =========================================================================
template <uint8_t k>
inline void hist_bins(
    const uint8_t *ori_bins, // [IN]  Signed orientation bins (0..17), size: H * W
    const float   *mag,      // [IN]  Gradient magnitudes, size: H * W
    float         *hist,     // [OUT] Output histograms, size: (H/k) * (W/k) * 27
    int            width,    // Image width (e.g. 128)
    int            height,   // Image height (e.g. 128)
    const float   *w_h,      // Precomputed home weights of size k
    const float   *w_n,      // Precomputed neighbor weights of size k
    const int     *nearest)  // Precomputed neighbor offsets of size k
{
    const int grid_w = width  / k;
    const int grid_h = height / k;
    const int row_stride = grid_w * NUM_BINS; // Floats per row of cells

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {

            // Base pointer to the 27 bins of cell (i, j)
            float *cell_hist = hist + (i * row_stride) + (j * NUM_BINS);
            
            // Loops unroll completely at compile-time when k is a template!
            for (int ii = 0; ii < k; ii++) {
                for (int jj = 0; jj < k; jj++) {
                    int py = i * k + ii;
                    int px = j * k + jj;

                    int pixel_idx = py * width + px;
                    // Read inputs for pixel
                    float   m     = *(mag + pixel_idx);
                    uint8_t s_bin = *(ori_bins + pixel_idx); // 0:17
                    uint8_t u_bin = (s_bin >= 9) ? (s_bin - 9) : s_bin; // 0:8


                    int bin_u = u_bin;
                    int bin_s = s_bin + NUM_SECTORS; // 9..26

                    // Spatial weights from stack arrays
                    float wy_h = w_h[ii];
                    float wy_n = w_n[ii];
                    float wx_h = w_h[jj];
                    float wx_n = w_n[jj];

                    int ci_nbr = i + nearest[ii];
                    int cj_nbr = j + nearest[jj];

                    // 1. Home Cell (i, j)
                    float w_home = m * wy_h * wx_h;
                    *(cell_hist + bin_u) += w_home;
                    *(cell_hist + bin_s) += w_home;

                    // 2. Vertical Neighbor Cell (ci_nbr, j)
                    if (ci_nbr >= 0 && ci_nbr < grid_h) {
                        float *v_hist = hist + (ci_nbr * row_stride) + (j * NUM_BINS);
                        float  w_vert = m * wy_n * wx_h;
                        *(v_hist + bin_u) += w_vert;
                        *(v_hist + bin_s) += w_vert;
                    }

                    // 3. Horizontal Neighbor Cell (i, cj_nbr)
                    if (cj_nbr >= 0 && cj_nbr < grid_w) {
                        float *h_hist = hist + (i * row_stride) + (cj_nbr * NUM_BINS);
                        float  w_horiz = m * wy_h * wx_n;
                        *(h_hist + bin_u) += w_horiz;
                        *(h_hist + bin_s) += w_horiz;
                    }

                    // 4. Diagonal Neighbor Cell (ci_nbr, cj_nbr)
                    if (ci_nbr >= 0 && ci_nbr < grid_h && cj_nbr >= 0 && cj_nbr < grid_w) {
                        float *d_hist = hist + (ci_nbr * row_stride) + (cj_nbr * NUM_BINS);
                        float  w_diag = m * wy_n * wx_n;
                        *(d_hist + bin_u) += w_diag;
                        *(d_hist + bin_s) += w_diag;
                    }
                }
            }
        }
    }
}

// =========================================================================
// 3. Top-Level Wrapper Function (Default k = 4 for KCF)
// =========================================================================
template <uint8_t k = 4>
void k3_scalar(
    const uint8_t *ori_bins, // [IN]  Orientations from Student 1 (size: H * W)
    const float   *mag,      // [IN]  Magnitudes from Student 1   (size: H * W)
    float         *hist,     // [OUT] Output buffer (size: (H/k) * (W/k) * 27)
    int            width,
    int            height)
{
    float w_h[k]; // weight for interpolation between cells 
    float w_n[k];
    int   nearest[k];

    // Precalculate weights
    generate_k3_weights(k, w_h, w_n, nearest);
    const int total_bins = (height / k) * (width / k) * NUM_BINS;
    std::memset(hist, 0, total_bins * sizeof(float));
    // Run accumulation
    hist_bins<k>(ori_bins, mag, hist, width, height, w_h, w_n, nearest);
}

}


#endif  /* K3_H */