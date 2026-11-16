#pragma once

#include <cstdint>
#include <cstddef>

#define MAX_GRID_WIDTH    64
#define MAX_GRID_HEIGHT   64
#define NUM_SECTORS       9
#define NUM_HIST_BINS     27
#define NUM_FHOG_CHANNELS 31

namespace vec {

// Phase 1: Precomputes the energy grid (sum of squares of 9 unsigned bins 0..8)
// raw_hist is in planar layout: [27][grid_h * grid_w] floats
// cell_sum_sq is a 2D grid: [grid_h * grid_w] floats
void compute_cell_energy_grid(
    const float* raw_hist,
    float*       cell_sum_sq,
    int          grid_w,
    int          grid_h
);

// Full Vectorized Kernel 4 (FHOG Normalization, Truncation, and Hanning Windowing)
// raw_hist: [27][grid_h * grid_w] floats (Planar output from K3)
// hann_win: (grid_h - 2) * (grid_w - 2) floats (Precomputed 2D Hanning window)
// out_fhog: [31][(grid_h - 2) * (grid_w - 2)] floats (Planar Layout A for K5 FFT)
// alpha:    truncation threshold (0.2f), as in ref::k4_normalization_hanning_scalar
void k4_normalization_hanning(
    const float* raw_hist,
    const float* hann_win,
    float*       out_fhog,
    int          grid_w,
    int          grid_h,
    float        alpha
);

} // namespace vec
