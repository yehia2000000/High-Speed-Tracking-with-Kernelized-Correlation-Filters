#include <cmath>
#include <cfloat>

#define NUM_SECTORS 9
#define NUM_BINS    27

void ref::k4_normalization_hanning_scalar(
    const float* raw_hist,    // IN  : Output from K3, size: GridH * GridW * 27
    const float* hann_win,    // IN  : 2D Hanning window, size: (GridH-2) * (GridW-2)
    float*       out_fhog,    // OUT : Final tensor, size: (GridH-2) * (GridW-2) * 31
    int          grid_w,      // IN  : Width in cells 
    int          grid_h,      // IN  : Height in cells 
    float        alpha        // IN  : Truncation threshold (0.2f)
) {
    // -------------------------------------------------------------
    // Step 1: Precompute energy (sum of squares of 9 unsigned bins)
    // -------------------------------------------------------------
    float cell_sum_sq[grid_h * grid_w];

    for (int i = 0; i < grid_h * grid_w; i++) {
        float sum_sq = 0.0f;
        int cell_idx = i * NUM_BINS;
        for (int j = 0; j < NUM_SECTORS; j++) {
            float val = raw_hist[cell_idx + j];
            sum_sq += val * val;
        }
        cell_sum_sq[i] = sum_sq;
    }

    const int out_w = grid_w - 2; // e.g. 30
    const int out_h = grid_h - 2; // e.g. 30

    // -------------------------------------------------------------
    // Step 2: Loop over inner cells (1 to grid_h-2, 1 to grid_w-2)
    // -------------------------------------------------------------
    for (int i = 1; i <= out_h; i++) {
        for (int j = 1; j <= out_w; j++) {

            // Pointers for current cell
            const float* cell_bins = raw_hist + (i * grid_w + j) * NUM_BINS;
            int out_cell_idx = (i - 1) * out_w + (j - 1);
            float* cell_out  = out_fhog + (out_cell_idx * 31);

            // ---------------------------------------------------------
            // Step 3: Compute the 4 Block Norms
            // ---------------------------------------------------------
            float norm[4];
            // Block 0: Bottom-Right
            norm[0] = std::sqrt(cell_sum_sq[i * grid_w + j] +
                                cell_sum_sq[i * grid_w + (j + 1)] +
                                cell_sum_sq[(i + 1) * grid_w + j] +
                                cell_sum_sq[(i + 1) * grid_w + (j + 1)]) + FLT_EPSILON;

            // Block 1: Top-Right
            norm[1] = std::sqrt(cell_sum_sq[i * grid_w + j] +
                                cell_sum_sq[i * grid_w + (j + 1)] +
                                cell_sum_sq[(i - 1) * grid_w + j] +
                                cell_sum_sq[(i - 1) * grid_w + (j + 1)]) + FLT_EPSILON;

            // Block 2: Bottom-Left
            norm[2] = std::sqrt(cell_sum_sq[i * grid_w + j] +
                                cell_sum_sq[i * grid_w + (j - 1)] +
                                cell_sum_sq[(i + 1) * grid_w + j] +
                                cell_sum_sq[(i + 1) * grid_w + (j - 1)]) + FLT_EPSILON;

            // Block 3: Top-Left
            norm[3] = std::sqrt(cell_sum_sq[i * grid_w + j] +
                                cell_sum_sq[i * grid_w + (j - 1)] +
                                cell_sum_sq[(i - 1) * grid_w + j] +
                                cell_sum_sq[(i - 1) * grid_w + (j - 1)]) + FLT_EPSILON;

            // ---------------------------------------------------------
            // Step 4: Normalize & Clamp for the 4 Blocks
            // ---------------------------------------------------------
            float block[4][27];
            for (int k = 0; k < 4; k++) {
                for (int b = 0; b < 27; b++) {
                    float val = cell_bins[b] / norm[k];
                    block[k][b] = (val > alpha) ? alpha : val;
                }
            }

            // ---------------------------------------------------------
            // Step 5: Reduce to 31 Channels
            // ---------------------------------------------------------
            // 18 Signed channels
            for (int s = 0; s < 18; s++) {
                cell_out[s] = (block[0][9 + s] + block[1][9 + s] + 
                               block[2][9 + s] + block[3][9 + s]) * 0.5f;
            }

            // 9 Unsigned channels
            for (int u = 0; u < 9; u++) {
                cell_out[18 + u] = (block[0][u] + block[1][u] + 
                                    block[2][u] + block[3][u]) * 0.5f;
            }

            // 4 Texture Energy channels
            for (int k = 0; k < 4; k++) {
                float sum_signed = 0.0f;
                for (int s = 0; s < 18; s++) {
                    sum_signed += block[k][9 + s];
                }
                cell_out[27 + k] = sum_signed * 0.23570226f; // (1 / sqrt(18))
            }

            // ---------------------------------------------------------
            // Step 6: Multiply by Hanning Window
            // ---------------------------------------------------------
            float h = hann_win[out_cell_idx];
            for (int ch = 0; ch < 31; ch++) {
                cell_out[ch] *= h;
            }
        }
    }
}