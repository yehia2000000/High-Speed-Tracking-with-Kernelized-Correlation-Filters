
#include <riscv_vector.h>
#include <cfloat>      // FLT_EPSILON
#include "K4_rvv.hpp"
namespace vec {

// Helper: processes a chunk of cells of vector length vl
static inline void accumulate_cell_energy_chunk(
    const float* raw_hist, float* cell_sum_sq,
    size_t i, size_t total_cells, size_t vl)
{
    vfloat32m4_t v_sum0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t v_sum1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    for (int j = 0; j < NUM_SECTORS - 1; j += 2) {
        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(raw_hist + (size_t)j * total_cells + i, vl);
        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(raw_hist + (size_t)(j + 1) * total_cells + i, vl);

        v_sum0 = __riscv_vfmacc_vv_f32m4(v_sum0, v0, v0, vl);
        v_sum1 = __riscv_vfmacc_vv_f32m4(v_sum1, v1, v1, vl);
    }

    vfloat32m4_t v8 = __riscv_vle32_v_f32m4(raw_hist + (size_t)8 * total_cells + i, vl);
    v_sum0 = __riscv_vfmacc_vv_f32m4(v_sum0, v8, v8, vl);

    vfloat32m4_t v_total = __riscv_vfadd_vv_f32m4(v_sum0, v_sum1, vl);
    __riscv_vse32_v_f32m4(cell_sum_sq + i, v_total, vl);
}

// Phase 1: Cell Energy Grid 
void compute_cell_energy_grid(
    const float* raw_hist,
    float*       cell_sum_sq,
    int          grid_w,
    int          grid_h)
{
    const size_t total_cells = (size_t)grid_w * grid_h;
    const size_t vlmax = __riscv_vsetvlmax_e32m4();

    size_t i = 0;
    for (; i + vlmax <= total_cells; i += vlmax) {
        accumulate_cell_energy_chunk(raw_hist, cell_sum_sq, i, total_cells, vlmax);
    }

    if (i < total_cells) {
        size_t vl_tail = __riscv_vsetvl_e32m4(total_cells - i);
        accumulate_cell_energy_chunk(raw_hist, cell_sum_sq, i, total_cells, vl_tail);
    }
}

// Phase 2: Block Norm Grid (31x31) 
//    ├── Vectorized 2x2 box-sum over energy grid
//    └── Vector square root + epsilon (vfsqrt.v + vfadd.vf)
static inline void norm_helper(
    const float* row_bot_ptr,
    vfloat32m2_t &sum_bot_l,
    vfloat32m2_t &sum_bot_r,
    vfloat32m2_t &v_sqrt_box_next_left,
    vfloat32m2_t &v_sqrt_box_next_right,
    size_t vlmax)
{
    vfloat32m2_t v_bot_prev = __riscv_vle32_v_f32m2(row_bot_ptr - 1, vlmax);
    vfloat32m2_t v_bot_curr = __riscv_vle32_v_f32m2(row_bot_ptr,     vlmax);
    vfloat32m2_t v_bot_next = __riscv_vle32_v_f32m2(row_bot_ptr + 1, vlmax);

    vfloat32m2_t next_sum_l = __riscv_vfadd_vv_f32m2(v_bot_prev, v_bot_curr, vlmax);
    vfloat32m2_t next_sum_r = __riscv_vfadd_vv_f32m2(v_bot_curr, v_bot_next, vlmax);

    vfloat32m2_t v_box_sum_l = __riscv_vfadd_vv_f32m2(sum_bot_l, next_sum_l, vlmax);
    vfloat32m2_t v_box_sum_r = __riscv_vfadd_vv_f32m2(sum_bot_r, next_sum_r, vlmax);

    v_sqrt_box_next_left  = __riscv_vfadd_vf_f32m2(__riscv_vfsqrt_v_f32m2(v_box_sum_l, vlmax), FLT_EPSILON, vlmax);
    v_sqrt_box_next_right = __riscv_vfadd_vf_f32m2(__riscv_vfsqrt_v_f32m2(v_box_sum_r, vlmax), FLT_EPSILON, vlmax);

    sum_bot_l = next_sum_l;
    sum_bot_r = next_sum_r;
}

static inline void fhog_normalization_chunk(
    const float* raw_hist,
    float* out_fhog,
    const float* hann_win,
    const float* cell_sum_sq,
    int out_w,
    int out_h,
    int col_c,
    float alpha,
    size_t vlmax)
{
    const int grid_w = out_w + 2;
    const size_t total_cells = (size_t)grid_w * (out_h + 2);
    const size_t out_plane = (size_t)out_w * out_h;

    vfloat32m2_t v_top0_prev = __riscv_vle32_v_f32m2(cell_sum_sq + 0 * grid_w + col_c - 1, vlmax);
    vfloat32m2_t v_top0_curr = __riscv_vle32_v_f32m2(cell_sum_sq + 0 * grid_w + col_c,     vlmax);
    vfloat32m2_t v_top0_next = __riscv_vle32_v_f32m2(cell_sum_sq + 0 * grid_w + col_c + 1, vlmax);
    vfloat32m2_t sum_top_l   = __riscv_vfadd_vv_f32m2(v_top0_prev, v_top0_curr, vlmax);
    vfloat32m2_t sum_top_r   = __riscv_vfadd_vv_f32m2(v_top0_curr, v_top0_next, vlmax);

    vfloat32m2_t v_bot1_prev = __riscv_vle32_v_f32m2(cell_sum_sq + 1 * grid_w + col_c - 1, vlmax);
    vfloat32m2_t v_bot1_curr = __riscv_vle32_v_f32m2(cell_sum_sq + 1 * grid_w + col_c,     vlmax);
    vfloat32m2_t v_bot1_next = __riscv_vle32_v_f32m2(cell_sum_sq + 1 * grid_w + col_c + 1, vlmax);
    vfloat32m2_t sum_bot_l   = __riscv_vfadd_vv_f32m2(v_bot1_prev, v_bot1_curr, vlmax);
    vfloat32m2_t sum_bot_r   = __riscv_vfadd_vv_f32m2(v_bot1_curr, v_bot1_next, vlmax);

    vfloat32m2_t v_box_sum_l = __riscv_vfadd_vv_f32m2(sum_top_l, sum_bot_l, vlmax);
    vfloat32m2_t v_box_sum_r = __riscv_vfadd_vv_f32m2(sum_top_r, sum_bot_r, vlmax);

    vfloat32m2_t v_sqrt_box_left  = __riscv_vfadd_vf_f32m2(__riscv_vfsqrt_v_f32m2(v_box_sum_l, vlmax), FLT_EPSILON, vlmax);
    vfloat32m2_t v_sqrt_box_right = __riscv_vfadd_vf_f32m2(__riscv_vfsqrt_v_f32m2(v_box_sum_r, vlmax), FLT_EPSILON, vlmax);

    const float* sliding_row_ptr = cell_sum_sq + 2 * grid_w + col_c;

    vfloat32m2_t inv_tr = __riscv_vfrdiv_vf_f32m2(v_sqrt_box_right, 1.0f, vlmax);
    vfloat32m2_t inv_tl = __riscv_vfrdiv_vf_f32m2(v_sqrt_box_left,  1.0f, vlmax);

    alignas(16) float saved_bot_l[64];
    alignas(16) float saved_bot_r[64];
    __riscv_vse32_v_f32m2(saved_bot_l, sum_bot_l, vlmax);
    __riscv_vse32_v_f32m2(saved_bot_r, sum_bot_r, vlmax);

    for (size_t j = 1; j <= (size_t)out_h; j += 1) {
        // Hanning window scale (load and scale half_hann; sq18_hann is deferred until needed)
        const size_t out_cell_off = (j - 1) * out_w + (col_c - 1);
        vfloat32m2_t half_hann    = __riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(hann_win + out_cell_off, vlmax), 0.5f, vlmax);

        // load next row , sliding window : 
        sum_bot_l = __riscv_vle32_v_f32m2(saved_bot_l, vlmax);
        sum_bot_r = __riscv_vle32_v_f32m2(saved_bot_r, vlmax);

        vfloat32m2_t v_sqrt_box_next_left, v_sqrt_box_next_right;
        norm_helper(sliding_row_ptr, sum_bot_l, sum_bot_r,
                    v_sqrt_box_next_left, v_sqrt_box_next_right, vlmax);

        __riscv_vse32_v_f32m2(saved_bot_l, sum_bot_l, vlmax);
        __riscv_vse32_v_f32m2(saved_bot_r, sum_bot_r, vlmax);

        vfloat32m2_t inv_br = __riscv_vfrdiv_vf_f32m2(v_sqrt_box_next_right, 1.0f, vlmax);
        vfloat32m2_t inv_bl = __riscv_vfrdiv_vf_f32m2(v_sqrt_box_next_left,  1.0f, vlmax);

        const size_t in_cell_off = (size_t)j * grid_w + col_c;
        const float* raw_bin_ptr = raw_hist + in_cell_off;

        // Loop 1: Unsigned bins 0..8 (output channels 18..26)
        // Highly register-efficient: only 2 temporaries (sum and b)
        for (int i = 0; i < 9; i += 1) {
            vfloat32m2_t cell_bin = __riscv_vle32_v_f32m2(raw_bin_ptr + (i * total_cells), vlmax);
          
            vfloat32m2_t sum = __riscv_vfmin_vf_f32m2(__riscv_vfmul_vv_f32m2(cell_bin, inv_br, vlmax), alpha, vlmax);
            vfloat32m2_t b   = __riscv_vfmin_vf_f32m2(__riscv_vfmul_vv_f32m2(cell_bin, inv_tr, vlmax), alpha, vlmax);
            sum = __riscv_vfadd_vv_f32m2(sum, b, vlmax);

            b = __riscv_vfmin_vf_f32m2(__riscv_vfmul_vv_f32m2(cell_bin, inv_bl, vlmax), alpha, vlmax);
            sum = __riscv_vfadd_vv_f32m2(sum, b, vlmax);

            b = __riscv_vfmin_vf_f32m2(__riscv_vfmul_vv_f32m2(cell_bin, inv_tl, vlmax), alpha, vlmax);
            sum = __riscv_vfadd_vv_f32m2(sum, b, vlmax);

            sum = __riscv_vfmul_vv_f32m2(sum, half_hann, vlmax);
            __riscv_vse32_v_f32m2(out_fhog + (size_t)(18 + i) * out_plane + out_cell_off, sum, vlmax);
        }

        // Initialize texture energy accumulators right before Loop 2
        vfloat32m2_t Block_sum_ch27 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax); 
        vfloat32m2_t Block_sum_ch28 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t Block_sum_ch29 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t Block_sum_ch30 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

        // Loop 2: Signed bins 9..26 (output channels 0..17)
        // Uses only 2 vector temporaries: sum and b (6 vector registers including cell_bin)
        for (int i = 9; i < 27; i += 1) {
            vfloat32m2_t cell_bin = __riscv_vle32_v_f32m2(raw_bin_ptr + (i * total_cells), vlmax);
          
            vfloat32m2_t sum = __riscv_vfmin_vf_f32m2(__riscv_vfmul_vv_f32m2(cell_bin, inv_br, vlmax), alpha, vlmax);
            Block_sum_ch27 = __riscv_vfadd_vv_f32m2(Block_sum_ch27, sum, vlmax);

            vfloat32m2_t b = __riscv_vfmin_vf_f32m2(__riscv_vfmul_vv_f32m2(cell_bin, inv_tr, vlmax), alpha, vlmax);
            Block_sum_ch28 = __riscv_vfadd_vv_f32m2(Block_sum_ch28, b, vlmax);
            sum = __riscv_vfadd_vv_f32m2(sum, b, vlmax);

            b = __riscv_vfmin_vf_f32m2(__riscv_vfmul_vv_f32m2(cell_bin, inv_bl, vlmax), alpha, vlmax);
            Block_sum_ch29 = __riscv_vfadd_vv_f32m2(Block_sum_ch29, b, vlmax);
            sum = __riscv_vfadd_vv_f32m2(sum, b, vlmax);

            b = __riscv_vfmin_vf_f32m2(__riscv_vfmul_vv_f32m2(cell_bin, inv_tl, vlmax), alpha, vlmax);
            Block_sum_ch30 = __riscv_vfadd_vv_f32m2(Block_sum_ch30, b, vlmax);
            sum = __riscv_vfadd_vv_f32m2(sum, b, vlmax);

            sum = __riscv_vfmul_vv_f32m2(sum, half_hann, vlmax);
            __riscv_vse32_v_f32m2(out_fhog + (size_t)(i - 9) * out_plane + out_cell_off, sum, vlmax);
        }

        // Texture channels 27..30: compute sq18_hann on demand
        vfloat32m2_t v_hann_reload = __riscv_vle32_v_f32m2(hann_win + out_cell_off, vlmax);
        vfloat32m2_t sq18_hann    = __riscv_vfmul_vf_f32m2(v_hann_reload, 0.23570226f, vlmax);

        Block_sum_ch27 = __riscv_vfmul_vv_f32m2(Block_sum_ch27, sq18_hann, vlmax);
        Block_sum_ch28 = __riscv_vfmul_vv_f32m2(Block_sum_ch28, sq18_hann, vlmax);
        Block_sum_ch29 = __riscv_vfmul_vv_f32m2(Block_sum_ch29, sq18_hann, vlmax);
        Block_sum_ch30 = __riscv_vfmul_vv_f32m2(Block_sum_ch30, sq18_hann, vlmax);

        __riscv_vse32_v_f32m2(out_fhog + (size_t)27 * out_plane + out_cell_off, Block_sum_ch27, vlmax);
        __riscv_vse32_v_f32m2(out_fhog + (size_t)28 * out_plane + out_cell_off, Block_sum_ch28, vlmax);
        __riscv_vse32_v_f32m2(out_fhog + (size_t)29 * out_plane + out_cell_off, Block_sum_ch29, vlmax);
        __riscv_vse32_v_f32m2(out_fhog + (size_t)30 * out_plane + out_cell_off, Block_sum_ch30, vlmax);

        inv_tr = inv_br; 
        inv_tl = inv_bl;   
        
        sliding_row_ptr += grid_w;
    }
}

// Full Vectorized Kernel 4 Entry Point
void k4_normalization_hanning(
    const float* raw_hist,
    const float* hann_win,
    float*       out_fhog,
    int          grid_w,
    int          grid_h,
    float        alpha)
{
    static float s_cell_sum_sq[MAX_GRID_WIDTH * MAX_GRID_HEIGHT];

    // Phase 1: Compute Energy Grid (sum of squares of 9 unsigned bins)
    compute_cell_energy_grid(raw_hist, s_cell_sum_sq, grid_w, grid_h);

    const int out_w = grid_w - 2;
    const int out_h = grid_h - 2;

    // Phase 2, 3, 4: Stream Column Chunks across inner grid
    for (size_t col_c = 1; col_c <= (size_t)out_w; ) {
        size_t vlmax = __riscv_vsetvl_e32m2(out_w + 1 - col_c);

        fhog_normalization_chunk(
            raw_hist, out_fhog, hann_win, s_cell_sum_sq,
            out_w, out_h, col_c, alpha, vlmax
        );

        col_c += vlmax;
    }
}

} // namespace vec
