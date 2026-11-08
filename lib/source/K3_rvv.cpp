#include "K3_rvv.hpp"
#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

#define MAX_GRID_WIDTH  64
#define NUM_BINS        18
#define MAX_PAD_W       (MAX_GRID_WIDTH + 2)

namespace vec {

struct ScanlineParams {
    float wh;
    float wx;
    float wy;
    float wd;
    int   dx; // Offset in floats: -18 or +18
};

static constexpr ScanlineParams SCAN_PARAMS[4][4] = {
    // ii = 0 (dy = -1, wy_h = 0.625, wy_n = 0.375)
    {
        {0.625f * 0.625f, 0.625f * 0.375f, 0.375f * 0.625f, 0.375f * 0.375f, -18}, // jj = 0
        {0.625f * 0.875f, 0.625f * 0.125f, 0.375f * 0.875f, 0.375f * 0.125f, -18}, // jj = 1
        {0.625f * 0.875f, 0.625f * 0.125f, 0.375f * 0.875f, 0.375f * 0.125f,  18}, // jj = 2
        {0.625f * 0.625f, 0.625f * 0.375f, 0.375f * 0.625f, 0.375f * 0.375f,  18}  // jj = 3
    },
    // ii = 1 (dy = -1, wy_h = 0.875, wy_n = 0.125)
    {
        {0.875f * 0.625f, 0.875f * 0.375f, 0.125f * 0.625f, 0.125f * 0.375f, -18},
        {0.875f * 0.875f, 0.875f * 0.125f, 0.125f * 0.875f, 0.125f * 0.125f, -18},
        {0.875f * 0.875f, 0.875f * 0.125f, 0.125f * 0.875f, 0.125f * 0.125f,  18},
        {0.875f * 0.625f, 0.875f * 0.375f, 0.125f * 0.625f, 0.125f * 0.375f,  18}
    },
    // ii = 2 (dy = +1, wy_h = 0.875, wy_n = 0.125)
    {
        {0.875f * 0.625f, 0.875f * 0.375f, 0.125f * 0.625f, 0.125f * 0.375f, -18},
        {0.875f * 0.875f, 0.875f * 0.125f, 0.125f * 0.875f, 0.125f * 0.125f, -18},
        {0.875f * 0.875f, 0.875f * 0.125f, 0.125f * 0.875f, 0.125f * 0.125f,  18},
        {0.875f * 0.625f, 0.875f * 0.375f, 0.125f * 0.625f, 0.125f * 0.375f,  18}
    },
    // ii = 3 (dy = +1, wy_h = 0.625, wy_n = 0.375)
    {
        {0.625f * 0.625f, 0.625f * 0.375f, 0.375f * 0.625f, 0.375f * 0.375f, -18},
        {0.625f * 0.875f, 0.625f * 0.125f, 0.375f * 0.875f, 0.375f * 0.125f, -18},
        {0.625f * 0.875f, 0.625f * 0.125f, 0.375f * 0.875f, 0.375f * 0.125f,  18},
        {0.625f * 0.625f, 0.625f * 0.375f, 0.375f * 0.625f, 0.375f * 0.375f,  18}
    }
};

enum class RowCat {
    FIRST,  // r == 0: Skip top vertical writes (ii = 0, 1)
    MID,    // 0 < r < grid_h - 1: Write both top and bottom
    LAST    // r == grid_h - 1: Skip bottom vertical writes (ii = 2, 3)
};

// 2-Way Independent Cell Unrolled Column Processing
// Inserts Vertical between Home and Horizontal to eliminate intra-scanline curr_row RAW hazards
// while keeping Block A and Block B paired for full memory pipeline dual-issue overlap.
template <RowCat row_cat, int JJ_VAL, int ii>
static inline __attribute__((always_inline)) void process_pixel_column_2way(
    const float*   cell_A_mag,
    const float*   cell_B_mag,
    const uint8_t* cell_A_bins,
    const uint8_t* cell_B_bins,
    size_t         vlmax,
    vuint32m2_t    v_c_bytes_A,
    vuint32m2_t    v_c_bytes_B,
    int            plane_stride_bytes,
    float*         curr_row,
    float*         vert_row)
{
    float wh = SCAN_PARAMS[ii][JJ_VAL].wh;
    float wx = SCAN_PARAMS[ii][JJ_VAL].wx;
    int   dx_bytes = (JJ_VAL < 2) ? -4 : 4;

    // --- 1. Load inputs for both Block A and Block B ---
    vfloat32m2_t v_mag_A  = __riscv_vlse32_v_f32m2(cell_A_mag + JJ_VAL, 4 * sizeof(float), vlmax);
    vfloat32m2_t v_mag_B  = __riscv_vlse32_v_f32m2(cell_B_mag + JJ_VAL, 4 * sizeof(float), vlmax);

    vuint8mf2_t  v_bin8_A = __riscv_vlse8_v_u8mf2(cell_A_bins + JJ_VAL, 4 * sizeof(uint8_t), vlmax);
    vuint8mf2_t  v_bin8_B = __riscv_vlse8_v_u8mf2(cell_B_bins + JJ_VAL, 4 * sizeof(uint8_t), vlmax);

    vuint32m2_t  v_bin_A  = __riscv_vzext_vf4_u32m2(v_bin8_A, vlmax);
    vuint32m2_t  v_bin_B  = __riscv_vzext_vf4_u32m2(v_bin8_B, vlmax);

    vuint32m2_t v_byte_home_A = __riscv_vmacc_vx_u32m2(v_c_bytes_A, plane_stride_bytes, v_bin_A, vlmax);
    vuint32m2_t v_byte_home_B = __riscv_vmacc_vx_u32m2(v_c_bytes_B, plane_stride_bytes, v_bin_B, vlmax);

    constexpr bool has_vert =
        (row_cat == RowCat::MID) ||
        (row_cat == RowCat::FIRST && ii >= 2) ||
        (row_cat == RowCat::LAST  && ii < 2);

    // ==========================================
    // 1. HOME & VERTICAL PIPELINED DISPATCH
    // ==========================================
    vfloat32m2_t v_val_A = __riscv_vfmul_vf_f32m2(v_mag_A, wh, vlmax);
    vfloat32m2_t v_val_B = __riscv_vfmul_vf_f32m2(v_mag_B, wh, vlmax);

    // Dispatch Home loads
    vfloat32m2_t v_acc_A = __riscv_vluxei32_v_f32m2(curr_row, v_byte_home_A, vlmax);
    vfloat32m2_t v_acc_B = __riscv_vluxei32_v_f32m2(curr_row, v_byte_home_B, vlmax);

    if constexpr (has_vert) {
        float wy = SCAN_PARAMS[ii][JJ_VAL].wy;
        vfloat32m2_t v_vert_A = __riscv_vfmul_vf_f32m2(v_mag_A, wy, vlmax);
        vfloat32m2_t v_vert_B = __riscv_vfmul_vf_f32m2(v_mag_B, wy, vlmax);

        // Dispatch Vertical loads immediately while Home loads are in flight (load-latency hiding)
        vfloat32m2_t v_vacc_A = __riscv_vluxei32_v_f32m2(vert_row, v_byte_home_A, vlmax);
        vfloat32m2_t v_vacc_B = __riscv_vluxei32_v_f32m2(vert_row, v_byte_home_B, vlmax);

        // Compute & store Home (Home data has arrived from L1 cache)
        v_acc_A = __riscv_vfadd_vv_f32m2(v_acc_A, v_val_A, vlmax);
        v_acc_B = __riscv_vfadd_vv_f32m2(v_acc_B, v_val_B, vlmax);

        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_home_A, v_acc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_home_B, v_acc_B, vlmax);

        // Compute & store Vertical (Vertical data has arrived from L1 cache)
        v_vacc_A = __riscv_vfadd_vv_f32m2(v_vacc_A, v_vert_A, vlmax);
        v_vacc_B = __riscv_vfadd_vv_f32m2(v_vacc_B, v_vert_B, vlmax);

        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_home_A, v_vacc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_home_B, v_vacc_B, vlmax);
    } else {
        v_acc_A = __riscv_vfadd_vv_f32m2(v_acc_A, v_val_A, vlmax);
        v_acc_B = __riscv_vfadd_vv_f32m2(v_acc_B, v_val_B, vlmax);

        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_home_A, v_acc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_home_B, v_acc_B, vlmax);
    }

    // ==========================================
    // 2. HORIZONTAL & DIAGONAL PIPELINED DISPATCH
    // ==========================================
    vuint32m2_t v_byte_horiz_A = __riscv_vadd_vx_u32m2(v_byte_home_A, dx_bytes, vlmax);
    vuint32m2_t v_byte_horiz_B = __riscv_vadd_vx_u32m2(v_byte_home_B, dx_bytes, vlmax);

    v_val_A = __riscv_vfmul_vf_f32m2(v_mag_A, wx, vlmax);
    v_val_B = __riscv_vfmul_vf_f32m2(v_mag_B, wx, vlmax);

    // Dispatch Horizontal loads
    v_acc_A = __riscv_vluxei32_v_f32m2(curr_row, v_byte_horiz_A, vlmax);
    v_acc_B = __riscv_vluxei32_v_f32m2(curr_row, v_byte_horiz_B, vlmax);

    if constexpr (has_vert) {
        float wd = SCAN_PARAMS[ii][JJ_VAL].wd;
        vfloat32m2_t v_diag_A = __riscv_vfmul_vf_f32m2(v_mag_A, wd, vlmax);
        vfloat32m2_t v_diag_B = __riscv_vfmul_vf_f32m2(v_mag_B, wd, vlmax);

        // Dispatch Diagonal loads immediately while Horiz loads are in flight (load-latency hiding)
        vfloat32m2_t v_vacc_A = __riscv_vluxei32_v_f32m2(vert_row, v_byte_horiz_A, vlmax);
        vfloat32m2_t v_vacc_B = __riscv_vluxei32_v_f32m2(vert_row, v_byte_horiz_B, vlmax);

        // Compute & store Horizontal
        v_acc_A = __riscv_vfadd_vv_f32m2(v_acc_A, v_val_A, vlmax);
        v_acc_B = __riscv_vfadd_vv_f32m2(v_acc_B, v_val_B, vlmax);

        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_horiz_A, v_acc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_horiz_B, v_acc_B, vlmax);

        // Compute & store Diagonal
        v_vacc_A = __riscv_vfadd_vv_f32m2(v_vacc_A, v_diag_A, vlmax);
        v_vacc_B = __riscv_vfadd_vv_f32m2(v_vacc_B, v_diag_B, vlmax);

        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_horiz_A, v_vacc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_horiz_B, v_vacc_B, vlmax);
    } else {
        v_acc_A = __riscv_vfadd_vv_f32m2(v_acc_A, v_val_A, vlmax);
        v_acc_B = __riscv_vfadd_vv_f32m2(v_acc_B, v_val_B, vlmax);

        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_horiz_A, v_acc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_horiz_B, v_acc_B, vlmax);
    }
}

// 1-Way Fallback for Tail / Odd Widths
template <RowCat row_cat, int JJ_VAL, int ii>
static inline __attribute__((always_inline)) void process_pixel_column_1way(
    const float*   cell_mag,
    const uint8_t* cell_bins,
    size_t         vl,
    vuint32m2_t    v_c_bytes,
    int            plane_stride_bytes,
    float*         curr_row,
    float*         vert_row)
{
    float wh = SCAN_PARAMS[ii][JJ_VAL].wh;
    float wx = SCAN_PARAMS[ii][JJ_VAL].wx;
    int   dx_bytes = (JJ_VAL < 2) ? -4 : 4;

    vfloat32m2_t v_mag  = __riscv_vlse32_v_f32m2(cell_mag + JJ_VAL, 4 * sizeof(float), vl);
    vuint8mf2_t  v_bin8 = __riscv_vlse8_v_u8mf2(cell_bins + JJ_VAL, 4 * sizeof(uint8_t), vl);
    vuint32m2_t  v_bin  = __riscv_vzext_vf4_u32m2(v_bin8, vl);
    vuint32m2_t  v_byte_home = __riscv_vmacc_vx_u32m2(v_c_bytes, plane_stride_bytes, v_bin, vl);

    constexpr bool has_vert =
        (row_cat == RowCat::MID) ||
        (row_cat == RowCat::FIRST && ii >= 2) ||
        (row_cat == RowCat::LAST  && ii < 2);

    // 1. Home & Vertical
    vfloat32m2_t v_val = __riscv_vfmul_vf_f32m2(v_mag, wh, vl);
    vfloat32m2_t v_acc = __riscv_vluxei32_v_f32m2(curr_row, v_byte_home, vl);

    if constexpr (has_vert) {
        float wy = SCAN_PARAMS[ii][JJ_VAL].wy;
        vfloat32m2_t v_vert = __riscv_vfmul_vf_f32m2(v_mag, wy, vl);
        vfloat32m2_t v_vacc = __riscv_vluxei32_v_f32m2(vert_row, v_byte_home, vl);

        v_acc = __riscv_vfadd_vv_f32m2(v_acc, v_val, vl);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_home, v_acc, vl);

        v_vacc = __riscv_vfadd_vv_f32m2(v_vacc, v_vert, vl);
        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_home, v_vacc, vl);
    } else {
        v_acc = __riscv_vfadd_vv_f32m2(v_acc, v_val, vl);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_home, v_acc, vl);
    }

    // 2. Horizontal & Diagonal
    vuint32m2_t v_byte_horiz = __riscv_vadd_vx_u32m2(v_byte_home, dx_bytes, vl);
    v_val = __riscv_vfmul_vf_f32m2(v_mag, wx, vl);
    v_acc = __riscv_vluxei32_v_f32m2(curr_row, v_byte_horiz, vl);

    if constexpr (has_vert) {
        float wd = SCAN_PARAMS[ii][JJ_VAL].wd;
        vfloat32m2_t v_diag = __riscv_vfmul_vf_f32m2(v_mag, wd, vl);
        vfloat32m2_t v_vacc = __riscv_vluxei32_v_f32m2(vert_row, v_byte_horiz, vl);

        v_acc = __riscv_vfadd_vv_f32m2(v_acc, v_val, vl);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_horiz, v_acc, vl);

        v_vacc = __riscv_vfadd_vv_f32m2(v_vacc, v_diag, vl);
        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_horiz, v_vacc, vl);
    } else {
        v_acc = __riscv_vfadd_vv_f32m2(v_acc, v_val, vl);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_horiz, v_acc, vl);
    }
}

template <RowCat row_cat, int ii>
static inline __attribute__((always_inline)) void process_scanline_2way(
    int            r,
    float*         curr_row,
    float*         vert_row,
    const uint8_t* ori_bins,
    const float*   mag,
    int            width,
    int            grid_w,
    int            plane_stride_bytes,
    size_t         vlmax,
    vuint32m2_t    v_c_home)
{
    const int py = r * 4 + ii;
    const float*   row_mag  = mag + py * width;
    const uint8_t* row_bins = ori_bins + py * width;

    int cell_index = 0;
    const int step_2way = 2 * vlmax; // 16 cells per iteration
    for (; cell_index <= grid_w - step_2way; cell_index += step_2way) {
        const float*   cell_A_mag  = row_mag  + (cell_index * 4);
        const float*   cell_B_mag  = row_mag  + ((cell_index + vlmax) * 4);

        const uint8_t* cell_A_bins = row_bins + (cell_index * 4);
        const uint8_t* cell_B_bins = row_bins + ((cell_index + vlmax) * 4);

        vuint32m2_t v_c_bytes_A = __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(v_c_home, cell_index, vlmax), 2, vlmax);
        vuint32m2_t v_c_bytes_B = __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(v_c_home, cell_index + vlmax, vlmax), 2, vlmax);

        process_pixel_column_2way<row_cat, 0, ii>(cell_A_mag, cell_B_mag, cell_A_bins, cell_B_bins, vlmax, v_c_bytes_A, v_c_bytes_B, plane_stride_bytes, curr_row, vert_row);
        process_pixel_column_2way<row_cat, 1, ii>(cell_A_mag, cell_B_mag, cell_A_bins, cell_B_bins, vlmax, v_c_bytes_A, v_c_bytes_B, plane_stride_bytes, curr_row, vert_row);
        process_pixel_column_2way<row_cat, 2, ii>(cell_A_mag, cell_B_mag, cell_A_bins, cell_B_bins, vlmax, v_c_bytes_A, v_c_bytes_B, plane_stride_bytes, curr_row, vert_row);
        process_pixel_column_2way<row_cat, 3, ii>(cell_A_mag, cell_B_mag, cell_A_bins, cell_B_bins, vlmax, v_c_bytes_A, v_c_bytes_B, plane_stride_bytes, curr_row, vert_row);
    }

    while (cell_index < grid_w) {
        size_t vl_tail = __riscv_vsetvl_e32m2(grid_w - cell_index);
        const float*   cell_tail_mag  = row_mag  + (cell_index * 4);
        const uint8_t* cell_tail_bins = row_bins + (cell_index * 4);
        vuint32m2_t v_c_bytes_tail = __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(v_c_home, cell_index, vl_tail), 2, vl_tail);

        process_pixel_column_1way<row_cat, 0, ii>(cell_tail_mag, cell_tail_bins, vl_tail, v_c_bytes_tail, plane_stride_bytes, curr_row, vert_row);
        process_pixel_column_1way<row_cat, 1, ii>(cell_tail_mag, cell_tail_bins, vl_tail, v_c_bytes_tail, plane_stride_bytes, curr_row, vert_row);
        process_pixel_column_1way<row_cat, 2, ii>(cell_tail_mag, cell_tail_bins, vl_tail, v_c_bytes_tail, plane_stride_bytes, curr_row, vert_row);
        process_pixel_column_1way<row_cat, 3, ii>(cell_tail_mag, cell_tail_bins, vl_tail, v_c_bytes_tail, plane_stride_bytes, curr_row, vert_row);
        cell_index += vl_tail;
    }
}

// 100% Contiguous Unit-Stride Transpose
static inline void transpose_and_add_row_planar(
    const float* ring_row,
    float*       planar_hist,
    int          grid_r,
    int          grid_w,
    int          total_cells,
    int          pad_w)
{
    const int out_row_off = grid_r * grid_w;

    for (int b = 0; b < 9; b++) {
        const float* plane_s0 = ring_row + (b)     * pad_w + 1;
        const float* plane_s1 = ring_row + (b + 9) * pad_w + 1;

        float* out_u  = planar_hist + (b)      * total_cells + out_row_off;
        float* out_s0 = planar_hist + (b + 9)  * total_cells + out_row_off;
        float* out_s1 = planar_hist + (b + 18) * total_cells + out_row_off;

        for (int c = 0; c < grid_w; ) {
            size_t vl = __riscv_vsetvl_e32m8(grid_w - c);

            vfloat32m8_t v_s0 = __riscv_vle32_v_f32m8(plane_s0 + c, vl);
            vfloat32m8_t v_s1 = __riscv_vle32_v_f32m8(plane_s1 + c, vl);

            __riscv_vse32_v_f32m8(out_s0 + c, v_s0, vl);
            __riscv_vse32_v_f32m8(out_s1 + c, v_s1, vl);

            vfloat32m8_t v_u = __riscv_vfadd_vv_f32m8(v_s0, v_s1, vl);
            __riscv_vse32_v_f32m8(out_u + c, v_u, vl);

            c += vl;
        }
    }
}

template <RowCat row_cat>
inline void cell_row_process_2way(
    int            r,
    float*         top_row,
    float*         curr_row,
    float*         bot_row,
    const uint8_t* ori_bins,
    const float*   mag,
    int            width,
    int            grid_w,
    int            pad_w)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    const vuint32m2_t v_c_home = __riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vlmax), 1, vlmax);
    const int plane_stride_bytes = pad_w * sizeof(float);

    process_scanline_2way<row_cat, 0>(r, curr_row, top_row, ori_bins, mag, width, grid_w, plane_stride_bytes, vlmax, v_c_home);
    process_scanline_2way<row_cat, 1>(r, curr_row, top_row, ori_bins, mag, width, grid_w, plane_stride_bytes, vlmax, v_c_home);
    process_scanline_2way<row_cat, 2>(r, curr_row, bot_row, ori_bins, mag, width, grid_w, plane_stride_bytes, vlmax, v_c_home);
    process_scanline_2way<row_cat, 3>(r, curr_row, bot_row, ori_bins, mag, width, grid_w, plane_stride_bytes, vlmax, v_c_home);
}

void Histogram_k3(
    const uint8_t* ori_bins,
    const float*   mag,
    float*         planar_hist,
    int            width,
    int            height)
{
    static float s_ring_hist[3 * MAX_PAD_W * NUM_BINS];
    const int k = 4;
    const int grid_w = width  / k;
    const int grid_h = height / k;
    const int pad_w = grid_w + 2;
    const int row_stride = pad_w * NUM_BINS;
    const int total_cells = grid_w * grid_h;
    size_t ring_size = 3 * row_stride;

    for (size_t i = 0; i < ring_size; ) {
        size_t vl = __riscv_vsetvl_e32m8(ring_size - i);
        vfloat32m8_t v_zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        __riscv_vse32_v_f32m8(s_ring_hist + i, v_zero, vl);
        i += vl;
    }

    float* top_row  = s_ring_hist;
    float* curr_row = s_ring_hist + row_stride;
    float* bot_row  = s_ring_hist + 2 * row_stride;

    cell_row_process_2way<RowCat::FIRST>(0, top_row, curr_row, bot_row, ori_bins, mag, width, grid_w, pad_w);

    for (int r = 1; r < grid_h - 1; r++) {
        float* tmp = top_row;
        top_row  = curr_row;
        curr_row = bot_row;
        bot_row  = tmp;

        for (size_t i = 0; i < (size_t)row_stride; ) {
            size_t vl = __riscv_vsetvl_e32m8(row_stride - i);
            vfloat32m8_t v_zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            __riscv_vse32_v_f32m8(bot_row + i, v_zero, vl);
            i += vl;
        }

        cell_row_process_2way<RowCat::MID>(r, top_row, curr_row, bot_row, ori_bins, mag, width, grid_w, pad_w);

        int done_r = r - 1;
        transpose_and_add_row_planar(top_row, planar_hist, done_r, grid_w, total_cells, pad_w);
    }

    // Last row
    float* tmp = top_row;
    top_row  = curr_row;
    curr_row = bot_row;
    bot_row  = tmp;

    for (size_t i = 0; i < (size_t)row_stride; ) {
        size_t vl = __riscv_vsetvl_e32m8(row_stride - i);
        vfloat32m8_t v_zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        __riscv_vse32_v_f32m8(bot_row + i, v_zero, vl);
        i += vl;
    }

    cell_row_process_2way<RowCat::LAST>(grid_h - 1, top_row, curr_row, bot_row, ori_bins, mag, width, grid_w, pad_w);

    int done_r = grid_h - 2;
    transpose_and_add_row_planar(top_row, planar_hist, done_r, grid_w, total_cells, pad_w);

    // Final row output
    transpose_and_add_row_planar(curr_row, planar_hist, (grid_h - 1), grid_w, total_cells, pad_w);
}

} // namespace vec

namespace vec {
void Histogram_k3_opt(
    const uint8_t* ori_bins,
    const float*   mag,
    float*         planar_hist,
    int            width,
    int            height)
{
    Histogram_k3(ori_bins, mag, planar_hist, width, height);
}
} // namespace vec
