#include "K3_rvv.hpp"
#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

#define MAX_GRID_WIDTH  64
#define NUM_BINS        18
#define MAX_PAD_W       (MAX_GRID_WIDTH + 2)
#define MAX_K           16

namespace vec {

enum class RowCat {
    FIRST,  // r == 0: Skip top vertical writes
    MID,    // 0 < r < grid_h - 1: Write both top and bottom
    LAST    // r == grid_h - 1: Skip bottom vertical writes
};


static inline void compute_k_weights(int k, float* w_h, float* w_n, int* nearest) {
    int half_k = k / 2;
    float inv_k = 1.0f / (float)k;
    for (int j = 0; j < half_k; j++) {
        nearest[j]         = -1;
        nearest[k - 1 - j] =  1;

        float b_x = (half_k) + j + 0.5f;
        float home_weight = b_x * inv_k;
        float nbr_weight  = 1.0f - home_weight;

        w_h[j] = home_weight;
        w_n[j] = nbr_weight;

        int mirror = k - 1 - j;
        w_h[mirror] = home_weight;
        w_n[mirror] = nbr_weight;
    }
}

// 2-Way Independent Column Processing: ONLY row_cat is a template
template <int k, RowCat row_cat>
static inline __attribute__((always_inline)) void process_pixel_column_single_2way(
    int             jj,
    int             ii,
    float           wh,
    float           wx,
    float           wy,
    float           wd,
    const float*    cell_A_mag,
    const float*    cell_B_mag,
    const uint32_t* cell_A_bins,
    const uint32_t* cell_B_bins,
    size_t          vlmax,
    vuint32m2_t     v_c_bytes_A,
    vuint32m2_t     v_c_bytes_B,
    int             plane_stride_bytes,
    float*          curr_row,
    float*          vert_row)
{
    constexpr int half_k = k / 2;
    int dx_bytes = (jj < half_k) ? -4 : 4;

    // Load inputs for both Block A and Block B
    vuint32m2_t v_bin_A = __riscv_vlse32_v_u32m2(cell_A_bins + jj, k * sizeof(uint32_t), vlmax);
    vuint32m2_t v_bin_B = __riscv_vlse32_v_u32m2(cell_B_bins + jj, k * sizeof(uint32_t), vlmax);
    vfloat32m2_t v_mag_A = __riscv_vlse32_v_f32m2(cell_A_mag + jj, k * sizeof(float), vlmax);
    vfloat32m2_t v_mag_B = __riscv_vlse32_v_f32m2(cell_B_mag + jj, k * sizeof(float), vlmax);

    vuint32m2_t v_byte_home_A = __riscv_vmacc_vx_u32m2(v_c_bytes_A, plane_stride_bytes, v_bin_A, vlmax);
    vuint32m2_t v_byte_home_B = __riscv_vmacc_vx_u32m2(v_c_bytes_B, plane_stride_bytes, v_bin_B, vlmax);

    const bool has_vert =
        (row_cat == RowCat::MID) ||
        (row_cat == RowCat::FIRST && ii >= half_k) ||
        (row_cat == RowCat::LAST  && ii < half_k);

    // 1. HOME & VERTICAL PIPELINED DISPATCH
    vfloat32m2_t v_val_A = __riscv_vfmul_vf_f32m2(v_mag_A, wh, vlmax);
    vfloat32m2_t v_val_B = __riscv_vfmul_vf_f32m2(v_mag_B, wh, vlmax);

    vfloat32m2_t v_acc_A = __riscv_vluxei32_v_f32m2(curr_row, v_byte_home_A, vlmax);
    vfloat32m2_t v_acc_B = __riscv_vluxei32_v_f32m2(curr_row, v_byte_home_B, vlmax);

    if (has_vert) {
        vfloat32m2_t v_vert_A = __riscv_vfmul_vf_f32m2(v_mag_A, wy, vlmax);
        vfloat32m2_t v_vert_B = __riscv_vfmul_vf_f32m2(v_mag_B, wy, vlmax);

        vfloat32m2_t v_vacc_A = __riscv_vluxei32_v_f32m2(vert_row, v_byte_home_A, vlmax);
        vfloat32m2_t v_vacc_B = __riscv_vluxei32_v_f32m2(vert_row, v_byte_home_B, vlmax);

        v_acc_A = __riscv_vfadd_vv_f32m2(v_acc_A, v_val_A, vlmax);
        v_acc_B = __riscv_vfadd_vv_f32m2(v_acc_B, v_val_B, vlmax);

        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_home_A, v_acc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_home_B, v_acc_B, vlmax);

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

    // 2. HORIZONTAL & DIAGONAL PIPELINED DISPATCH
    int dx_signed = dx_bytes;
    vuint32m2_t v_byte_nbr_A = __riscv_vadd_vx_u32m2(v_byte_home_A, dx_signed, vlmax);
    vuint32m2_t v_byte_nbr_B = __riscv_vadd_vx_u32m2(v_byte_home_B, dx_signed, vlmax);

    vfloat32m2_t v_horiz_A = __riscv_vfmul_vf_f32m2(v_mag_A, wx, vlmax);
    vfloat32m2_t v_horiz_B = __riscv_vfmul_vf_f32m2(v_mag_B, wx, vlmax);

    vfloat32m2_t v_hacc_A = __riscv_vluxei32_v_f32m2(curr_row, v_byte_nbr_A, vlmax);
    vfloat32m2_t v_hacc_B = __riscv_vluxei32_v_f32m2(curr_row, v_byte_nbr_B, vlmax);

    if (has_vert) {
        vfloat32m2_t v_diag_A = __riscv_vfmul_vf_f32m2(v_mag_A, wd, vlmax);
        vfloat32m2_t v_diag_B = __riscv_vfmul_vf_f32m2(v_mag_B, wd, vlmax);

        vfloat32m2_t v_dacc_A = __riscv_vluxei32_v_f32m2(vert_row, v_byte_nbr_A, vlmax);
        vfloat32m2_t v_dacc_B = __riscv_vluxei32_v_f32m2(vert_row, v_byte_nbr_B, vlmax);

        v_hacc_A = __riscv_vfadd_vv_f32m2(v_hacc_A, v_horiz_A, vlmax);
        v_hacc_B = __riscv_vfadd_vv_f32m2(v_hacc_B, v_horiz_B, vlmax);

        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_nbr_A, v_hacc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_nbr_B, v_hacc_B, vlmax);

        v_dacc_A = __riscv_vfadd_vv_f32m2(v_dacc_A, v_diag_A, vlmax);
        v_dacc_B = __riscv_vfadd_vv_f32m2(v_dacc_B, v_diag_B, vlmax);

        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_nbr_A, v_dacc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_nbr_B, v_dacc_B, vlmax);
    } else {
        v_hacc_A = __riscv_vfadd_vv_f32m2(v_hacc_A, v_horiz_A, vlmax);
        v_hacc_B = __riscv_vfadd_vv_f32m2(v_hacc_B, v_horiz_B, vlmax);

        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_nbr_A, v_hacc_A, vlmax);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_nbr_B, v_hacc_B, vlmax);
    }
}

// 1-Way Tail Processing
template <int k, RowCat row_cat>
static inline __attribute__((always_inline)) void process_pixel_column_single_1way(
    int             jj,
    int             ii,
    float           wh,
    float           wx,
    float           wy,
    float           wd,
    const float*    cell_mag,
    const uint32_t* cell_bins,
    size_t          vl,
    vuint32m2_t     v_c_bytes,
    int             plane_stride_bytes,
    float*          curr_row,
    float*          vert_row)
{
    int half_k = k / 2;
    int dx_bytes = (jj < half_k) ? -4 : 4;

    vuint32m2_t v_bin = __riscv_vlse32_v_u32m2(cell_bins + jj, k * sizeof(uint32_t), vl);
    vfloat32m2_t v_mag = __riscv_vlse32_v_f32m2(cell_mag + jj, k * sizeof(float), vl);
    vuint32m2_t v_byte_home = __riscv_vmacc_vx_u32m2(v_c_bytes, plane_stride_bytes, v_bin, vl);

    const bool has_vert =
        (row_cat == RowCat::MID) ||
        (row_cat == RowCat::FIRST && ii >= half_k) ||
        (row_cat == RowCat::LAST  && ii < half_k);

    vfloat32m2_t v_val = __riscv_vfmul_vf_f32m2(v_mag, wh, vl);
    vfloat32m2_t v_acc = __riscv_vluxei32_v_f32m2(curr_row, v_byte_home, vl);

    if (has_vert) {
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

    int dx_signed = dx_bytes;
    vuint32m2_t v_byte_nbr = __riscv_vadd_vx_u32m2(v_byte_home, dx_signed, vl);
    vfloat32m2_t v_horiz = __riscv_vfmul_vf_f32m2(v_mag, wx, vl);
    vfloat32m2_t v_hacc = __riscv_vluxei32_v_f32m2(curr_row, v_byte_nbr, vl);

    if (has_vert) {
        vfloat32m2_t v_diag = __riscv_vfmul_vf_f32m2(v_mag, wd, vl);
        vfloat32m2_t v_dacc = __riscv_vluxei32_v_f32m2(vert_row, v_byte_nbr, vl);

        v_hacc = __riscv_vfadd_vv_f32m2(v_hacc, v_horiz, vl);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_nbr, v_hacc, vl);

        v_dacc = __riscv_vfadd_vv_f32m2(v_dacc, v_diag, vl);
        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_nbr, v_dacc, vl);
    } else {
        v_hacc = __riscv_vfadd_vv_f32m2(v_hacc, v_horiz, vl);
        __riscv_vsuxei32_v_f32m2(curr_row, v_byte_nbr, v_hacc, vl);
    }
}

template <int k, RowCat row_cat>
inline void process_scanline_single_2way(
    int             r,
    int             ii,
    const float*    w_h,
    const float*    w_n,
    float*          curr_row,
    float*          vert_row,
    const uint32_t* ori_bins,
    const float*    mag,
    int             width,
    int             grid_w,
    int             plane_stride_bytes,
    size_t          vlmax,
    vuint32m2_t     v_c_home)
{
    const int py = r * k + ii;
    const int line_offset = py * width;
    const float*    line_mag  = mag + line_offset;
    const uint32_t* line_bins = ori_bins + line_offset;

    float wy_h = w_h[ii];
    float wy_n = w_n[ii];

    int cell_index = 0;

    while (cell_index + 2 * (int)vlmax <= grid_w) {
        int c_A = cell_index;
        int c_B = cell_index + vlmax;

        const float*    cell_A_mag  = line_mag  + c_A * k;
        const float*    cell_B_mag  = line_mag  + c_B * k;
        const uint32_t* cell_A_bins = line_bins + c_A * k;
        const uint32_t* cell_B_bins = line_bins + c_B * k;

        vuint32m2_t v_c_bytes_A = __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(v_c_home, c_A, vlmax), 2, vlmax);
        vuint32m2_t v_c_bytes_B = __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(v_c_home, c_B, vlmax), 2, vlmax);

        for (int jj = 0; jj < k; jj++) {
            float wx_h = w_h[jj];
            float wx_n = w_n[jj];
            float wh = wy_h * wx_h;
            float wx = wy_h * wx_n;
            float wy = wy_n * wx_h;
            float wd = wy_n * wx_n;

            process_pixel_column_single_2way<k, row_cat>(
                jj, ii, wh, wx, wy, wd,
                cell_A_mag, cell_B_mag, cell_A_bins, cell_B_bins,
                vlmax, v_c_bytes_A, v_c_bytes_B,
                plane_stride_bytes, curr_row, vert_row);
        }

        cell_index += 2 * vlmax;
    }

    while (cell_index < grid_w) {
        size_t vl_tail = __riscv_vsetvl_e32m2(grid_w - cell_index);
        const float*    cell_tail_mag  = line_mag  + cell_index * k;
        const uint32_t* cell_tail_bins = line_bins + cell_index * k;
        vuint32m2_t v_c_bytes_tail = __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(v_c_home, cell_index, vl_tail), 2, vl_tail);

        for (int jj = 0; jj < k; jj++) {
            float wx_h = w_h[jj];
            float wx_n = w_n[jj];
            float wh = wy_h * wx_h;
            float wx = wy_h * wx_n;
            float wy = wy_n * wx_h;
            float wd = wy_n * wx_n;

            process_pixel_column_single_1way<k, row_cat>(
                jj, ii, wh, wx, wy, wd,
                cell_tail_mag, cell_tail_bins, vl_tail,
                v_c_bytes_tail, plane_stride_bytes,
                curr_row, vert_row);
        }
        cell_index += vl_tail;
    }
}

static inline void transpose_and_add_row_planar_single(
    const float* ring_row,
    float*       planar_hist,
    int          grid_r,
    int          grid_w,
    int          total_cells,
    int          pad_w)
{
    const int out_row_off = grid_r * grid_w;
    const size_t vlmax = __riscv_vsetvlmax_e32m4();

    // 2-way unrolled across bins with LMUL=4
    for (int b = 0; b < 8; b += 2) {
        int b0 = b;
        int b1 = b + 1;

        const float* plane_s0_A = ring_row + (b0)     * pad_w + 1;
        const float* plane_s1_A = ring_row + (b0 + 9) * pad_w + 1;
        const float* plane_s0_B = ring_row + (b1)     * pad_w + 1;
        const float* plane_s1_B = ring_row + (b1 + 9) * pad_w + 1;

        float* out_u_A  = planar_hist + (b0)      * total_cells + out_row_off;
        float* out_s0_A = planar_hist + (b0 + 9)  * total_cells + out_row_off;
        float* out_s1_A = planar_hist + (b0 + 18) * total_cells + out_row_off;

        float* out_u_B  = planar_hist + (b1)      * total_cells + out_row_off;
        float* out_s0_B = planar_hist + (b1 + 9)  * total_cells + out_row_off;
        float* out_s1_B = planar_hist + (b1 + 18) * total_cells + out_row_off;

        int c = 0;
        // Main loop: full vlmax chunks without vsetvl overhead
        while (c + (int)vlmax <= grid_w) {
            vfloat32m4_t v_s0_A = __riscv_vle32_v_f32m4(plane_s0_A + c, vlmax);
            vfloat32m4_t v_s1_A = __riscv_vle32_v_f32m4(plane_s1_A + c, vlmax);
            vfloat32m4_t v_s0_B = __riscv_vle32_v_f32m4(plane_s0_B + c, vlmax);
            vfloat32m4_t v_s1_B = __riscv_vle32_v_f32m4(plane_s1_B + c, vlmax);

            vfloat32m4_t v_u_A = __riscv_vfadd_vv_f32m4(v_s0_A, v_s1_A, vlmax);
            vfloat32m4_t v_u_B = __riscv_vfadd_vv_f32m4(v_s0_B, v_s1_B, vlmax);

            __riscv_vse32_v_f32m4(out_s0_A + c, v_s0_A, vlmax);
            __riscv_vse32_v_f32m4(out_s1_A + c, v_s1_A, vlmax);
            __riscv_vse32_v_f32m4(out_u_A  + c, v_u_A,  vlmax);

            __riscv_vse32_v_f32m4(out_s0_B + c, v_s0_B, vlmax);
            __riscv_vse32_v_f32m4(out_s1_B + c, v_s1_B, vlmax);
            __riscv_vse32_v_f32m4(out_u_B  + c, v_u_B,  vlmax);

            c += vlmax;
        }
        // Tail loop: remainder
        while (c < grid_w) {
            size_t vl = __riscv_vsetvl_e32m4(grid_w - c);

            vfloat32m4_t v_s0_A = __riscv_vle32_v_f32m4(plane_s0_A + c, vl);
            vfloat32m4_t v_s1_A = __riscv_vle32_v_f32m4(plane_s1_A + c, vl);
            vfloat32m4_t v_s0_B = __riscv_vle32_v_f32m4(plane_s0_B + c, vl);
            vfloat32m4_t v_s1_B = __riscv_vle32_v_f32m4(plane_s1_B + c, vl);

            vfloat32m4_t v_u_A = __riscv_vfadd_vv_f32m4(v_s0_A, v_s1_A, vl);
            vfloat32m4_t v_u_B = __riscv_vfadd_vv_f32m4(v_s0_B, v_s1_B, vl);

            __riscv_vse32_v_f32m4(out_s0_A + c, v_s0_A, vl);
            __riscv_vse32_v_f32m4(out_s1_A + c, v_s1_A, vl);
            __riscv_vse32_v_f32m4(out_u_A  + c, v_u_A,  vl);

            __riscv_vse32_v_f32m4(out_s0_B + c, v_s0_B, vl);
            __riscv_vse32_v_f32m4(out_s1_B + c, v_s1_B, vl);
            __riscv_vse32_v_f32m4(out_u_B  + c, v_u_B,  vl);

            c += vl;
        }
    }

    // Remainder bin b = 8 with LMUL=4
    {
        int b = 8;
        const float* plane_s0 = ring_row + (b)     * pad_w + 1;
        const float* plane_s1 = ring_row + (b + 9) * pad_w + 1;

        float* out_u  = planar_hist + (b)      * total_cells + out_row_off;
        float* out_s0 = planar_hist + (b + 9)  * total_cells + out_row_off;
        float* out_s1 = planar_hist + (b + 18) * total_cells + out_row_off;

        int c = 0;
        while (c + (int)vlmax <= grid_w) {
            vfloat32m4_t v_s0 = __riscv_vle32_v_f32m4(plane_s0 + c, vlmax);
            vfloat32m4_t v_s1 = __riscv_vle32_v_f32m4(plane_s1 + c, vlmax);

            vfloat32m4_t v_u = __riscv_vfadd_vv_f32m4(v_s0, v_s1, vlmax);

            __riscv_vse32_v_f32m4(out_s0 + c, v_s0, vlmax);
            __riscv_vse32_v_f32m4(out_s1 + c, v_s1, vlmax);
            __riscv_vse32_v_f32m4(out_u  + c, v_u,  vlmax);

            c += vlmax;
        }
        while (c < grid_w) {
            size_t vl = __riscv_vsetvl_e32m4(grid_w - c);

            vfloat32m4_t v_s0 = __riscv_vle32_v_f32m4(plane_s0 + c, vl);
            vfloat32m4_t v_s1 = __riscv_vle32_v_f32m4(plane_s1 + c, vl);

            vfloat32m4_t v_u = __riscv_vfadd_vv_f32m4(v_s0, v_s1, vl);

            __riscv_vse32_v_f32m4(out_s0 + c, v_s0, vl);
            __riscv_vse32_v_f32m4(out_s1 + c, v_s1, vl);
            __riscv_vse32_v_f32m4(out_u  + c, v_u,  vl);

            c += vl;
        }
    }
}

// Fast 2-way unrolled row zeroing using LMUL=4 with single v_zero initialization
static inline void clear_row_buffer_m4_2way(float* row, size_t count) {
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    const vfloat32m4_t v_zero = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    const size_t step = 2 * vlmax;
    size_t i = 0;

    while (i + step <= count) {
        __riscv_vse32_v_f32m4(row + i,         v_zero, vlmax);
        __riscv_vse32_v_f32m4(row + i + vlmax, v_zero, vlmax);
        i += step;
    }
    while (i < count) {
        size_t vl = __riscv_vsetvl_e32m4(count - i);
        __riscv_vse32_v_f32m4(row + i, v_zero, vl);
        i += vl;
    }
}

template <int k, RowCat row_cat>
inline void cell_row_process_single_2way(
    int             r,
    const float*    w_h,
    const float*    w_n,
    float*          top_row,
    float*          curr_row,
    float*          bot_row,
    const uint32_t* ori_bins,
    const float*    mag,
    int             width,
    int             grid_w,
    int             pad_w)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    const vuint32m2_t v_c_home = __riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vlmax), 1, vlmax);
    const int plane_stride_bytes = pad_w * sizeof(float);
    constexpr int half_k = k / 2;

    for (int ii = 0; ii < k; ii++) {
        float* v_row = (ii < half_k) ? top_row : bot_row;
        process_scanline_single_2way<k, row_cat>(
            r, ii, w_h, w_n, curr_row, v_row,
            ori_bins, mag, width, grid_w, plane_stride_bytes, vlmax, v_c_home);
    }
}

template <int k>
void Histogram_k3_single_template(
    const uint32_t* ori_bins,
    const float*    mag,
    float*          planar_hist,
    int             width,
    int             height)
{
    const int grid_w = width / k;
    const int grid_h = height / k;
    const int pad_w  = grid_w + 2;
    const int row_stride = NUM_BINS * pad_w;
    const int total_cells = grid_w * grid_h;

    float w_h[MAX_K];
    float w_n[MAX_K];
    int   nearest[MAX_K];
    compute_k_weights(k, w_h, w_n, nearest);

    static float s_ring_hist[3 * NUM_BINS * MAX_PAD_W];

    float* ring_base = s_ring_hist;
    float* r0 = ring_base + 0 * row_stride;
    float* r1 = ring_base + 1 * row_stride;
    float* r2 = ring_base + 2 * row_stride;

    // Initial setup: r0 and r1 cleared
    clear_row_buffer_m4_2way(r0, 2 * row_stride);

    float* top_row  = r0;
    float* curr_row = r1;
    float* bot_row  = r2;

    // bot_row is about to receive vertical additions for r=0
    clear_row_buffer_m4_2way(bot_row, row_stride);

    cell_row_process_single_2way<k, RowCat::FIRST>(0, w_h, w_n, top_row, curr_row, bot_row, ori_bins, mag, width, grid_w, pad_w);

    for (int r = 1; r < grid_h - 1; r++) {
        float* tmp = top_row;
        top_row  = curr_row;
        curr_row = bot_row;
        bot_row  = tmp;

        // Process row r: writes to curr_row, top_row, and bot_row
        cell_row_process_single_2way<k, RowCat::MID>(r, w_h, w_n, top_row, curr_row, bot_row, ori_bins, mag, width, grid_w, pad_w);

        // top_row has received all contributions and is finished!
        int done_r = r - 1;
        transpose_and_add_row_planar_single(top_row, planar_hist, done_r, grid_w, total_cells, pad_w);

        // Now top_row can be safely cleared for future rotations
        clear_row_buffer_m4_2way(top_row, row_stride);
    }

    float* tmp = top_row;
    top_row  = curr_row;
    curr_row = bot_row;
    bot_row  = tmp;

    cell_row_process_single_2way<k, RowCat::LAST>(grid_h - 1, w_h, w_n, top_row, curr_row, bot_row, ori_bins, mag, width, grid_w, pad_w);

    int done_r = grid_h - 2;
    transpose_and_add_row_planar_single(top_row, planar_hist, done_r, grid_w, total_cells, pad_w);

    transpose_and_add_row_planar_single(curr_row, planar_hist, (grid_h - 1), grid_w, total_cells, pad_w);
}




// Explicit template instantiations
template void Histogram_k3_single_template<4>(const uint32_t*, const float*, float*, int, int);
template void Histogram_k3_single_template<8>(const uint32_t*, const float*, float*, int, int);
template void Histogram_k3_single_template<16>(const uint32_t*, const float*, float*, int, int);

template <int k>
void Histogram_k3(
    const uint32_t* ori_bins,
    const float*    mag,
    float*          planar_hist,
    int             width,
    int             height)
{
    Histogram_k3_single_template<k>(ori_bins, mag, planar_hist, width, height);
}

template void Histogram_k3<4>(const uint32_t*, const float*, float*, int, int);
template void Histogram_k3<8>(const uint32_t*, const float*, float*, int, int);
template void Histogram_k3<16>(const uint32_t*, const float*, float*, int, int);

} // namespace vec
