#include <riscv_vector.h>
#include "K3_rvv.hpp"

#define MAX_GRID_WIDTH  64
#define NUM_BINS        18
#define MAX_PAD_W       (MAX_GRID_WIDTH + 2)
namespace vec {

struct ScanlineParams {
    float wh;
    float wx;
    float wy;
    float wd;
    int   dx; // Offset in floats: -18 (left cell) or +18 (right cell)
};

static const ScanlineParams SCAN_PARAMS[4][4] = {
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

template <RowCat row_cat, int JJ_VAL, int ii>
static inline void process_pixel_column(
    vfloat32m2x4_t v_m,
    vuint8mf2x4_t  v_b,
    vuint32m2_t    v_c_base,
    float*         curr_row,
    float*         vert_row,
    size_t         vl)
{
    vfloat32m2_t v_mag  = __riscv_vget_v_f32m2x4_f32m2(v_m, JJ_VAL);
    vuint8mf2_t  v_bin8 = __riscv_vget_v_u8mf2x4_u8mf2(v_b, JJ_VAL);
    vuint32m2_t  v_bin  = __riscv_vzext_vf4_u32m2(v_bin8, vl);

    float wh = SCAN_PARAMS[ii][JJ_VAL].wh;
    float wx = SCAN_PARAMS[ii][JJ_VAL].wx;
    int   dx = SCAN_PARAMS[ii][JJ_VAL].dx; // -18 or +18

    // Pure vector additions: NO vmacc inside the loop!
    vuint32m2_t v_idx_home  = __riscv_vadd_vv_u32m2(v_c_base, v_bin, vl);
    vuint32m2_t v_idx_horiz = __riscv_vadd_vx_u32m2(v_idx_home, dx, vl);

    vuint32m2_t v_byte_home  = __riscv_vsll_vx_u32m2(v_idx_home, 2, vl);
    vuint32m2_t v_byte_horiz = __riscv_vsll_vx_u32m2(v_idx_horiz, 2, vl);

    // 1. Home Cell (Always written)
    vfloat32m2_t v_wh = __riscv_vfmul_vf_f32m2(v_mag, wh, vl);
    vfloat32m2_t v_curr = __riscv_vluxei32_v_f32m2(curr_row, v_byte_home, vl);
    v_curr = __riscv_vfadd_vv_f32m2(v_curr, v_wh, vl);
    __riscv_vsuxei32_v_f32m2(curr_row, v_byte_home, v_curr, vl);

    // 2. Horizontal Neighbor (Always written)
    vfloat32m2_t v_wx = __riscv_vfmul_vf_f32m2(v_mag, wx, vl);
    vfloat32m2_t v_h  = __riscv_vluxei32_v_f32m2(curr_row, v_byte_horiz, vl);
    v_h = __riscv_vfadd_vv_f32m2(v_h, v_wx, vl);
    __riscv_vsuxei32_v_f32m2(curr_row, v_byte_horiz, v_h, vl);

    // 3. Vertical & Diagonal Neighbors (Only compiled if neighbor exists!)
    constexpr bool has_vert =
        (row_cat == RowCat::MID) ||
        (row_cat == RowCat::FIRST && ii >= 2) ||
        (row_cat == RowCat::LAST  && ii < 2);

    if constexpr (has_vert) {
        float wy = SCAN_PARAMS[ii][JJ_VAL].wy;
        float wd = SCAN_PARAMS[ii][JJ_VAL].wd;

        vfloat32m2_t v_wy = __riscv_vfmul_vf_f32m2(v_mag, wy, vl);
        vfloat32m2_t v_wd = __riscv_vfmul_vf_f32m2(v_mag, wd, vl);

        // Vertical neighbor
        vfloat32m2_t v_y = __riscv_vluxei32_v_f32m2(vert_row, v_byte_home, vl);
        v_y = __riscv_vfadd_vv_f32m2(v_y, v_wy, vl);
        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_home, v_y, vl);

        // Diagonal neighbor
        vfloat32m2_t v_d = __riscv_vluxei32_v_f32m2(vert_row, v_byte_horiz, vl);
        v_d = __riscv_vfadd_vv_f32m2(v_d, v_wd, vl);
        __riscv_vsuxei32_v_f32m2(vert_row, v_byte_horiz, v_d, vl);
    }
}

template <RowCat row_cat, int ii>
static inline void process_scanline(
    int            r,
    float*         curr_row,
    float*         vert_row,
    const uint8_t* ori_bins,
    const float*   mag,
    int            width,
    int            grid_w,
    size_t         vlmax,
    vuint32m2_t    v_c_home)
{
    int py = (r << 2) + ii;
    const float*   row_mag  = mag + py * width;
    const uint8_t* row_bins = ori_bins + py * width;
    int c = 0;
    for (; c <= grid_w - (int)vlmax; c += vlmax) {
        vfloat32m2x4_t v_m = __riscv_vlseg4e32_v_f32m2x4(row_mag + c * 4, vlmax);
        vuint8mf2x4_t  v_b = __riscv_vlseg4e8_v_u8mf2x4(row_bins + c * 4, vlmax);

        // Single vector add with scalar (c * 18): shifts the pre-multiplied base by c cells
        vuint32m2_t v_c_base = __riscv_vadd_vx_u32m2(v_c_home, c * 18, vlmax);

        process_pixel_column<row_cat, 0, ii>(v_m, v_b, v_c_base, curr_row, vert_row, vlmax);
        process_pixel_column<row_cat, 1, ii>(v_m, v_b, v_c_base, curr_row, vert_row, vlmax);
        process_pixel_column<row_cat, 2, ii>(v_m, v_b, v_c_base, curr_row, vert_row, vlmax);
        process_pixel_column<row_cat, 3, ii>(v_m, v_b, v_c_base, curr_row, vert_row, vlmax);
    }
    if (c < grid_w) {
        size_t vl_tail = __riscv_vsetvl_e32m2(grid_w - c);
        vfloat32m2x4_t v_m = __riscv_vlseg4e32_v_f32m2x4(row_mag + c * 4, vl_tail);
        vuint8mf2x4_t  v_b = __riscv_vlseg4e8_v_u8mf2x4(row_bins + c * 4, vl_tail);
        vuint32m2_t v_c_base_tail = __riscv_vadd_vx_u32m2(v_c_home, c * 18, vl_tail);
        process_pixel_column<row_cat, 0, ii>(v_m, v_b, v_c_base_tail, curr_row, vert_row, vl_tail);
        process_pixel_column<row_cat, 1, ii>(v_m, v_b, v_c_base_tail, curr_row, vert_row, vl_tail);
        process_pixel_column<row_cat, 2, ii>(v_m, v_b, v_c_base_tail, curr_row, vert_row, vl_tail);
        process_pixel_column<row_cat, 3, ii>(v_m, v_b, v_c_base_tail, curr_row, vert_row, vl_tail);

    }
}

// Stream completed cell row immediately to planar output using LMUL=4 vector transpose
static inline void transpose_and_add_row(
    const float* ring_row,
    float*       planar_hist,
    int          grid_r,
    int          grid_w,
    int          total_cells)
{
    const float* row_base = ring_row + NUM_BINS; // Skip left border padding
    const int out_row_off = grid_r * grid_w;
    const ptrdiff_t stride = NUM_BINS * sizeof(float);

    for (int b = 0; b < 9; b++) {
        float* out_u  = planar_hist + (b)      * total_cells + out_row_off;
        float* out_s0 = planar_hist + (b + 9)  * total_cells + out_row_off;
        float* out_s1 = planar_hist + (b + 18) * total_cells + out_row_off;

        for (int c = 0; c < grid_w; ) {
            size_t vl = __riscv_vsetvl_e32m4(grid_w - c);

            vfloat32m4_t v_s0 = __riscv_vlse32_v_f32m4(row_base + c * NUM_BINS + b, stride, vl);
            vfloat32m4_t v_s1 = __riscv_vlse32_v_f32m4(row_base + c * NUM_BINS + b + 9, stride, vl);

            __riscv_vse32_v_f32m4(out_s0 + c, v_s0, vl);
            __riscv_vse32_v_f32m4(out_s1 + c, v_s1, vl);

            vfloat32m4_t v_u = __riscv_vfadd_vv_f32m4(v_s0, v_s1, vl);
            __riscv_vse32_v_f32m4(out_u + c, v_u, vl);

            c += vl;
        }
    }
}

template <RowCat row_cat>
inline void cell_row_process(
    int            r,
    float*         top_row,
    float*         curr_row,
    float*         bot_row,
    const uint8_t* ori_bins,
    const float*   mag,
    int            width,
    int            grid_w)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    const vuint32m2_t v_base_id = __riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vlmax), 1, vlmax);
    // Multiply once outside the scanline loops!
    const vuint32m2_t v_c_home  = __riscv_vmul_vx_u32m2(v_base_id, 18, vlmax);

    // Scanlines 0 & 1: neighbor is top_row (if exists)
    process_scanline<row_cat, 0>(r, curr_row, top_row, ori_bins, mag, width, grid_w, vlmax, v_c_home);
    process_scanline<row_cat, 1>(r, curr_row, top_row, ori_bins, mag, width, grid_w, vlmax, v_c_home);

    // Scanlines 2 & 3: neighbor is bot_row (if exists)
    process_scanline<row_cat, 2>(r, curr_row, bot_row, ori_bins, mag, width, grid_w, vlmax, v_c_home);
    process_scanline<row_cat, 3>(r, curr_row, bot_row, ori_bins, mag, width, grid_w, vlmax, v_c_home);
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

    cell_row_process<RowCat::FIRST>(0, top_row, curr_row, bot_row, ori_bins, mag, width, grid_w);

    // Rows 1 to grid_h - 2
    int r = 1;
    for (; r < grid_h - 1; r++) {
        // OPTIMIZED: 3-way circular pointer rotation (zero division, zero multiplication!)
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

        cell_row_process<RowCat::MID>(r, top_row, curr_row, bot_row, ori_bins, mag, width, grid_w);

        int done_r = r - 1;
        transpose_and_add_row(top_row, planar_hist, done_r, grid_w, total_cells);
    }

    // Last row (grid_h - 1)
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

    cell_row_process<RowCat::LAST>(r, top_row, curr_row, bot_row, ori_bins, mag, width, grid_w);

    int done_r = r - 1;
    transpose_and_add_row(top_row, planar_hist, done_r, grid_w, total_cells);

    // Final row output
    transpose_and_add_row(curr_row, planar_hist, (grid_h - 1), grid_w, total_cells);
}

} // namespace vec
