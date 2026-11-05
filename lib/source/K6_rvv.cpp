
#include <riscv_vector.h>
#include "K6_rvv.hpp"
namespace vec {

float subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;
    if (divisor == 0)
        return 0;

    return 0.5 * (right - left) / divisor;
}

/* Max of the vl columns col_ptr[0 .. vl-1] of the response.
   Each lane walks down its column and keeps the largest value and the first row that has it
   (strict >, like the scalar scan). Then the lanes are reduced in registers: the largest
   value, the smallest row that has it, the leftmost column with that row. That is the first
   maximum in row-major order, the same cell the scalar loop picks, also on ties. */
static inline void find_max(
    size_t       vl,
    const float* col_ptr,
    const int    height,
    const int    stride,
    int&         out_max_r,
    int&         out_max_c,
    float&       out_max_val)
{
    vuint32m8_t v_max_row_id = __riscv_vmv_v_x_u32m8(0, vl);
    const float *ptr = col_ptr;
    vfloat32m8_t row_data = __riscv_vle32_v_f32m8(ptr, vl);

    for (size_t j = 1; j < (size_t)height; j += 1) {
        ptr += stride;
        vfloat32m8_t next_row_data = __riscv_vle32_v_f32m8(ptr, vl);
        vbool4_t mask = __riscv_vmfgt_vv_f32m8_b4(next_row_data, row_data, vl);
        row_data = __riscv_vmerge_vvm_f32m8(row_data, next_row_data, mask, vl);   // value and row
        v_max_row_id = __riscv_vmerge_vxm_u32m8(v_max_row_id, (uint32_t)j, mask, vl); // move together
    }

    /* largest value of the vl columns */
    vfloat32m1_t v_max = __riscv_vfredmax_vs_f32m8_f32m1(
        row_data, __riscv_vfmv_s_f_f32m1(-__builtin_huge_valf(), 1), vl);
    const float max_val = __riscv_vfmv_f_s_f32m1_f32(v_max);

    /* smallest row among the columns that have it */
    vbool4_t at_max = __riscv_vmfeq_vf_f32m8_b4(row_data, max_val, vl);
    vuint32m1_t v_row = __riscv_vredminu_vs_u32m8_u32m1_m(
        at_max, v_max_row_id, __riscv_vmv_s_x_u32m1(0xffffffffu, 1), vl);
    const uint32_t max_r = __riscv_vmv_x_s_u32m1_u32(v_row);

    /* leftmost column with that value in that row */
    vbool4_t pick = __riscv_vmand_mm_b4(at_max, __riscv_vmseq_vx_u32m8_b4(v_max_row_id, max_r, vl), vl);

    out_max_r   = (int)max_r;
    out_max_c   = (int)__riscv_vfirst_m_b4(pick, vl);
    out_max_val = max_val;
}

void k6_detection(
    const float* res,
    const int    Width,
    const int    height,
    const int    cell_size,
    float&       location_x,
    float&       location_y)
{
    float peak = res[0];      // same start as the scalar: cell (0, 0)
    int   max_x = 0;
    int   max_y = 0;

    for (int c = 0; c < Width; ) {
        size_t vl = __riscv_vsetvl_e32m8(Width - c);

        int   chunk_max_r = 0;
        int   chunk_max_c = 0;
        float chunk_max_val = 0.0f;

        find_max(vl, res + c, height, Width, chunk_max_r, chunk_max_c, chunk_max_val);

        /* a later chunk has later columns: on a tie it wins only with an earlier row */
        if (chunk_max_val > peak || (chunk_max_val == peak && chunk_max_r < max_y)) {
            peak = chunk_max_val;
            max_y = chunk_max_r;
            max_x = c + chunk_max_c;
        }

        c += (int)vl;
    }

    /* K5's response is periodic, zero shift at (0, 0) (top-left, no fftshift): the neighbours
       wrap around the edges, and a peak past the middle is a negative shift. Same steps as
       ref::k6_detection. */
    int xl = (max_x == 0) ? Width - 1 : max_x - 1;
    int xr = (max_x == Width - 1) ? 0 : max_x + 1;
    int yu = (max_y == 0) ? height - 1 : max_y - 1;
    int yd = (max_y == height - 1) ? 0 : max_y + 1;

    float dx = subPixelPeak(res[max_y * Width + xl], peak, res[max_y * Width + xr]);
    float dy = subPixelPeak(res[yu * Width + max_x], peak, res[yd * Width + max_x]);

    float sx = (float)max_x + dx;
    float sy = (float)max_y + dy;
    if (max_x >= Width / 2)  sx -= (float)Width;      /* e.g. index 31 of 32 -> shift -1 */
    if (max_y >= height / 2) sy -= (float)height;

    location_x = sx * (float)cell_size;               /* template pixels */
    location_y = sy * (float)cell_size;
}

} // namespace vec
