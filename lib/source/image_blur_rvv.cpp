
#include "image_blur_rvv.hpp"
#include <riscv_vector.h>

static vuint16m2_t row_sum3(const uint8_t *row, int x, size_t vl)
{
    vuint8m1_t l = __riscv_vle8_v_u8m1(row + x - 1, vl);
    vuint8m1_t m = __riscv_vle8_v_u8m1(row + x,     vl);
    vuint8m1_t r = __riscv_vle8_v_u8m1(row + x + 1, vl);

    vuint16m2_t s = __riscv_vwaddu_vv_u16m2(l, m, vl);   
    return __riscv_vwaddu_wv_u16m2(s, r, vl);            
}

void vec::blur3x3_rvv(Image<uint8_t, ImageType::GRAY>& Input,
                    Image<uint8_t, ImageType::GRAY>& Output)
{
    const int W  = Output.GetWidth();
    const int H  = Output.GetHeight();
    const int SS = Input.GetWidth();
    const uint8_t *src = Input.GetPtr(0, 0);
    uint8_t       *dst = Output.GetPtr(0, 0);

    int    x = 1;
    size_t n = W;


    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e8m1(n);

        // prime the window: input rows 0 and 1 for this strip
        vuint16m2_t r0 = row_sum3(src + 0 * SS, x, vl);
        vuint16m2_t r1 = row_sum3(src + 1 * SS, x, vl);
        vuint16m2_t s ; 
        vuint16m2_t t ; 

        bool flag = true ; 
        for (int y = 0; y < H; y++)
        {
            vuint16m2_t r2 = row_sum3(src + (y + 2) * SS, x, vl);  

            if(flag == true){
                s = __riscv_vadd_vv_u16m2(r0, r1, vl);
                s = __riscv_vadd_vv_u16m2(s, r2, vl);
                flag = false;
            }
           else {
            s= __riscv_vadd_vv_u16m2(s, r2, vl);
           }
            t  = __riscv_vmulhu_vx_u16m2(s, 7282, vl);

            vuint8m1_t o = __riscv_vncvt_x_x_w_u8m1(t , vl);
            __riscv_vse8_v_u8m1(dst + y * W + (x - 1), o, vl);
            s = __riscv_vsub_vv_u16m2(s, r0, vl);
            r0 = r1;  
            r1 = r2;
        }

        x += vl;
        n -= vl;
    }
}