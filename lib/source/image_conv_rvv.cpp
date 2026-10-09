#include "image_conv_rvv.hpp"
#include <riscv_vector.h>

void vec::image_conv_rvv(Image<uint8_t, ImageType::RGB>& Input,
                         Image<uint8_t, ImageType::GRAY>& Output)
{
    size_t n = Input.Getsize();         
    const uint8_t* rgb  = Input.GetPtr(0, 0);
    uint8_t*       gray = Output.GetPtr(0, 0);
    size_t vl = __riscv_vsetvl_e8m2(n);

    for ( vl; n > 0; n -= vl, rgb += 3 * vl, gray += vl) {

        vuint8m2x3_t px = __riscv_vlseg3e8_v_u8m2x3(rgb, vl);
        vuint8m2_t r = __riscv_vget_v_u8m2x3_u8m2(px, 0);
        vuint8m2_t g = __riscv_vget_v_u8m2x3_u8m2(px, 1);
        vuint8m2_t b = __riscv_vget_v_u8m2x3_u8m2(px, 2);

        vuint16m4_t acc = __riscv_vwmulu_vx_u16m4(r, WRRV, vl);
        acc = __riscv_vwmaccu_vx_u16m4(acc, WGRV, g, vl);
        acc = __riscv_vwmaccu_vx_u16m4(acc, WBRV, b, vl);

        vuint8m2_t y = __riscv_vnclipu_wx_u8m2(acc, 8, __RISCV_VXRM_RNU, vl);
        __riscv_vse8_v_u8m2(gray, y, vl);
    }
    
}