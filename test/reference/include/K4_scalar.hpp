#ifndef K4_H
#define K4_H
#include <stdint.h>
#include <stddef.h>


namespace ref
{
    void k4_normalization_hanning_scalar(
    const float* raw_hist,    // IN  : Output from K3, size: GridH * GridW * 27
    const float* hann_win,    // IN  : 2D Hanning window, size: (GridH-2) * (GridW-2)
    float*       out_fhog,    // OUT : Final tensor, size: (GridH-2) * (GridW-2) * 31
    int          grid_w,      // IN  : Width in cells 
    int          grid_h,      // IN  : Height in cells 
    float        alpha        // IN  : Truncation threshold (0.2f)
);
}


#endif  /* K4_H */