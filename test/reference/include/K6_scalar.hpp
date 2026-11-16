#ifndef K6_H
#define K6_H
#include <stdint.h>
#include <stddef.h>


namespace ref
{
    void k6_detection(
    const float* res,    // IN  : Output from K5, response 2d heat map 
    const int Width,
    const int height,
    const int cell_size,
    float &location_x,   // OUT : Displacement in pixels (float for subpixel precision)
    float &location_y,   // OUT : Displacement in pixels (float for subpixel precision)
    float *peak_value = nullptr   // OUT (optional): the largest response value. Multi-scale
                                  // detection compares it between scales.
); 
}


#endif  /* K6_H */