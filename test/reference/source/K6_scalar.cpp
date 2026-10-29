#include <cmath>
#include <cfloat>
#include "K6_scalar.hpp"

#define NUM_SECTORS 9
#define NUM_BINS    27

float ref::subPixelPeak(float left, float center, float right)
{ 
    // solve parapolic equation ax^2 + bx + c 
    // at x = 0 ; f(0)  = c ; solve for  x = -1 ;x = 1 and find max by differntiation of f(x) = 0
    float divisor = 2 * center - right - left;
    if (divisor == 0)
        return 0;
    
    return 0.5 * (right - left) / divisor;
}

void ref::k6_detection(
    const float* res,    // IN  : Output from K5, response 2d heat map 
    const int Width,
    const int height,
    const int cell_size,
    float &location_x,   // OUT : Displacement in pixels (float for subpixel precision)
    float &location_y    // OUT : Displacement in pixels (float for subpixel precision)
){
    //--------------- find max location -------------------//      
    float peak = *res;
    int max_x = 0;
    int max_y = 0;
    const float *curr_indx;

    for (int i = 0; i < height; i++) {
        curr_indx = (i * Width) + res; 
        for (int j = 0; j < Width; j++) {
            if (*(curr_indx + j) > peak) {
                peak = *(curr_indx + j);
                max_x = j;
                max_y = i;
            }
        }
    }

    float dx = 0.0f;
    float dy = 0.0f;

    if (max_x > 0 && max_x < (Width - 1)) {
        dx = subPixelPeak(res[max_y * Width + (max_x - 1)], peak, res[max_y * Width + (max_x + 1)]);
    }

    if (max_y > 0 && max_y < (height - 1)) {
        dy = subPixelPeak(res[(max_y - 1) * Width + max_x], peak, res[(max_y + 1) * Width + max_x]);
    }

    location_x = ((float)max_x + dx - (float)(Width / 2)) * (float)cell_size;
    location_y = ((float)max_y + dy - (float)(height / 2)) * (float)cell_size;
}