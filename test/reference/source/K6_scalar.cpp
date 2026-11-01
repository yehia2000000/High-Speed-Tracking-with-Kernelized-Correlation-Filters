#include "riscv_math.hpp"   // <cmath> is not available on bare metal
#include <cfloat>
#include "K6_scalar.hpp"

#define NUM_SECTORS 9
#define NUM_BINS    27

static float subPixelPeak(float left, float center, float right)
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

    /* The response from K5 is periodic and its zero shift is at index (0, 0) (top-left
       convention, no fftshift). So the neighbours wrap around the edges, and an index past
       the middle is a negative shift. [integration fix: was centred, KCFcpp style] */
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
