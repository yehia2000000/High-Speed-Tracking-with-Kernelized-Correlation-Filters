#ifndef K1_SCALAR_H_
#define K1_SCALAR_H_

#include <cstdint>
#include "image.hpp"

namespace ref
{
    typedef struct {
        float sx, sy;
        float x_base, y_base;
        int P;
    } K1Map;

    typedef struct {
        float cx, cy;
        float tw, th;
    } Roi;

    //
    void  k1_map(Roi r, int T, K1Map &m);

    void  k1_coord(float base, float s, int j, int n, int &i0, int &i1, float &fr);

    float k1_bilinear(const Image<uint8_t, ImageType::GRAY>& in, int x0, int x1, int y0, int y1, float fx, float fy);
    
    void  k1_scalar(const Image<uint8_t, ImageType::GRAY>& in, Roi r, int T, float* out);

}  // namespace ref

#endif // K1_SCALAR_H_