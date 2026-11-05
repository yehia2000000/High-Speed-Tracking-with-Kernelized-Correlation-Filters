#pragma once

#include <cstdint>
#include <cstddef>

namespace vec {

float subPixelPeak(float left, float center, float right);

void k6_detection(
    const float* res,
    const int    Width,
    const int    height,
    const int    cell_size,
    float&       location_x,
    float&       location_y);

} // namespace vec
