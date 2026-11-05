#pragma once

#include <cstdint>
#include <cstddef>

namespace vec {

/**
 * @brief Kernel 3: Vectorized FHOG Histogram Accumulation (Histogram_k3)
 * 
 * Takes raw gradient orientation bins and magnitudes, accumulates them
 * with bilinear spatial and orientation weights into a 3-row L1 circular buffer,
 * and streams 27 planar channels to planar_hist.
 * 
 * @param ori_bins    [IN]  Raw orientation bins (0..17), size: height * width
 * @param mag         [IN]  Gradient magnitudes, size: height * width
 * @param planar_hist [OUT] 27 planar channels, size: 27 * (height/4) * (width/4)
 * @param width       Image width (e.g. 128)
 * @param height      Image height (e.g. 128)
 */
void Histogram_k3(
    const uint8_t* ori_bins,
    const float*   mag,
    float*         planar_hist,
    int            width,
    int            height);

} // namespace vec
