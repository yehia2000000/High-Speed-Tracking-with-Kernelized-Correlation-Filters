#pragma once

#include <cstdint>
#include <cstddef>

namespace vec {

template <int k = 4>
void Histogram_k3_single_template(
    const uint32_t* ori_bins,
    const float*    mag,
    float*          planar_hist,
    int             width,
    int             height);

template <int k = 4>
void Histogram_k3(
    const uint32_t* ori_bins,
    const float*    mag,
    float*          planar_hist,
    int             width,
    int             height);

} // namespace vec
