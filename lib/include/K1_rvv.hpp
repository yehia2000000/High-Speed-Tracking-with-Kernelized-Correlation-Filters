#ifndef K1_VECTOR_H_
#define K1_VECTOR_H_

#include <cstdint>
#include "image.hpp"
#include "K1_scalar.hpp"   // ref::Roi, ref::K1Map, ref::k1_map, ref::k1_coord

namespace vec
{
    using ref::Roi;
    using ref::K1Map;

    void k1_rvv_table(float base, float s, int n, int P, uint32_t *i0, uint32_t *i1, float *fr);

    void k1_rvv(const Image<uint8_t, ImageType::GRAY>& in, Roi r, int T, float *out,
                uint32_t *i0, uint32_t *i1, float *fx);

}  // namespace vec

#endif // K1_VECTOR_H_