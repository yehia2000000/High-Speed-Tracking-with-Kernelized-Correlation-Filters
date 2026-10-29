#ifndef K1_VECTOR_H_
#define K1_VECTOR_H_

#include <cstdint>
#include "image.hpp"
#include "K1_scalar.hpp"   // ref::Roi, ref::K1Map, ref::k1_map, ref::k1_coord

namespace vec
{

void vscale_scalar(const float *in, float *out, int n, float k);
void vscale_rvv(const float *in, float *out, int n, float k);

void k1_rvv_table(float base, float s, int n, int P, uint32_t *i0, uint32_t *i1, float *fr);

void k1_rvv(const Image<uint8_t, ImageType::GRAY>& in, ref::Roi r, int T, float *out,
            uint32_t *i0, uint32_t *i1, float *fx);

void k1_rvv_sep(const Image<uint8_t, ImageType::GRAY>& in, ref::Roi r, int T, float *out,
                uint32_t *i0, uint32_t *i1, float *fx, float *lineA, float *lineB);

}  // namespace vec

#endif  // K1_VECTOR_H_