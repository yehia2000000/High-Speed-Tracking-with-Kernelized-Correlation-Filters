#ifndef K2_VECTOR_H_
#define K2_VECTOR_H_

#include <cstdint>
#include "K2_scalar.hpp"   // ref::K2Method, ref::K2_COS / K2_SIN / K2_ATAN_C

namespace vec
{


void k2_rvv(const float *in, int T, float *mag, uint8_t *bin, ref::K2Method method);

}  // namespace vec

#endif  // K2_VECTOR_H_
