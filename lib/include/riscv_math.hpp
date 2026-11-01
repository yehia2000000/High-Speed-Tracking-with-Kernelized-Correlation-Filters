#pragma once

#ifdef RISCV_BAREMETAL

#include <stddef.h>   // size_t: kernels got it through <cmath> on the hosted side

extern "C" {
float  sqrtf(float);
double sqrt(double);
float  fmaf(float, float, float);
double fma(double, double, double);
float  fabsf(float);
double fabs(double);
float  fminf(float, float);
float  fmaxf(float, float);
float  floorf(float);
float  rintf(float);
double rint(double);
float  expf(float);
float  cosf(float);
float  sinf(float);
double cos(double);
double sin(double);
float  atanf(float);
float  atan2f(float, float);
}

// The std:: overloads the kernels call (std::sqrt in K4, std::fma in K1), which <cmath>
// would normally provide.
namespace std {
inline float  sqrt(float x) { return ::sqrtf(x); }
inline double sqrt(double x) { return ::sqrt(x); }
inline float  fma(float a, float b, float c) { return ::fmaf(a, b, c); }
inline double fma(double a, double b, double c) { return ::fma(a, b, c); }
inline float  fabs(float x) { return ::fabsf(x); }
}

#else

#include <cmath>

#endif
