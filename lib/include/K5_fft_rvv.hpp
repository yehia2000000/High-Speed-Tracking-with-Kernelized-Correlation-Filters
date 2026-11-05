#ifndef K5_FFT_RVV_H_
#define K5_FFT_RVV_H_

namespace vec
{

void fft2(float *re, float *im, int rows, int cols, int inverse);

}  // namespace vec

#endif  // K5_FFT_RVV_H_
