#include "image_hist.hpp"

void ref::hist256(const uint8_t* src, size_t n, uint32_t* hist) {
    for (int b = 0; b < 256; ++b) hist[b] = 0;
    for (size_t i = 0; i < n; ++i) hist[src[i]]++;
}

