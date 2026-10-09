#pragma once
#include "image.hpp"
#include "types.hpp"
#include <cstdint>
namespace vec 
{
    void image_hist_rvv(const uint8_t* src, size_t n, uint32_t* hist);
    
}
