
#ifndef IN_MG_HIST_H_ 
#define IN_MG_HIST_H_

#pragma once
#include "image.hpp"
#include "types.hpp"
#include <cstdint>


namespace ref {
void hist256(const uint8_t* src, size_t n, uint32_t* hist) ;
}




#endif