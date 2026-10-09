#ifndef IN_MG_CONV_RVV_H_ 
#define IN_MG_CONV_RVV_H_

#pragma once
#include "image.hpp"
#include "types.hpp"
#include <cstdint>
constexpr uint8_t WRRV = (uint8_t)(0.299f * 256) ; 
constexpr uint8_t WGRV = (uint8_t) (0.587f * 256);
constexpr uint8_t WBRV =  (uint8_t)(0.114f *256) ;

namespace vec {
void image_conv_rvv(Image <uint8_t,ImageType::RGB> & Input , Image <uint8_t, ImageType::GRAY> & Output) ;
}

#endif