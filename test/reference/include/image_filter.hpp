#ifndef IMAGE_FILTER_H_
#define IMAGE_FILTER_H_

#include "image.hpp"
#include <cstdint>

namespace ref
{

    template <int K>
    bool Blur(const Image<uint8_t, ImageType::GRAY>& Input,
              Image<uint8_t, ImageType::GRAY>& Output,
              const BorderType border_mode,
              const uint8_t value = 0)
    {
        constexpr int R    = K / 2;
        constexpr int Area = K * K;

        if (Input.Width() <= 0 || Input.Height() <= 0) return false;

        const int off  = (border_mode == BorderType::NO_BORDER) ? R : 0;
        const int outW = Input.Width()  - 2 * off;
        const int outH = Input.Height() - 2 * off;
        if (outW <= 0 || outH <= 0) return false;
        if (Output.Width() != outW || Output.Height() != outH) return false;

        for (int y = 0; y < outH; ++y)
            for (int x = 0; x < outW; ++x)
            {
                const int cx = x + off;
                const int cy = y + off;
                int32_t sum = 0;
                for (int ky = -R; ky <= R; ++ky)
                    for (int kx = -R; kx <= R; ++kx)
                        sum += Input.GetPixelBorder(cx + kx, cy + ky, 0, border_mode, value);
                Output.SetPixel(x, y, static_cast<uint8_t>(sum / Area));
            }
        return true;
    }

    inline bool Blur(const Image<uint8_t, ImageType::GRAY>& Input,
                     Image<uint8_t, ImageType::GRAY>& Output, int k,
                     const BorderType border_mode,
                     const uint8_t value = 0)
    {
        switch (k)
        {
            case 3:  return Blur<3>(Input, Output, border_mode, value);
            case 5:  return Blur<5>(Input, Output, border_mode, value);
            case 7:  return Blur<7>(Input, Output, border_mode, value);
            case 9:  return Blur<9>(Input, Output, border_mode, value);
            case 11: return Blur<11>(Input, Output, border_mode, value);
            default: return false;
        }
    }
}

#endif