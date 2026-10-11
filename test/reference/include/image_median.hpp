#ifndef IMAGE_MEDIAN_H_
#define IMAGE_MEDIAN_H_

#include "image.hpp"   // also brings in assert() through riscv_port.hpp
#include <cstdint>

namespace ref {
struct MedSort {
    uint8_t operator()(uint8_t* w) const {
        for (int i = 1; i < 49; ++i) {
            uint8_t key = w[i];
            int j = i - 1;
            while (j >= 0 && w[j] > key) {
                w[j + 1] = w[j];   
                --j;
            }
            w[j + 1] = key;
        }
        return w[24];            
    }
};
}



namespace ref
{
    constexpr int K = 7;        // kernel size
    constexpr int R = K / 2;    // border needed on each side = 3

    template <typename MED>
    void median_scalar_rows(const Image<uint8_t, ImageType::GRAY>& pad,
                            Image<uint8_t, ImageType::GRAY>& out, MED med)
    {
        for (int y = 0; y < out.GetHeight(); y++)
        {
            const uint8_t* r0 = pad.GetPtr(0, y);
            const uint8_t* r1 = pad.GetPtr(0, y + 1);
            const uint8_t* r2 = pad.GetPtr(0, y + 2);
            const uint8_t* r3 = pad.GetPtr(0, y + 3);
            const uint8_t* r4 = pad.GetPtr(0, y + 4);
            const uint8_t* r5 = pad.GetPtr(0, y + 5);
            const uint8_t* r6 = pad.GetPtr(0, y + 6);

            uint8_t* dst = out.GetPtr(0, y);    // GetPtr already returns the pointer, no '&'

            for (int x = 0; x < out.GetWidth(); x++)
            {
                uint8_t window[K * K] = {
                    r0[x], r0[x + 1], r0[x + 2], r0[x + 3], r0[x + 4], r0[x + 5], r0[x + 6],
                    r1[x], r1[x + 1], r1[x + 2], r1[x + 3], r1[x + 4], r1[x + 5], r1[x + 6],
                    r2[x], r2[x + 1], r2[x + 2], r2[x + 3], r2[x + 4], r2[x + 5], r2[x + 6],
                    r3[x], r3[x + 1], r3[x + 2], r3[x + 3], r3[x + 4], r3[x + 5], r3[x + 6],
                    r4[x], r4[x + 1], r4[x + 2], r4[x + 3], r4[x + 4], r4[x + 5], r4[x + 6],
                    r5[x], r5[x + 1], r5[x + 2], r5[x + 3], r5[x + 4], r5[x + 5], r5[x + 6],
                    r6[x], r6[x + 1], r6[x + 2], r6[x + 3], r6[x + 4], r6[x + 5], r6[x + 6]};

                dst[x] = med(window);
            }
        }
    }

    // med comes before type: a parameter with a default must come after all parameters without one.
    template <typename MED =  MedSort>
    Image<uint8_t, ImageType::GRAY> median_filter(const Image<uint8_t, ImageType::GRAY>& input,
                                                  const BorderType type = BorderType::REPLICATE , MED med = MedSort())
    {
        Image<uint8_t, ImageType::GRAY> Padded = input.MakeBorder(R, type, 0);

        assert(Padded.Width() == input.Width() + 2 * R);
        assert(Padded.Height() == input.Height() + 2 * R);

        Image<uint8_t, ImageType::GRAY> out(input.Width(), input.Height());
        median_scalar_rows(Padded, out, med);
        return out;
    }
}

#endif