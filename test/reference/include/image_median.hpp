#ifndef IMAGE_MEDIAN_H_
#define IMAGE_MEDIAN_H_

#include "image.hpp"   // also brings in assert() through riscv_port.hpp
#include <cstdint>

// Scalar reference K x K median filter. K = 3, 5, 7 match vec::median_filter<K>.
// Usage: ref::median_filter<3>(img), ref::median_filter<5>(img), ref::median_filter(img) == <7>

namespace ref
{
    // Insertion sort of the K*K window, returns the middle element.
    template <int K>
    struct MedSort {
        static constexpr int N = K * K;
        uint8_t operator()(uint8_t* w) const {
            for (int i = 1; i < N; ++i) {
                uint8_t key = w[i];
                int j = i - 1;
                while (j >= 0 && w[j] > key) {
                    w[j + 1] = w[j];
                    --j;
                }
                w[j + 1] = key;
            }
            return w[N / 2];
        }
    };
 
    // pad: input already bordered by R = K/2 on every side. out: W x H.
    template <int K = 7, typename MED = MedSort<K>>
    void median_scalar_rows(const Image<uint8_t, ImageType::GRAY>& pad,
                            Image<uint8_t, ImageType::GRAY>& out, MED med)
    {
        static_assert(K >= 3 && K % 2 == 1, "K must be odd and >= 3");
        for (int y = 0; y < out.GetHeight(); y++)
        {
            const uint8_t* rows[K];                          // rows y .. y+K-1 of the padded image
            for (int dy = 0; dy < K; ++dy)
                rows[dy] = pad.GetPtr(0, y + dy);

            uint8_t* dst = out.GetPtr(0, y);    // GetPtr already returns the pointer, no '&'

            for (int x = 0; x < out.GetWidth(); x++)
            {
                uint8_t window[K * K];                       // row-major, same order as before
                for (int dy = 0; dy < K; ++dy)
                    for (int dx = 0; dx < K; ++dx)
                        window[dy * K + dx] = rows[dy][x + dx];

                dst[x] = med(window);
            }
        }
    }

}

#endif