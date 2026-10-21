#ifndef IMAGE_MEDIAN_HIST_H_
#define IMAGE_MEDIAN_HIST_H_

#include "image.hpp"   // also brings in assert() through riscv_port.hpp
#include <cstdint>

namespace ref
{
    // pad: input already bordered by R = K/2 on every side. out: W x H.
    template <int K>
    void median_hist_rows(const Image<uint8_t, ImageType::GRAY>& pad,
                          Image<uint8_t, ImageType::GRAY>& out)
    {
        
        constexpr int N = K * K;
        constexpr int T = N / 2 + 1;                      // the median is the T-th smallest of N

        for (int y = 0; y < out.GetHeight(); ++y)
        {
            const uint8_t* rows[K];                       // rows y .. y+K-1 of the padded image
            for (int dy = 0; dy < K; ++dy)
                rows[dy] = pad.GetPtr(0, y + dy);

            uint16_t hist[256] = {};
            for (int dy = 0; dy < K; ++dy)                // window at x = 0: columns 0 .. K-1
                for (int dx = 0; dx < K; ++dx)
                    ++hist[rows[dy][dx]];

            int med = 0, below = 0;                       // smallest med with below + hist[med] >= T
            while (below + hist[med] < T)
                below += hist[med++];

            uint8_t* dst = out.GetPtr(0, y);
            dst[0] = static_cast<uint8_t>(med);

            for (int x = 1; x < out.GetWidth(); ++x)
            {
                for (int dy = 0; dy < K; ++dy)
                {
                    const uint8_t gone = rows[dy][x - 1];         // column x-1 leaves
                    const uint8_t come = rows[dy][x + K - 1];     // column x+K-1 enters
                    --hist[gone]; below -= (gone < med);
                    ++hist[come]; below += (come < med);
                }

                if (below >= T)                                   // median moved down
                    do { below -= hist[--med]; } while (below >= T);
                else                                              // moved up, or stayed
                    while (below + hist[med] < T) below += hist[med++];

                dst[x] = static_cast<uint8_t>(med);
            }
        }
    }


}

#endif