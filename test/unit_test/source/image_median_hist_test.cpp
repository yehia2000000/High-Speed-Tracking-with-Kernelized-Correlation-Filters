#include "gtest_lite.hpp"

#include <tuple>

#include "riscv_cv.hpp"
#include "reference_cv.hpp"
#include "test_utils.hpp"

// Both implementations take the kernel size as a template parameter,
// so the runtime test parameter picks the instantiation here.

constexpr uint8_t kConstVal = 128;
 
class MedianTestHist : public ::testing::TestWithParam<std::tuple<int, int, int, BorderType>>
{
};

TEST_P(MedianTestHist, VectorizedMatchesReference)
{
    const auto [width, height, ksize, border] = GetParam();

    const int pad = ksize / 2;

    // print the test parameters
    printf("[ %-7s %16d ]\n", "width:", width);
    printf("[ %-7s %16d ]\n", "height:", height);
    printf("[ %-7s %16d ]\n", "kernel:", ksize);
    printf("[ %-7s %16d ]\n", "border:", static_cast<int>(border));

    Image<uint8_t> input0(width, height);

    Image<uint8_t> output_reference(width, height);
    Image<uint8_t> output_vectorized(width, height);

    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input0);

    Image<uint8_t, ImageType::GRAY> padded(width + 2 * pad, height + 2 * pad);
    input0.MakeBorder(padded, pad, border, kConstVal);

    switch (ksize)
    {
        case 9:
            ref::median_hist_rows<9>(padded, output_reference);
            vec::median_hist_rvv_rows<9>(padded, output_vectorized);
            break;
        case 11:
            ref::median_hist_rows<11>(padded, output_reference);
            vec::median_hist_rvv_rows<11>(padded, output_vectorized);
            break;
        default:
            assert(false && "kernel size must be 9 or 11");
    }

    ExpectImagesEqual(output_reference, output_vectorized);
}

// gtest_lite keeps at most 128 instantiations per fixture and silently drops
// the rest (it only prints "gtest_lite: too many instantiations"). Its name
// buffer is also 128 entries for the whole binary, with no bounds check.
// The two suites below register 54 + 16 = 70 tests, well under that limit.
//
// Both kernels read the same padded image, so the border type only changes
// the input values near the edges, not the kernel's code path. The width
// sweep therefore uses one border; a small second suite covers all four.
//
// VLEN = 128: VL = 16 at SEW = 8 and VL = 8 at SEW = 16.
// Padded width W + K - 1 is exactly 16 at W = 8 (K = 9) and W = 6 (K = 11).

// REPLICATE only clamps indices, so it stays well-defined even when the
// image is smaller than the pad (width 1 with pad 5).
INSTANTIATE_TEST_SUITE_P(ImageSizesAndKernels, MedianTestHist,
    ::testing::Combine(
        ::testing::Values(          // width
            1,                      // tail only: width < VL
            6,                      // padded width = 16 for K = 11
            8,                      // VL at SEW = 16; padded width = 16 for K = 9
            9,                      // VL (SEW = 16) + tail
            16,                     // VL at SEW = 8
            17,                     // VL (SEW = 8) + tail
            32,                     // multiple exact vectors
            33, 47),                // multiple vectors + tail
        ::testing::Values(          // height
            1,                      // single row
            13,                     // odd height
            22),                    // even height
        ::testing::Values(9, 11),   // kernel size
        ::testing::Values(BorderType::REPLICATE))
);

