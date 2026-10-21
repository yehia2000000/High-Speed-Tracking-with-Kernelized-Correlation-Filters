#include "gtest_lite.hpp"

#include <tuple>

#include "riscv_cv.hpp"
#include "reference_cv.hpp"
#include "test_utils.hpp"

// Both implementations take the kernel size as a template parameter,
// so the runtime test parameter picks the instantiation here.
constexpr uint8_t kConstVal = 128;

class MedianTest : public ::testing::TestWithParam<std::tuple<int, int, int, BorderType>>
{
};
 
TEST_P(MedianTest, VectorizedMatchesReference)
{

    const auto [width, height, ksize, border] = GetParam();

    const int pad = ksize / 2;
    // print the test parameters
    printf("[ %-7s %16d ]\n", "width:", width);
    printf("[ %-7s %16d ]\n", "height:", height);
    printf("[ %-7s %16d ]\n", "kernel:", ksize);

    Image<uint8_t> input0(width, height);

    Image<uint8_t> output_reference(width, height);
    Image<uint8_t> output_vectorized(width, height);

    Image<uint8_t, ImageType::GRAY> padded(width + 2 * pad, height + 2 * pad);
    input0.MakeBorder(padded, pad, border, kConstVal);

    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input0);

    switch (ksize)
    {
        case 3: vec::median_rvv_rows<3>(padded, output_vectorized); 
        ref::median_scalar_rows<3>(padded, output_reference, ref::MedSort<3>());
        break;
        case 5: vec::median_rvv_rows<5>(padded, output_vectorized); 
        ref::median_scalar_rows<5>(padded, output_reference, ref::MedSort<5>());
        break;
        case 7: vec::median_rvv_rows<7>(padded, output_vectorized); 
        ref::median_scalar_rows<7>(padded, output_reference, ref::MedSort<7>());
        break;
        default: assert(false && "kernel size must be 3, 5 or 7");
    }

    ExpectImagesEqual(output_reference, output_vectorized);
}

// VL = VLEN/SEW = 128/8 = 16 elements per vector.
// Stage 2 (one output vector at a time) loops over the width W.
// Stage 1 (column sort) loops over the padded width W + K - 1, so its tail depends on K:
// W = 14 / 12 / 10 gives a padded width of exactly 16 for K = 3 / 5 / 7.
INSTANTIATE_TEST_SUITE_P(ImageSizesAndKernels, MedianTest,
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
        ::testing::Values(3, 5, 7),   // kernel size
        ::testing::Values(BorderType::REPLICATE))
);