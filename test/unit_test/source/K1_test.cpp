#include "gtest_lite.hpp"

#include <tuple>

#include "riscv_cv.hpp"
#include "reference_cv.hpp"
#include "test_utils.hpp"

class AddTest : public ::testing::TestWithParam<std::tuple<int, int, OverFlowPolicy>>
{
};

TEST_P(AddTest, VectorizedMatchesReference)
{
    const auto [width, height, overFlowPolicy] = GetParam();

    // print the test parameters
    printf("[ %-7s %16d ]\n", "width:", width);
    printf("[ %-7s %16d ]\n", "height:", height);
    printf("[ %-7s %16s ]\n", "policy:",
           overFlowPolicy == OverFlowPolicy::SATURATE ? "SATURATE" : "WRAP");

    Image<uint8_t> input0(width, height);   
    Image<uint8_t> input1(width, height);

    Image<uint8_t> output_reference(width, height);
    Image<uint8_t> output_vectorized(width, height);

    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input0);
    random.ImageRandomInitialize(input1);

    ref::Add(input0, input1, output_reference, overFlowPolicy);
    vec::Add(input0, input1, output_vectorized, overFlowPolicy);

    ExpectImagesEqual(output_reference, output_vectorized);
}

// VL = VLEN/SEW = 128/8 = 16 elements per vector; loop is over width.
// "tail" = remainder elements after full vector chunks (width % VL != 0).
INSTANTIATE_TEST_SUITE_P(ImageSizesAndKernels, AddTest,
    ::testing::Combine(
        ::testing::Values(
            1, 7,       // tail: width < VL
            16,         // exact one vector
            17, 24, 31, // one vector + tail
            32, 48,     // multiple exact vectors
            33, 47),    // multiple vectors + tail
        ::testing::Values(
            1,          // single row
            13,          // odd height
            22),         // even height
        ::testing::Values(OverFlowPolicy::SATURATE, OverFlowPolicy::WRAP))
    );
