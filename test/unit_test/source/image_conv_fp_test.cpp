#include <tuple>
#include "riscv_cv.hpp" 
#include "reference_cv.hpp"
#include "test_utils.hpp" 

using Image_arr = unsigned char*;  

class ImageConvTest : public ::testing::TestWithParam< const char *>
{
    
};
TEST_P(ImageConvTest, TEST_CASE_1)
{  
     const char * path_test  = GetParam();

    
    Image <uint8_t, ImageType::RGB> Input; 
    Input.Read(path_test);
    Image <uint8_t, ImageType::GRAY> Output (Input.GetWidth(),Input.GetHeight())  ;

    ref::Image_conv_fp(Input, Output);
    //Output.Write("output_gray.ppm");
}

INSTANTIATE_TEST_SUITE_P(ImageSizesAndKernels, ImageConvTest,
    ::testing::Values(
        "sample_640×426.ppm"
    )
);