#include <tuple>
#include "gtest_lite.hpp"
#include "output_image.hpp"
#include "riscv_cv.hpp"
#include "reference_cv.hpp"
#include "test_utils.hpp"
#include "image1.hpp"
#include "image_blur_rvv.hpp"

class ImageFilterTest : public ::testing::TestWithParam<const char *>
{
};

TEST_P(ImageFilterTest, TEST_CASE_1)
{
    //const unsigned char* buf = GetParam();

    const char * path_test  = GetParam();
    Image<uint8_t, ImageType::GRAY> Input;

    //Input.Read(640, 426, buf);
    Input.Read(path_test);

    Kernel<3> ker = { {1, 1, 1,
                       1, 1, 1,
                       1, 1, 1} };

    Image<uint8_t, ImageType::GRAY> Output_vec(Input.GetWidth(), Input.GetHeight());
    Image<uint8_t, ImageType::GRAY> Output_ref(Input.GetWidth(), Input.GetHeight());
    Image<uint8_t, ImageType::GRAY> Padded =Input.MakeBorder(1,BorderType::CONSTANT,0) ;
    vec::blur3x3_rvv(Padded,Output_vec);

    ref::Image_filter(Input, Output_ref, ker, BorderType::CONSTANT, 0);
    ExpectImagesEqual(Output_ref, Output_vec);
    Output_vec.Write("output_ff.pgm");
    
}

INSTANTIATE_TEST_SUITE_P(ImageSizesAndKernels, ImageFilterTest,
    ::testing::Values(
        "image1.pgm"         
    )
);