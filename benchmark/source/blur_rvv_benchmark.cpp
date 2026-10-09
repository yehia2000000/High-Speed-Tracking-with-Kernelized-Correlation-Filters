#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"
#include "image1.hpp"


int main (){

    #ifdef RISCV_BAREMETAL
    // Enable the vector unit before any auto-vectorized loop runs (nexus-am's
    // _start only enables the FPU).
    riscv_enable_vector();
    #endif

    const int width = 640 ; 
    const int height = 426 ; 
    const int loop_count = 3;

    Image <uint8_t,ImageType::GRAY> input0(width, height);

    Image<uint8_t,ImageType::GRAY> output_reference(width, height);
    Image<uint8_t,ImageType::GRAY> output_vectorized(width, height); 

    bool all_correct = true ; 

    uint64_t ref_cycles =0 , vec_cycles = 0 ; 
    uint64_t ref_instr =0  , vec_instr =0 ; 

    input0.Read(width ,height, image1_data );

    Kernel<3> ker = { {1, 1, 1,
                       1, 1, 1,
                       1, 1, 1} };

    Image<uint8_t, ImageType::GRAY> Padded =input0.MakeBorder(1,BorderType::CONSTANT,0) ;


    
    for (int i=0 ; i<loop_count;i++){
        Timer timer_sc , timer_vec ; 
        timer_sc.Start(); 
        ref::Image_filter(input0, output_reference, ker, BorderType::CONSTANT, 0);
        timer_sc.Stop(); 

        timer_vec.Start();
        vec::blur3x3_rvv(Padded,output_vectorized);
        timer_vec.Stop();

        ref_cycles += timer_sc.ElapsedCycles();
        ref_instr += timer_sc.ElapsedInstructions();

        vec_cycles += timer_vec.ElapsedCycles();
        vec_instr += timer_vec.ElapsedInstructions();

        if (!CheckCorrectness(output_reference, output_vectorized))
         {
             all_correct = false;
             break;
         }
         

    }
    PrintTime("rgbTOgray", width * height,
                  vec_cycles / loop_count, vec_instr / loop_count,
                  ref_cycles / loop_count, ref_instr / loop_count);


    if (all_correct)
    {
        printf("Output is correct.\n");
        return 0;
    }
    else
    {
        printf("Output is wrong!\n");
        return 1;
    }
}