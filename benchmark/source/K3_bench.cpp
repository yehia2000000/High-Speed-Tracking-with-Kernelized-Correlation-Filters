#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"

int main()
{
#ifdef RISCV_BAREMETAL
    // Enable the vector unit before any auto-vectorized loop runs (nexus-am's
    // _start only enables the FPU).
    riscv_enable_vector();
#endif

    const int width = 128;
    const int height = 100;
    const int loop_count = 3;

    printf("Add benchmark: %dx%d, %d iterations\n", width, height, loop_count);

    Image<uint8_t> input0(width, height);
    Image<uint8_t> input1(width, height);

    Image<uint8_t> output_reference(width, height);
    Image<uint8_t> output_vectorized(width, height);

    // Deterministic fill so runs are reproducible.
    RandomInt<uint8_t> random(0, 255);
    random.ImageRandomInitialize(input0);
    random.ImageRandomInitialize(input1);

    bool all_correct = true;

    // Exercise both overflow policies.
    const OverFlowPolicy policies[] = {OverFlowPolicy::SATURATE, OverFlowPolicy::WRAP};
    const char* policy_names[] = {"Add SATURATE", "Add WRAP"};

    for (int p = 0; p < 2; ++p)
    {
        const OverFlowPolicy overFlowPolicy = policies[p];

        uint64_t reference_cycles = 0, vectorized_cycles = 0;
        uint64_t reference_instrs = 0, vectorized_instrs = 0;

        for (int i = 0; i < loop_count; ++i)
        {
            Timer timer_scalar, timer_vectorized;

            timer_scalar.Start();
            ref::Add(input0, input1, output_reference, overFlowPolicy);
            timer_scalar.Stop();

            timer_vectorized.Start();
            vec::Add(input0, input1, output_vectorized, overFlowPolicy);
            timer_vectorized.Stop();

            reference_cycles += timer_scalar.ElapsedCycles();
            reference_instrs += timer_scalar.ElapsedInstructions();

            vectorized_cycles += timer_vectorized.ElapsedCycles();
            vectorized_instrs += timer_vectorized.ElapsedInstructions();

            if (!CheckCorrectness(output_reference, output_vectorized))
            {
                all_correct = false;
                break;
            }
        }

        PrintTime(policy_names[p], width * height,
                  vectorized_cycles / loop_count, vectorized_instrs / loop_count,
                  reference_cycles / loop_count, reference_instrs / loop_count);

        if (!all_correct)
        {
            break;
        }
    }

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
