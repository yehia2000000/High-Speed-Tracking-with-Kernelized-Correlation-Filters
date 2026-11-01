cmake_minimum_required(VERSION 3.22)

# Every source/*.cpp with a main() becomes <name>_gem5 + <name>_qemu with run_* targets.
#
# Examples listed in EXAMPLE_QEMU_ONLY read files through the hosted libc, which bare metal
# does not have, so they are built for qemu-user only:
#   kcf_tracker   the KCF tracker on a whole PGM sequence (boxes, accuracy).
#                 Its gem5 counterpart is benchmark/source/KCF_benchmark.cpp (cycles).
set(EXAMPLE_QEMU_ONLY kcf_tracker)

# Where kcf_tracker_qemu looks for its PGM sequence when run without an argument
# (run_kcf_tracker_qemu passes none). Make it with scripts/kcf_sequence.py --pgm.
set(KCF_SEQ_DIR "${CMAKE_SOURCE_DIR}/images/kcf_seq_pgm" CACHE PATH
    "PGM sequence folder read by kcf_tracker_qemu (scripts/kcf_sequence.py --pgm)")

file(GLOB_RECURSE EXAMPLE_SOURCES source/*.cpp)

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

    if(NOT EXAMPLE_NAME IN_LIST EXAMPLE_QEMU_ONLY)
        riscv_add_baremetal_image(${EXAMPLE_SOURCE})
        target_include_directories(${EXAMPLE_NAME}_gem5 PRIVATE ${CMAKE_SOURCE_DIR}/images)
        target_link_libraries(${EXAMPLE_NAME}_gem5 PRIVATE ${CMAKE_PROJECT_NAME}_gem5 reference_gem5 riscv_baremetal_runtime)

        set_target_properties(${EXAMPLE_NAME}_gem5 PROPERTIES
          RISCV_GEM5_PASS_REGEX "Example passed\\."
          RISCV_GEM5_FAIL_REGEX "Example failed")
        riscv_add_gem5_run(${EXAMPLE_NAME}_gem5)
    endif()

    riscv_add_linux_executable(${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME}_qemu PRIVATE ${CMAKE_PROJECT_NAME}_qemu reference_qemu)
    target_compile_definitions(${EXAMPLE_NAME}_qemu PRIVATE KCF_SEQ_DIR="${KCF_SEQ_DIR}")

    riscv_add_qemu_run(${EXAMPLE_NAME}_qemu)
endforeach()
