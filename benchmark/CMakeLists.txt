cmake_minimum_required(VERSION 3.22)

# Benchmarks build as bare-metal images for gem5 and static Linux ELFs for
# qemu-user. Each .cpp under source/ becomes one pair (e.g. Add_benchmark ->
# Add_benchmark_gem5 + Add_benchmark_qemu) with matching run_* targets.
#
# Multi-core benchmarks: set BENCHMARK_NCPU_<name> to the number of gem5 cores
# the image expects (passed to gem5 as -n). These use nexus-am's multi-hart API
# (_mpe_wakeup, _mpe_init), which has no qemu-user equivalent, so they are
# built for gem5 only. Benchmarks not listed here run on 1 core.

set(BENCHMARK_NCPU_image_hist_bench 2)

file(GLOB_RECURSE BENCHMARK_SOURCES source/*.cpp)

foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)

    if(DEFINED BENCHMARK_NCPU_${BENCHMARK_NAME})
        set(BENCHMARK_NCPU ${BENCHMARK_NCPU_${BENCHMARK_NAME}})
    else()
        set(BENCHMARK_NCPU 1)
    endif()

    riscv_add_baremetal_image(${BENCHMARK_SOURCE})
    target_link_libraries(${BENCHMARK_NAME}_gem5 PRIVATE ${CMAKE_PROJECT_NAME}_gem5 reference_gem5 riscv_baremetal_runtime)

    target_include_directories(${BENCHMARK_NAME}_gem5 PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Benchmarks are not gtest programs; they print "Output is correct." on success and "Output is wrong!" on failure.
    set_target_properties(${BENCHMARK_NAME}_gem5 PROPERTIES
      RISCV_GEM5_PASS_REGEX "Output is correct\\."
      RISCV_GEM5_FAIL_REGEX "Output is wrong")
    riscv_add_gem5_run(${BENCHMARK_NAME}_gem5 NCPU ${BENCHMARK_NCPU})

    # Multi-hart benchmarks are gem5-only (see the note at the top).
    if(BENCHMARK_NCPU GREATER 1)
        continue()
    endif()

    riscv_add_linux_executable(${BENCHMARK_SOURCE})
    target_link_libraries(${BENCHMARK_NAME}_qemu PRIVATE ${CMAKE_PROJECT_NAME}_qemu reference_qemu)

    target_include_directories(${BENCHMARK_NAME}_qemu PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)

    riscv_add_qemu_run(${BENCHMARK_NAME}_qemu)
endforeach()