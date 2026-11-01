cmake_minimum_required(VERSION 3.22)

# =============================================================================
# Examples. Two kinds of sources live here:
#   - library sources (KCF_TRACKER_SOURCES): the scalar KCF tracker that glues the
#     K1–K6 reference kernels together. No main(); built once as kcf_tracker_gem5 /
#     kcf_tracker_qemu and linked into the applications.
#   - applications: every other source/*.cpp has a main() and becomes an image pair
#     <name>_gem5 / <name>_qemu with run_<name>_gem5 / run_<name>_qemu targets.
# =============================================================================

set(KCF_TRACKER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/kcf_scalar.cpp)

# Profiling: cycles and instructions per kernel (example/include/kcf_prof.hpp).
option(KCF_PROFILE "Count cycles/instructions per kernel in the KCF tracker" ON)

# Where kcf_app_qemu looks for its PGM sequence when run without an argument
# (run_kcf_app_qemu passes none). Make it with scripts/kcf_sequence.py --pgm.
set(KCF_SEQ_DIR "${CMAKE_SOURCE_DIR}/images/kcf_seq_pgm" CACHE PATH
    "PGM sequence folder read by kcf_app_qemu (scripts/kcf_sequence.py --pgm)")

# ---- the tracker library -----------------------------------------------------
add_library(kcf_tracker_gem5 STATIC ${KCF_TRACKER_SOURCES})
set_target_properties(kcf_tracker_gem5 PROPERTIES CXX_STANDARD 20 CXX_EXTENSIONS OFF)
target_include_directories(kcf_tracker_gem5 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kcf_tracker_gem5 PUBLIC reference_gem5)
target_compile_options(kcf_tracker_gem5 PRIVATE ${RISCV_BAREMETAL_COMPILE_FLAGS}
                       -fno-tree-vectorize -fno-tree-slp-vectorize)

add_library(kcf_tracker_qemu STATIC ${KCF_TRACKER_SOURCES})
set_target_properties(kcf_tracker_qemu PROPERTIES CXX_STANDARD 20 CXX_EXTENSIONS OFF)
target_include_directories(kcf_tracker_qemu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kcf_tracker_qemu PUBLIC reference_qemu)
target_compile_options(kcf_tracker_qemu PRIVATE -fno-freestanding -fbuiltin
                       -fno-tree-vectorize -fno-tree-slp-vectorize)

if(KCF_PROFILE)
  target_compile_definitions(kcf_tracker_gem5 PUBLIC KCF_PROFILE)
  target_compile_definitions(kcf_tracker_qemu PUBLIC KCF_PROFILE)
endif()

# ---- the applications ----------------------------------------------------------
file(GLOB_RECURSE EXAMPLE_SOURCES source/*.cpp)
list(REMOVE_ITEM EXAMPLE_SOURCES ${KCF_TRACKER_SOURCES})

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

    riscv_add_baremetal_image(${EXAMPLE_SOURCE})
    target_include_directories(${EXAMPLE_NAME}_gem5 PRIVATE ${CMAKE_SOURCE_DIR}/images)
    target_link_libraries(${EXAMPLE_NAME}_gem5 PRIVATE
      kcf_tracker_gem5 ${CMAKE_PROJECT_NAME}_gem5 reference_gem5 riscv_baremetal_runtime)

    set_target_properties(${EXAMPLE_NAME}_gem5 PROPERTIES
      RISCV_GEM5_PASS_REGEX "Example passed\\."
      RISCV_GEM5_FAIL_REGEX "Example failed")
    riscv_add_gem5_run(${EXAMPLE_NAME}_gem5)

    riscv_add_linux_executable(${EXAMPLE_SOURCE})
    target_compile_definitions(${EXAMPLE_NAME}_qemu PRIVATE KCF_SEQ_DIR="${KCF_SEQ_DIR}")
    target_link_libraries(${EXAMPLE_NAME}_qemu PRIVATE
      kcf_tracker_qemu ${CMAKE_PROJECT_NAME}_qemu reference_qemu)

    riscv_add_qemu_run(${EXAMPLE_NAME}_qemu)
endforeach()
