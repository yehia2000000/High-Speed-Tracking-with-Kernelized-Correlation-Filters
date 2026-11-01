cmake_minimum_required(VERSION 3.22)

# Source files
file(GLOB_RECURSE SOURCES source/*.cpp)

# Bare-metal library (gem5): nexus-am includes + freestanding flags.
add_library(${CMAKE_PROJECT_NAME}_gem5 STATIC ${SOURCES})
set_target_properties(${CMAKE_PROJECT_NAME}_gem5 PROPERTIES
                        CXX_STANDARD 20
                        CXX_EXTENSIONS OFF)

target_include_directories(${CMAKE_PROJECT_NAME}_gem5 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/images
    ${NEXUS_AM_HOME}/am
    ${NEXUS_AM_HOME}/am/include
    ${NEXUS_AM_HOME}/libs/klib/include
    ${NEXUS_AM_HOME}/am/src/nemu/include
    ${NEXUS_AM_HOME}/am/src/xs/include)
# K1_rvv.hpp reuses ref::Roi / ref::k1_map from the scalar reference headers. Added
# after lib/include, so lib's own image_Add.hpp (vec::) still wins over reference's (ref::).
target_include_directories(${CMAKE_PROJECT_NAME}_gem5 PRIVATE
    ${CMAKE_SOURCE_DIR}/test/reference/include)

# nexus-am defines are needed to compile image.hpp's riscv_port.hpp include.
target_compile_options(${CMAKE_PROJECT_NAME}_gem5 PUBLIC ${RISCV_AM_DEFINE_FLAGS})
target_compile_options(${CMAKE_PROJECT_NAME}_gem5 PRIVATE ${RISCV_BAREMETAL_COMPILE_FLAGS})

# Hosted library (qemu-user): project headers only, no nexus-am or freestanding.
add_library(${CMAKE_PROJECT_NAME}_qemu STATIC ${SOURCES})
set_target_properties(${CMAKE_PROJECT_NAME}_qemu PROPERTIES
                        CXX_STANDARD 20
                        CXX_EXTENSIONS OFF)

target_include_directories(${CMAKE_PROJECT_NAME}_qemu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${CMAKE_PROJECT_NAME}_qemu PRIVATE   # after lib/include, as above
    ${CMAKE_SOURCE_DIR}/test/reference/include)

# Hosted glibc build: counter stale global freestanding flags in CMAKE_*_FLAGS cache.
target_compile_options(${CMAKE_PROJECT_NAME}_qemu PRIVATE
  -fno-freestanding -fbuiltin)

target_compile_definitions(${CMAKE_PROJECT_NAME}_qemu PUBLIC RISCV_QEMU)
