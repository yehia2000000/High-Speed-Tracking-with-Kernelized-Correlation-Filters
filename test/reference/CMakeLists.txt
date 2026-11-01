cmake_minimum_required(VERSION 3.22)
project(reference)

# Source files
file(GLOB_RECURSE SOURCES source/*.cpp)

# Bare-metal reference library (gem5).
add_library(${PROJECT_NAME}_gem5 STATIC ${SOURCES})
set_target_properties(${PROJECT_NAME}_gem5 PROPERTIES
                        CXX_STANDARD 20
                        CXX_EXTENSIONS OFF)

target_link_libraries(${PROJECT_NAME}_gem5 PUBLIC ${CMAKE_PROJECT_NAME}_gem5)
target_compile_options(${PROJECT_NAME}_gem5 PRIVATE ${RISCV_BAREMETAL_COMPILE_FLAGS})
target_include_directories(${PROJECT_NAME}_gem5 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Hosted reference library (qemu-user).
add_library(${PROJECT_NAME}_qemu STATIC ${SOURCES})
set_target_properties(${PROJECT_NAME}_qemu PROPERTIES
                        CXX_STANDARD 20
                        CXX_EXTENSIONS OFF)

target_link_libraries(${PROJECT_NAME}_qemu PUBLIC ${CMAKE_PROJECT_NAME}_qemu)
target_include_directories(${PROJECT_NAME}_qemu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME}_qemu PRIVATE ${CMAKE_SOURCE_DIR}/images)

# The ref:: kernels are the scalar baseline. With the RVV -march and -O3, GCC auto-vectorizes
# some of their loops, so their gem5 cycle counts would already contain vector code (and
# their float sums change order). Turn it off unless asked.
option(REFERENCE_SCALAR_ONLY "Compile the ref:: kernels without GCC auto-vectorization" ON)
if(REFERENCE_SCALAR_ONLY)
  target_compile_options(${PROJECT_NAME}_gem5 PRIVATE -fno-tree-vectorize -fno-tree-slp-vectorize)
  target_compile_options(${PROJECT_NAME}_qemu PRIVATE -fno-tree-vectorize -fno-tree-slp-vectorize)
endif()
