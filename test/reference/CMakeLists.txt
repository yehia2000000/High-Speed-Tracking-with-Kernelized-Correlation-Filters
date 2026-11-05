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

# Which version of each kernel KCF_flow.cpp runs, as compile definitions, so one source tree
# can be built in several configurations (one build directory each). Empty: the #defines at
# the top of KCF_flow.cpp. Example: cmake -S . -B build_k1vec -DKCF_FLOW_DEFS="KCF_K1_VEC=1"
set(KCF_FLOW_DEFS "" CACHE STRING "KCF_Kn_VEC overrides for KCF_flow.cpp, e.g. KCF_K1_VEC=1;KCF_K5_VEC=0")
if(KCF_FLOW_DEFS)
  set_source_files_properties(source/KCF_flow.cpp PROPERTIES COMPILE_DEFINITIONS "${KCF_FLOW_DEFS}")
endif()
