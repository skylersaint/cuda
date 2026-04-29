cmake_minimum_required(VERSION 3.24)

project(cuda_benchmarks LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

option(CUDA_ENABLE_FAST_MATH "Enable --use_fast_math for CUDA targets" OFF)
option(CUDA_ENABLE_LINEINFO "Enable -lineinfo for Nsight Compute correlation" ON)

add_library(cuda_project_options INTERFACE)
target_compile_options(cuda_project_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(CUDA_ENABLE_FAST_MATH)
  target_compile_options(cuda_project_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()

if(CUDA_ENABLE_LINEINFO)
  target_compile_options(cuda_project_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
endif()

function(add_cuda_benchmark target_name source_file)
  add_executable(${target_name} ${source_file})
  target_include_directories(${target_name} PRIVATE include)
  target_link_libraries(${target_name} PRIVATE cuda_project_options)
  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endfunction()

add_cuda_benchmark(vector_add_bench benchmarks/vector_add_bench.cu)
add_cuda_benchmark(matmul_bench benchmarks/matmul_bench.cu)
