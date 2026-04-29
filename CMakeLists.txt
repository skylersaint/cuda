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
set(CUDA_PTX_ARCHITECTURE "" CACHE STRING "Optional SM architecture for explicit PTX generation, for example 80 or 90")

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

function(add_cuda_ptx_target target_name source_file)
  set(ptx_output_dir "${CMAKE_BINARY_DIR}/ptx")
  set(ptx_output_file "${ptx_output_dir}/${target_name}.ptx")
  set(ptx_flags
    --ptx
    --std=c++17
    --expt-relaxed-constexpr
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  if(CUDA_ENABLE_FAST_MATH)
    list(APPEND ptx_flags --use_fast_math)
  endif()

  if(CUDA_ENABLE_LINEINFO)
    list(APPEND ptx_flags -lineinfo)
  endif()

  if(CUDA_PTX_ARCHITECTURE)
    list(APPEND ptx_flags -arch=sm_${CUDA_PTX_ARCHITECTURE})
  endif()

  add_custom_command(
    OUTPUT ${ptx_output_file}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ptx_output_dir}
    COMMAND ${CMAKE_CUDA_COMPILER} ${ptx_flags} ${CMAKE_CURRENT_SOURCE_DIR}/${source_file} -o ${ptx_output_file}
    DEPENDS ${source_file}
    VERBATIM
    COMMENT "Generating PTX for ${target_name}"
  )

  add_custom_target(${target_name}_ptx DEPENDS ${ptx_output_file})
endfunction()

function(add_cuda_cubin_target target_name source_file)
  set(cubin_output_dir "${CMAKE_BINARY_DIR}/cubin")
  set(cubin_output_file "${cubin_output_dir}/${target_name}.cubin")
  set(cubin_flags
    --cubin
    --std=c++17
    --expt-relaxed-constexpr
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  if(CUDA_ENABLE_FAST_MATH)
    list(APPEND cubin_flags --use_fast_math)
  endif()

  if(CUDA_ENABLE_LINEINFO)
    list(APPEND cubin_flags -lineinfo)
  endif()

  if(CUDA_PTX_ARCHITECTURE)
    list(APPEND cubin_flags -arch=sm_${CUDA_PTX_ARCHITECTURE})
  endif()

  add_custom_command(
    OUTPUT ${cubin_output_file}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${cubin_output_dir}
    COMMAND ${CMAKE_CUDA_COMPILER} ${cubin_flags} ${CMAKE_CURRENT_SOURCE_DIR}/${source_file} -o ${cubin_output_file}
    DEPENDS ${source_file}
    VERBATIM
    COMMENT "Generating cubin for ${target_name}"
  )

  add_custom_target(${target_name}_cubin DEPENDS ${cubin_output_file})
endfunction()

add_cuda_benchmark(vector_add_bench benchmarks/vector_add_bench.cu)
add_cuda_benchmark(matmul_bench benchmarks/matmul_bench.cu)
add_cuda_ptx_target(vector_add_bench benchmarks/vector_add_bench.cu)
add_cuda_ptx_target(matmul_bench benchmarks/matmul_bench.cu)
add_cuda_cubin_target(vector_add_bench benchmarks/vector_add_bench.cu)
add_cuda_cubin_target(matmul_bench benchmarks/matmul_bench.cu)
