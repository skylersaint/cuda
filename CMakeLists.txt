cmake_minimum_required(VERSION 3.24)

project(cuda_baseline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

add_library(cuda_project_options INTERFACE)
target_compile_options(cuda_project_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(vector_add
  src/vector_add.cu
)

target_include_directories(vector_add PRIVATE include)
target_link_libraries(vector_add PRIVATE cuda_project_options)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

