#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status__ = (call);                                              \
    if (status__ != cudaSuccess) {                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__)               \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

inline int div_up(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

inline int current_cuda_device() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

inline cudaDeviceProp cuda_device_properties(int device) {
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  return properties;
}
