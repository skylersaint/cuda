#include "cuda_utils.cuh"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    c[idx] = a[idx] + b[idx];
  }
}

bool verify(const std::vector<float>& a,
            const std::vector<float>& b,
            const std::vector<float>& c) {
  constexpr float kTolerance = 1e-5f;
  for (std::size_t i = 0; i < c.size(); ++i) {
    const float expected = a[i] + b[i];
    if (std::fabs(c[i] - expected) > kTolerance) {
      std::cerr << "Mismatch at index " << i
                << ", expected " << expected
                << ", got " << c[i] << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr int count = 1 << 20;
  constexpr std::size_t bytes = count * sizeof(float);

  std::vector<float> host_a(count);
  std::vector<float> host_b(count);
  std::vector<float> host_c(count, 0.0f);

  for (int i = 0; i < count; ++i) {
    host_a[i] = static_cast<float>(i % 1024) * 0.5f;
    host_b[i] = static_cast<float>(i % 2048) * 0.25f;
  }

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), bytes));

  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = div_up(count, kThreadsPerBlock);

  auto start = std::chrono::steady_clock::now();
  vector_add_kernel<<<blocks, kThreadsPerBlock>>>(device_a, device_b, device_c, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();

  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));

  if (!verify(host_a, host_b, host_c)) {
    return EXIT_FAILURE;
  }

  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

  std::cout << "vector_add passed, elements=" << count
            << ", blocks=" << blocks
            << ", threads_per_block=" << kThreadsPerBlock
            << ", kernel_time_ms=" << elapsed_ms.count()
            << std::endl;

  return EXIT_SUCCESS;
}
