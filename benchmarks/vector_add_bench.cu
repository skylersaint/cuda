#include "cuda_benchmark.cuh"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char* kBenchmarkName = "vector_add";
constexpr int kDefaultThreadsPerBlock = 256;
constexpr std::size_t kDefaultElements = 1 << 24;

struct Options {
  std::size_t elements = kDefaultElements;
  int threads_per_block = kDefaultThreadsPerBlock;
  BenchmarkConfig benchmark_config{};
  bool check_results = true;
};

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    c[idx] = a[idx] + b[idx];
  }
}

Options parse_args(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--elements") == 0 && i + 1 < argc) {
      options.elements = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      options.threads_per_block = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      options.benchmark_config.warmup_iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      options.benchmark_config.measured_iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--no-check") == 0) {
      options.check_results = false;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::cout << "Usage: vector_add_bench [--elements N] [--threads N] "
                << "[--warmup N] [--iters N] [--no-check]" << std::endl;
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << argv[i] << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (options.elements == 0) {
    std::cerr << "--elements must be greater than 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (options.threads_per_block <= 0) {
    std::cerr << "--threads must be greater than 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (options.benchmark_config.warmup_iterations < 0 ||
      options.benchmark_config.measured_iterations <= 0) {
    std::cerr << "warmup must be >= 0 and iters must be > 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return options;
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

int main(int argc, char** argv) {
  const Options options = parse_args(argc, argv);
  const int count = static_cast<int>(options.elements);
  const std::size_t bytes = options.elements * sizeof(float);

  print_benchmark_header(kBenchmarkName);

  std::vector<float> host_a(options.elements);
  std::vector<float> host_b(options.elements);
  std::vector<float> host_c(options.elements, 0.0f);

  for (std::size_t i = 0; i < options.elements; ++i) {
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

  const int blocks = div_up(count, options.threads_per_block);
  const auto launch = [&]() {
    vector_add_kernel<<<blocks, options.threads_per_block>>>(
        device_a, device_b, device_c, count);
  };

  const BenchmarkStats stats =
      benchmark_cuda_kernel(options.benchmark_config, launch);

  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));

  if (options.check_results && !verify(host_a, host_b, host_c)) {
    return EXIT_FAILURE;
  }

  print_benchmark_stats(
      kBenchmarkName,
      options.elements,
      options.threads_per_block,
      options.benchmark_config,
      stats,
      bytes * 3);

  return EXIT_SUCCESS;
}

