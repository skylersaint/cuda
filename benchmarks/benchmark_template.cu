#include "cuda_benchmark.cuh"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr const char* kBenchmarkName = "template_kernel";
constexpr int kDefaultThreadsPerBlock = 256;
constexpr std::size_t kDefaultElements = 1 << 20;

struct Options {
  std::size_t elements = kDefaultElements;
  int threads_per_block = kDefaultThreadsPerBlock;
  BenchmarkConfig benchmark_config{};
};

__global__ void template_kernel(const float* input, float* output, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    output[idx] = input[idx];
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
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::cout << "Usage: template_kernel [--elements N] [--threads N] "
                << "[--warmup N] [--iters N]" << std::endl;
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << argv[i] << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  return options;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_args(argc, argv);
  const int count = static_cast<int>(options.elements);
  const std::size_t bytes = options.elements * sizeof(float);

  print_benchmark_header(kBenchmarkName);

  std::vector<float> host_input(options.elements, 1.0f);
  std::vector<float> host_output(options.elements, 0.0f);

  float* device_input = nullptr;
  float* device_output = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), bytes));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

  const int blocks = div_up(count, options.threads_per_block);
  const auto launch = [&]() {
    template_kernel<<<blocks, options.threads_per_block>>>(
        device_input, device_output, count);
  };

  const BenchmarkStats stats =
      benchmark_cuda_kernel(options.benchmark_config, launch);

  CUDA_CHECK(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));

  print_benchmark_stats(
      kBenchmarkName,
      options.elements,
      options.threads_per_block,
      options.benchmark_config,
      stats,
      bytes * 2);

  return EXIT_SUCCESS;
}
