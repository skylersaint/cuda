#pragma once

#include "cuda_utils.cuh"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct BenchmarkConfig {
  int warmup_iterations = 10;
  int measured_iterations = 100;
};

struct BenchmarkStats {
  float min_ms = 0.0f;
  float max_ms = 0.0f;
  float mean_ms = 0.0f;
  float median_ms = 0.0f;
};

inline BenchmarkStats compute_stats(std::vector<float> samples_ms) {
  BenchmarkStats stats;
  if (samples_ms.empty()) {
    return stats;
  }

  std::sort(samples_ms.begin(), samples_ms.end());
  stats.min_ms = samples_ms.front();
  stats.max_ms = samples_ms.back();

  const float sum =
      std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0f);
  stats.mean_ms = sum / static_cast<float>(samples_ms.size());

  const std::size_t middle = samples_ms.size() / 2;
  if (samples_ms.size() % 2 == 0) {
    stats.median_ms = (samples_ms[middle - 1] + samples_ms[middle]) * 0.5f;
  } else {
    stats.median_ms = samples_ms[middle];
  }

  return stats;
}

template <typename LaunchFn>
BenchmarkStats benchmark_cuda_kernel(const BenchmarkConfig& config,
                                     LaunchFn&& launch_fn) {
  std::vector<float> samples_ms;
  samples_ms.reserve(static_cast<std::size_t>(config.measured_iterations));

  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  for (int i = 0; i < config.warmup_iterations; ++i) {
    launch_fn();
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < config.measured_iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(start_event));
    launch_fn();
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    samples_ms.push_back(elapsed_ms);
  }

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));

  return compute_stats(std::move(samples_ms));
}

inline double bandwidth_gb_per_s(std::size_t bytes, float milliseconds) {
  if (milliseconds <= 0.0f) {
    return 0.0;
  }
  const double seconds = static_cast<double>(milliseconds) / 1.0e3;
  return static_cast<double>(bytes) / seconds / 1.0e9;
}

inline void print_benchmark_header(const std::string& benchmark_name) {
  const int device = current_cuda_device();
  const auto properties = cuda_device_properties(device);

  std::cout << "benchmark=" << benchmark_name
            << " device=\"" << properties.name << "\""
            << " sm_count=" << properties.multiProcessorCount
            << " compute_capability=" << properties.major << "." << properties.minor
            << std::endl;
}

inline void print_benchmark_stats(const std::string& benchmark_name,
                                  std::size_t elements,
                                  int threads_per_block,
                                  const BenchmarkConfig& config,
                                  const BenchmarkStats& stats,
                                  std::size_t bytes_per_iteration) {
  std::cout << std::fixed << std::setprecision(4)
            << "result benchmark=" << benchmark_name
            << " elements=" << elements
            << " threads=" << threads_per_block
            << " warmup=" << config.warmup_iterations
            << " iters=" << config.measured_iterations
            << " min_ms=" << stats.min_ms
            << " median_ms=" << stats.median_ms
            << " mean_ms=" << stats.mean_ms
            << " max_ms=" << stats.max_ms
            << " bandwidth_gb_s="
            << bandwidth_gb_per_s(bytes_per_iteration, stats.mean_ms)
            << std::endl;
}

