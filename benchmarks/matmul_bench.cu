#include "cuda_benchmark.cuh"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr const char* kBenchmarkName = "matmul";
constexpr int kDefaultTileSize = 16;
constexpr int kDefaultM = 1024;
constexpr int kDefaultN = 1024;
constexpr int kDefaultK = 1024;

struct Options {
  int m = kDefaultM;
  int n = kDefaultN;
  int k = kDefaultK;
  int tile_size = kDefaultTileSize;
  BenchmarkConfig benchmark_config{};
  bool check_results = true;
};

__global__ void matmul_tiled_kernel(const float* a,
                                    const float* b,
                                    float* c,
                                    int m,
                                    int n,
                                    int k) {
  extern __shared__ float shared_mem[];
  float* tile_a = shared_mem;
  float* tile_b = shared_mem + blockDim.y * blockDim.x;

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  const int tiles = div_up(k, blockDim.x);

  for (int tile = 0; tile < tiles; ++tile) {
    const int a_col = tile * blockDim.x + threadIdx.x;
    const int b_row = tile * blockDim.y + threadIdx.y;

    tile_a[threadIdx.y * blockDim.x + threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y * blockDim.x + threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    __syncthreads();

    for (int inner = 0; inner < blockDim.x; ++inner) {
      acc += tile_a[threadIdx.y * blockDim.x + inner] *
             tile_b[inner * blockDim.x + threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

Options parse_args(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
      options.m = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      options.n = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
      options.k = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--tile") == 0 && i + 1 < argc) {
      options.tile_size = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      options.benchmark_config.warmup_iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      options.benchmark_config.measured_iterations = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--no-check") == 0) {
      options.check_results = false;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::cout << "Usage: matmul_bench [--m N] [--n N] [--k N] [--tile N] "
                << "[--warmup N] [--iters N] [--no-check]" << std::endl;
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << argv[i] << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (options.m <= 0 || options.n <= 0 || options.k <= 0) {
    std::cerr << "matrix dimensions must be greater than 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (options.tile_size <= 0 || options.tile_size > 32) {
    std::cerr << "--tile must be in range [1, 32]" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (options.benchmark_config.warmup_iterations < 0 ||
      options.benchmark_config.measured_iterations <= 0) {
    std::cerr << "warmup must be >= 0 and iters must be > 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return options;
}

void fill_matrix(std::vector<float>& matrix, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      matrix[static_cast<std::size_t>(row) * cols + col] =
          static_cast<float>((row * 17 + col * 13) % 101) / 100.0f;
    }
  }
}

bool verify(const std::vector<float>& a,
            const std::vector<float>& b,
            const std::vector<float>& c,
            int m,
            int n,
            int k) {
  constexpr float kTolerance = 1e-3f;

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float expected = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        expected += a[static_cast<std::size_t>(row) * k + inner] *
                    b[static_cast<std::size_t>(inner) * n + col];
      }

      const float actual = c[static_cast<std::size_t>(row) * n + col];
      if (std::fabs(actual - expected) > kTolerance) {
        std::cerr << "Mismatch at (" << row << ", " << col << ")"
                  << ", expected " << expected
                  << ", got " << actual << std::endl;
        return false;
      }
    }
  }

  return true;
}

double tflops(double flops, float milliseconds) {
  if (milliseconds <= 0.0f) {
    return 0.0;
  }
  return flops / (static_cast<double>(milliseconds) * 1.0e-3) / 1.0e12;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_args(argc, argv);

  const std::size_t a_elements =
      static_cast<std::size_t>(options.m) * options.k;
  const std::size_t b_elements =
      static_cast<std::size_t>(options.k) * options.n;
  const std::size_t c_elements =
      static_cast<std::size_t>(options.m) * options.n;

  const std::size_t a_bytes = a_elements * sizeof(float);
  const std::size_t b_bytes = b_elements * sizeof(float);
  const std::size_t c_bytes = c_elements * sizeof(float);

  print_benchmark_header(kBenchmarkName);

  std::vector<float> host_a(a_elements);
  std::vector<float> host_b(b_elements);
  std::vector<float> host_c(c_elements, 0.0f);

  fill_matrix(host_a, options.m, options.k);
  fill_matrix(host_b, options.k, options.n);

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_a), a_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_b), b_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_c), c_bytes));

  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), b_bytes, cudaMemcpyHostToDevice));

  const dim3 block_dim(options.tile_size, options.tile_size);
  const dim3 grid_dim(
      div_up(options.n, options.tile_size),
      div_up(options.m, options.tile_size));
  const std::size_t shared_mem_bytes =
      2ULL * options.tile_size * options.tile_size * sizeof(float);

  const auto launch = [&]() {
    matmul_tiled_kernel<<<grid_dim, block_dim, shared_mem_bytes>>>(
        device_a, device_b, device_c, options.m, options.n, options.k);
  };

  const BenchmarkStats stats =
      benchmark_cuda_kernel(options.benchmark_config, launch);

  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));

  if (options.check_results &&
      !verify(host_a, host_b, host_c, options.m, options.n, options.k)) {
    return EXIT_FAILURE;
  }

  print_benchmark_stats(
      kBenchmarkName,
      c_elements,
      options.tile_size * options.tile_size,
      options.benchmark_config,
      stats,
      a_bytes + b_bytes + c_bytes);

  const double flop_count =
      2.0 * static_cast<double>(options.m) * options.n * options.k;
  std::cout << "matmul shape_m=" << options.m
            << " shape_n=" << options.n
            << " shape_k=" << options.k
            << " tile=" << options.tile_size
            << " mean_tflops=" << tflops(flop_count, stats.mean_ms)
            << " median_tflops=" << tflops(flop_count, stats.median_ms)
            << std::endl;

  return EXIT_SUCCESS;
}
