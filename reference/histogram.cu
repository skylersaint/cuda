#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 512;
constexpr int ELEMS_PER_THREAD = 16;
constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE * ELEMS_PER_THREAD;
constexpr int MAX_BINS = 1024;

__global__ void hist_kernel(const int *__restrict__ input,
                            int *__restrict__ histogram,
                            int N, int num_bins)
{
    __shared__ int sHist[MAX_BINS];

    // ── 1. 初始化 shared 直方图 ──
    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
    {
        sHist[i] = 0;
    }
    __syncthreads();

    // ── 2. 每线程处理 ELEMS_PER_THREAD 个元素,atomic 到 shared ──
    int base = blockIdx.x * ELEMS_PER_BLOCK;
#pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k)
    {
        int idx = base + k * BLOCK_SIZE + threadIdx.x;
        if (idx < N)
        {
            atomicAdd(&sHist[input[idx]], 1);
        }
    }
    __syncthreads();

    // ── 3. 合并 shared 直方图到 global ──
    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
    {
        int count = sHist[i];
        if (count > 0)
        {
            atomicAdd(&histogram[i], count);
        }
    }
}

extern "C" void solve(const int *input, int *histogram, int N, int num_bins)
{
    int num_blocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

    // 假设 histogram 已经清零(LeetGPU 通常会清零,不放心可以加 cudaMemset)
    cudaMemset(histogram, 0, num_bins * sizeof(int));

    hist_kernel<<<num_blocks, BLOCK_SIZE>>>(input, histogram, N, num_bins);
}