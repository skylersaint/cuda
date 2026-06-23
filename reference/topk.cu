#include <cuda_runtime.h>
#include <float.h>

constexpr int BLOCK_SIZE = 256;
constexpr int BITS_PER_PASS = 8;
constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS; // 256
constexpr int NUM_PASSES = 32 / BITS_PER_PASS;  // 4

// ============================================================
// float ↔ sortable uint 转换
// ============================================================
__device__ __forceinline__ unsigned int float_to_sortable(float f)
{
    unsigned int u = __float_as_uint(f);
    unsigned int mask = (unsigned int)(-(int)(u >> 31)) | 0x80000000u;
    return u ^ mask;
}

// ============================================================
// Kernel 1: 直方图统计
//   在 prefix 约束下,统计当前 pass 各 bucket 的元素数
// ============================================================
__global__ void histogram_kernel(const float *__restrict__ input,
                                 int N,
                                 unsigned int prefix,
                                 unsigned int prefix_mask,
                                 int shift,
                                 unsigned int *__restrict__ global_hist)
{
    __shared__ unsigned int s_hist[NUM_BUCKETS];

    // 初始化 shared 直方图
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += BLOCK_SIZE)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // grid-stride 遍历 input
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;
    for (int i = idx; i < N; i += stride)
    {
        unsigned int key = float_to_sortable(input[i]);
        if ((key & prefix_mask) == prefix)
        {
            unsigned int bucket = (key >> shift) & (NUM_BUCKETS - 1);
            atomicAdd(&s_hist[bucket], 1u);
        }
    }
    __syncthreads();

    // 合并到 global
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += BLOCK_SIZE)
    {
        unsigned int v = s_hist[i];
        if (v > 0)
            atomicAdd(&global_hist[i], v);
    }
}

// ============================================================
// Kernel 2: 筛选 > threshold 的元素到 output
// ============================================================
__global__ void select_greater_kernel(const float *__restrict__ input,
                                      int N,
                                      unsigned int threshold,
                                      float *__restrict__ output,
                                      int *__restrict__ out_count,
                                      int k)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;
    for (int i = idx; i < N; i += stride)
    {
        unsigned int key = float_to_sortable(input[i]);
        if (key > threshold)
        {
            int pos = atomicAdd(out_count, 1);
            if (pos < k)
            {
                output[pos] = input[i];
            }
        }
    }
}

// ============================================================
// Kernel 3: 筛选 = threshold 的元素,凑够 k 个
// ============================================================
__global__ void select_equal_kernel(const float *__restrict__ input,
                                    int N,
                                    unsigned int threshold,
                                    float *__restrict__ output,
                                    int *__restrict__ out_count,
                                    int k)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;
    for (int i = idx; i < N; i += stride)
    {
        unsigned int key = float_to_sortable(input[i]);
        if (key == threshold)
        {
            int pos = atomicAdd(out_count, 1);
            if (pos < k)
            {
                output[pos] = input[i];
            }
        }
    }
}

// ============================================================
// Kernel 4: 对 k 个元素降序排序 (bitonic sort, 单 block)
//   假设 k <= 1024
// ============================================================
__global__ void sort_desc_kernel(float *data, int k)
{
    __shared__ float s[1024];
    int tid = threadIdx.x;

    // 加载,不足补 -FLT_MAX
    s[tid] = (tid < k) ? data[tid] : -FLT_MAX;
    __syncthreads();

    // Bitonic sort 降序(大的在前)
    for (int size = 2; size <= 1024; size <<= 1)
    {
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            int partner = tid ^ stride;
            if (partner > tid)
            {
                bool ascending = ((tid & size) == 0);
                float a = s[tid];
                float b = s[partner];
                // 降序: 大的应该在小 tid 位置
                bool should_swap = ascending ? (a < b) : (a > b);
                if (should_swap)
                {
                    s[tid] = b;
                    s[partner] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid < k)
        data[tid] = s[tid];
}

// ============================================================
// Host 端 solve
// ============================================================
extern "C" void solve(const float *input, float *output, int N, int k)
{
    // 工作内存
    unsigned int *d_hist;
    int *d_count;
    cudaMalloc(&d_hist, NUM_BUCKETS * sizeof(unsigned int));
    cudaMalloc(&d_count, sizeof(int));

    int num_blocks = min(1024, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ===== 阶段 1: Radix Select 找第 k 大的阈值 =====
    unsigned int prefix = 0;
    unsigned int prefix_mask = 0;
    int remaining_k = k;

    unsigned int hist[NUM_BUCKETS];

    for (int pass = 0; pass < NUM_PASSES; pass++)
    {
        int shift = 32 - (pass + 1) * BITS_PER_PASS;

        cudaMemset(d_hist, 0, NUM_BUCKETS * sizeof(unsigned int));
        histogram_kernel<<<num_blocks, BLOCK_SIZE>>>(
            input, N, prefix, prefix_mask, shift, d_hist);

        cudaMemcpy(hist, d_hist, NUM_BUCKETS * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);

        // 从高 bucket 往低,找第 remaining_k 大落在哪个 bucket
        unsigned int cumulative = 0;
        int target_bucket = 0;
        for (int b = NUM_BUCKETS - 1; b >= 0; b--)
        {
            if (cumulative + hist[b] >= (unsigned int)remaining_k)
            {
                target_bucket = b;
                break;
            }
            cumulative += hist[b];
        }

        // 更新 prefix
        prefix |= ((unsigned int)target_bucket << shift);
        prefix_mask |= ((unsigned int)(NUM_BUCKETS - 1) << shift);
        remaining_k -= cumulative;
    }

    unsigned int threshold = prefix;

    // ===== 阶段 2: 筛选 =====
    cudaMemset(d_count, 0, sizeof(int));
    select_greater_kernel<<<num_blocks, BLOCK_SIZE>>>(
        input, N, threshold, output, d_count, k);

    int count_after_greater;
    cudaMemcpy(&count_after_greater, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (count_after_greater < k)
    {
        select_equal_kernel<<<num_blocks, BLOCK_SIZE>>>(
            input, N, threshold, output, d_count, k);
    }

    // ===== 阶段 3: 排序 =====
    sort_desc_kernel<<<1, 1024>>>(output, k);

    cudaFree(d_hist);
    cudaFree(d_count);
}