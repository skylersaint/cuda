// softmax_multiblock.cu
// 多 block 处理一行的 softmax,适合 D 较大的场景

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                        \
    do                                          \
    {                                           \
        cudaError_t e = (call);                 \
        if (e != cudaSuccess)                   \
        {                                       \
            fprintf(stderr, "CUDA error: %s\n", \
                    cudaGetErrorString(e));     \
            exit(1);                            \
        }                                       \
    } while (0)

constexpr int BLOCK_SIZE = 256;
constexpr int ELEMS_PER_BLOCK = 4096; // 每个 block 处理 4096 个元素

// ============================================================
// Warp / Block 归约工具
// ============================================================
__device__ __forceinline__ float warpReduceMax(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// ============================================================
// 关键:Online Softmax 的两个值 (m, s) 的 warp/block 归约
// 一个 warp 内合并 32 个 (m, s) 对
// ============================================================
__device__ __forceinline__ void warpReduceMS(float &m, float &s)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        float m_other = __shfl_down_sync(0xffffffff, m, o);
        float s_other = __shfl_down_sync(0xffffffff, s, o);
        float m_new = fmaxf(m, m_other);
        // online merge:两边的 s 各自缩放到新 max
        s = s * __expf(m - m_new) + s_other * __expf(m_other - m_new);
        m = m_new;
    }
}

// block 内合并 (m, s),lane 0 拿到结果,其他 lane 未定义
__device__ __forceinline__ void blockReduceMS(float &m, float &s)
{
    __shared__ float s_m[32], s_s[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    warpReduceMS(m, s);
    if (lane == 0)
    {
        s_m[wid] = m;
        s_s[wid] = s;
    }
    __syncthreads();

    if (wid == 0)
    {
        m = (threadIdx.x < BLOCK_SIZE / 32) ? s_m[lane] : -INFINITY;
        s = (threadIdx.x < BLOCK_SIZE / 32) ? s_s[lane] : 0.0f;
        warpReduceMS(m, s);
    }
}

// ============================================================
// Kernel 1: Partial Reduce
// 每个 block 处理一行的一段,产出该段的 (m, s)
// grid 维度:(num_blocks_per_row, N) —— x 维表示行内的段,y 维表示行
// ============================================================
__global__ void partial_reduce_kernel(const float *__restrict__ input,
                                      float *__restrict__ partial_m,
                                      float *__restrict__ partial_s,
                                      int N, int D, int num_blocks_per_row)
{
    int row = blockIdx.y;
    int block_x = blockIdx.x;

    int seg_start = block_x * ELEMS_PER_BLOCK;
    int seg_end = min(seg_start + ELEMS_PER_BLOCK, D);
    if (seg_start >= D)
        return;

    const float *row_in = input + row * D;

    // ─── Online softmax 串行扫描:每个线程维护自己的 (m, s) ───
    float m = -INFINITY;
    float s = 0.0f;
    for (int i = seg_start + threadIdx.x; i < seg_end; i += BLOCK_SIZE)
    {
        float x = row_in[i];
        float m_new = fmaxf(m, x);
        s = s * __expf(m - m_new) + __expf(x - m_new);
        m = m_new;
    }

    // ─── Block 内合并 (m, s) ───
    blockReduceMS(m, s);

    // ─── 线程 0 写出这个段的 (m, s) ───
    if (threadIdx.x == 0)
    {
        int idx = row * num_blocks_per_row + block_x;
        partial_m[idx] = m;
        partial_s[idx] = s;
    }
}

// ============================================================
// Kernel 2: Final Reduce + 写出 row_m, row_s
// 每个 block 处理一行的所有 partial,合并出最终 (row_m, row_s)
// grid 维度:(N,)
// ============================================================
__global__ void final_reduce_kernel(const float *__restrict__ partial_m,
                                    const float *__restrict__ partial_s,
                                    float *__restrict__ row_m,
                                    float *__restrict__ row_s,
                                    int N, int num_blocks_per_row)
{
    int row = blockIdx.x;

    const float *pm = partial_m + row * num_blocks_per_row;
    const float *ps = partial_s + row * num_blocks_per_row;

    float m = -INFINITY;
    float s = 0.0f;

    // 每个线程负责一部分 partial(假设 num_blocks_per_row <= BLOCK_SIZE,通常成立)
    for (int i = threadIdx.x; i < num_blocks_per_row; i += BLOCK_SIZE)
    {
        float m_i = pm[i];
        float s_i = ps[i];
        float m_new = fmaxf(m, m_i);
        s = s * __expf(m - m_new) + s_i * __expf(m_i - m_new);
        m = m_new;
    }

    blockReduceMS(m, s);

    if (threadIdx.x == 0)
    {
        row_m[row] = m;
        row_s[row] = s;
    }
}

// ============================================================
// Kernel 3: 写出归一化结果
// 每个 block 处理一行的一段,读全局 row_m 和 row_s
// grid 维度:(num_blocks_per_row, N)
// ============================================================
__global__ void output_kernel(const float *__restrict__ input,
                              const float *__restrict__ row_m,
                              const float *__restrict__ row_s,
                              float *__restrict__ output,
                              int N, int D)
{
    int row = blockIdx.y;
    int block_x = blockIdx.x;

    int seg_start = block_x * ELEMS_PER_BLOCK;
    int seg_end = min(seg_start + ELEMS_PER_BLOCK, D);
    if (seg_start >= D)
        return;

    // 所有线程读同一个 row_m 和 row_s(broadcast,无 conflict)
    float m = row_m[row];
    float inv_s = 1.0f / row_s[row];

    const float *in = input + row * D;
    float *out = output + row * D;

    for (int i = seg_start + threadIdx.x; i < seg_end; i += BLOCK_SIZE)
    {
        out[i] = __expf(in[i] - m) * inv_s;
    }
}

// ============================================================
// Host 端封装
// ============================================================
void launchSoftmaxMultiBlock(const float *d_input, float *d_output, int N, int D)
{
    int num_blocks_per_row = (D + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

    // 分配 partial buffer 和 row 结果
    float *d_partial_m, *d_partial_s, *d_row_m, *d_row_s;
    CUDA_CHECK(cudaMalloc(&d_partial_m, N * num_blocks_per_row * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_s, N * num_blocks_per_row * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_m, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_s, N * sizeof(float)));

    // Kernel 1
    dim3 grid1(num_blocks_per_row, N);
    partial_reduce_kernel<<<grid1, BLOCK_SIZE>>>(
        d_input, d_partial_m, d_partial_s, N, D, num_blocks_per_row);

    // Kernel 2
    final_reduce_kernel<<<N, BLOCK_SIZE>>>(
        d_partial_m, d_partial_s, d_row_m, d_row_s, N, num_blocks_per_row);

    // Kernel 3
    dim3 grid3(num_blocks_per_row, N);
    output_kernel<<<grid3, BLOCK_SIZE>>>(
        d_input, d_row_m, d_row_s, d_output, N, D);

    CUDA_CHECK(cudaFree(d_partial_m));
    CUDA_CHECK(cudaFree(d_partial_s));
    CUDA_CHECK(cudaFree(d_row_m));
    CUDA_CHECK(cudaFree(d_row_s));
}