#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int TILE = 4096;
constexpr int VEC_SIZE = 4;

__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, o);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val)
{
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
    if (wid == 0)
        val = warpReduceSum(val);
    return val;
}

__global__ void dot_kernel(const float *__restrict__ A,
                           const float *__restrict__ B,
                           float *__restrict__ result,
                           int N)
{
    int blk_start = blockIdx.x * TILE;
    int blk_end = min(blk_start + TILE, N);
    if (blk_start >= N)
        return;
    int seg_len = blk_end - blk_start;

    float val = 0.0f;

    // Float4 向量化路径:一次读 4 对,4 次 FMA
    if ((seg_len & 3) == 0)
    {
        const float4 *A4 = reinterpret_cast<const float4 *>(A + blk_start);
        const float4 *B4 = reinterpret_cast<const float4 *>(B + blk_start);
        int n4 = seg_len >> 2;
        for (int i = threadIdx.x; i < n4; i += BLOCK_SIZE)
        {
            float4 a = A4[i];
            float4 b = B4[i];
            // 4 次独立 FMA,编译器能 ILP 调度
            val = fmaf(a.x, b.x, val);
            val = fmaf(a.y, b.y, val);
            val = fmaf(a.z, b.z, val);
            val = fmaf(a.w, b.w, val);
        }
    }
    else
    {
        // 标量 fallback(最后一个 block 不满 4 倍数时)
        for (int i = blk_start + threadIdx.x; i < blk_end; i += BLOCK_SIZE)
        {
            val = fmaf(A[i], B[i], val);
        }
    }

    val = blockReduceSum(val);

    if (threadIdx.x == 0)
    {
        atomicAdd(result, val);
    }
}

extern "C" void solve(const float *A, const float *B, float *result, int N)
{
    int num_blocks = (N + TILE - 1) / TILE;

    // 假设 LeetGPU 已经把 result 清零;不放心可以加:
    // cudaMemset(result, 0, sizeof(float));

    dot_kernel<<<num_blocks, BLOCK_SIZE>>>(A, B, result, N);
}