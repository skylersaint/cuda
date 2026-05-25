#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

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

__global__ void spmv_kernel(const float *__restrict__ A,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int M, int N)
{
    int row = blockIdx.x;
    if (row >= M)
        return;

    const float *row_A = A + row * N;
    float sum = 0.0f;

    // Float4 向量化路径
    int n4 = N >> 2;
    if ((N & 3) == 0)
    {
        const float4 *A4 = reinterpret_cast<const float4 *>(row_A);
        const float4 *x4 = reinterpret_cast<const float4 *>(x);
        for (int i = threadIdx.x; i < n4; i += BLOCK_SIZE)
        {
            float4 a = A4[i];
            float4 xv = x4[i];
            sum = fmaf(a.x, xv.x, sum);
            sum = fmaf(a.y, xv.y, sum);
            sum = fmaf(a.z, xv.z, sum);
            sum = fmaf(a.w, xv.w, sum);
        }
    }
    else
    {
        // 标量 fallback(N 不是 4 倍数时)
        for (int i = threadIdx.x; i < N; i += BLOCK_SIZE)
        {
            sum = fmaf(row_A[i], x[i], sum);
        }
    }

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0)
    {
        y[row] = sum;
    }
}

extern "C" void solve(const float *A, const float *x, float *y,
                      int M, int N, int nnz)
{
    spmv_kernel<<<M, BLOCK_SIZE>>>(A, x, y, M, N);
}