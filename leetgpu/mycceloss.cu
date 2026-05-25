#include <cuda_runtime.h>
#include <float.h>

constexpr int BLOCK_SIZE = 256;
struct MS
{
    float m;
    float s;
};
__device __forceinline__ MS msMerge(MS a, MS b)
{
    MS r;
    r.m = fmaxf(a.m, b.m);
    r.s = a.s * __expf(a.m - r.m) + b.s * __expf(b.m - r.m);
    return r;
}

__device__ __forceinline__ MS warpReduceMS(MS val)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        MS other;
        other.m = __shfl_xor_sync(0xffffffff, val.m, o);
        other.s = __shfl_xor_sync(0xffffffff, val.s, o);
        val = msMerge(val, other);
    }
    return val;
}
__device__ __forceinline__ MS blockReduceMS(MS val)
{
    __shared__ float s_m[32];
    __shared__ float s_s[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warpReduceMS(val);
    if (lane == 0)
    {
        s_m[wid] = val.m;
        s_s[wid] = val.s;
    }
    __syncthreads();
    if (wid == 0)
    {
        MS v;
        if (lane < BLOCK_SIZE / 32)
        {
            v.m = s_m[lane];
            v.s = s_s[lane];
        }
        else
        {
            v.m = -FLT_MAX;
            v.s = 0.0f;
        }
        v = warpReduceMS(v);
        if (lane == 0)
        {
            s_m[0] = v.m;
            s_s[0] = v.s;
        }
    }
    __syncthreads();
    val.m = s_m[0];
    val.s = s_s[0];
    return val;
}

__global__ void cce_kernel(const float *__restrict__ logits,
                           const float *__restrict__ true_labels,
                           float *_restrict__ loss,
                           int N, int C)
{
    int row = blockIdx.x;
    if (row >= N)
        return;
    const float *z = logits + row * C;
    int label = true_labels[row];

    MS val;
    val.m = -FLT_MAX;
    val.s = 0.0f;

    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE)
    {
        float zi = z[i];
        float new_m = fmaxf(val.m, zi);
        val.s = val.s * __expf(val.m - new_m) + __expf(zi - new_m);
        val.m = new_m;
    }
    val = blockReduceMS(val);
    if (threadIdx.x == 0)
    {
        float z_target = z[label];
        float loss_j = val.m + __logf(val.s) - z_target;
        atomicAdd(loss, loss_j / (float)N);
    }
}
extern "C" void solve(const float *logits, const int *true_labels, float *loss, int N, int C)
{
    cudaMemset(loss, 0, sizeof(float));
    cce_kernel<<<N, BLOCK_SIZE>>>(logits, true_labels, loss, N, C);
}