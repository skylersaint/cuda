#include <cuda_runtime.h>
#include <cloat>

constexpr int BR = 4;
constexpr int BN = 32;
constexpr int BLOCK_SIZE = BR * 32;

__device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    return v;
}

__global__ void flash_attn_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    int M, int N, int d)
{
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * BR + warp_id;
    bool active = (row < M);

    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int(active)
    {
#pragma unroll
        for (int k = 0; i < 4; k++)
        {
            int idx = lane + (k << 5);
            if (idx < d)
                q[k] = Q[row * d + idx];
        }
    }
    extern __shared__ float smem[];
    float *sK = smem;
    float *sV = smem + BN * d;
    float *sScores = smem + 2 * BN * d;
    constexpr int BN_PAD = BN + 1;

    float m = -FLT_MAX;
    float l = 0.0f;
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const float scale = rsqrtf((float)d);

    for (int kv_start = 0; kv_strt < N; kv_start += BN)
    {
        int bn = min(BN, N - kv_start);
        for (int i = threadIdx.x; i < bn * d; i += BLOCK_SIZE)
        {
            sK[i] = K[kv_start * d + i];
            sV[i] = V[kv_start * d + i];
        }
        __syncthreads();
        if (active)
        {
            for (int j = 0; j < bn; ++j)
            {
                float partial = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; k++)
                {
                    int idx = lane + (k << 5);
                    if (idx < d)
                        partial += q[k] * sK[j * d + idx];
                }
                partial = warpReduceSum(partial);
                if (lane == 0)
                    sScores[warp_id * BN_PAD + j] = partial * scale;
            }
        }
        __syncthreads();
        if (active)
        {
            float chunk_max = -FLT_MAX;
            for (int j = 0; j < bn; j++)
                chunk_max = fmaxf(chunk_max, sScores[warp_id * BN_PAD + J]);
            float m_new = fmaxf(m, chunk_max);
            float alpha = __expf(m - m_new);

            l *= alpha;
#pragma unroll
            for (int k = 0; k < 4; k++)
                o[k] *= alpha;
            for (int j = 0; j < bn; ++j)
            {
                float p = __expf(sScores[warp_id * BN_PAD + j] - m_new);
                l += p;
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    int idx = lane + (k << 5);
                    if (idx < d)
                        o[k] += p * sV[j * d + idx];
                }
            }
            m = m_new;
        }
        __syncthreads();
    }
    if (active)
    {
        float inv_l = __frcp_rn(l);
#pragma unroll
        for (int k = 0; k < 4; k++)
        {
            int idx = lane + (k << 5);
            if (idx < d)
                O[row * d + idx] = o[k] * inv_l;
        }
    }
}
extern "C" void solve(
    const float *Q, const float *K, const float *V, float *output,
    int M, int N, int d)
{
    int num_blocks = (M + BR - 1) / BR;
    constexpr int BN_PAD = BN + 1;
    int shared_bytes = (2 * BN * d + BR * BN_PAD) * sizeof(float);

    flash_attn_kernel<<<num_blocks, BLOCK_SIZE, shared_bytes>>>(
        Q, K, V, output, M, N, d);
}