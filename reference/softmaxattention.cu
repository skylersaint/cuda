#include <cuda_runtime.h>
#include <cfloat>

constexpr int BR = 4;               // 每 block 处理的 Q 行数(= warps per block)
constexpr int BN = 32;              // KV chunk 大小
constexpr int BLOCK_SIZE = BR * 32; // 128 线程

__device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, o);
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

    // Q 行加载到寄存器:每 lane 持有 d/32 个元素(d ≤ 128 → 最多 4 个)
    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active)
    {
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int idx = lane + (k << 5);
            if (idx < d)
                q[k] = Q[row * d + idx];
        }
    }

    // Shared memory:K, V 块 + 每 warp 的 scores
    extern __shared__ float smem[];
    float *sK = smem;                   // [BN][d]
    float *sV = smem + BN * d;          // [BN][d]
    float *sScores = smem + 2 * BN * d; // [BR][BN+1] (+1 防 bank conflict)
    constexpr int BN_PAD = BN + 1;

    // online softmax 状态
    float m = -FLT_MAX;
    float l = 0.0f;
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const float scale = rsqrtf((float)d);

    // ============================================================
    // 主循环:遍历 KV chunks
    // ============================================================
    for (int kv_start = 0; kv_start < N; kv_start += BN)
    {
        int bn = min(BN, N - kv_start);

        // ── 4 warp 协作加载 K, V chunk ──
        for (int i = threadIdx.x; i < bn * d; i += BLOCK_SIZE)
        {
            sK[i] = K[kv_start * d + i];
            sV[i] = V[kv_start * d + i];
        }
        __syncthreads();

        // ── 每个 warp 算自己行的 bn 个 scores ──
        if (active)
        {
            for (int j = 0; j < bn; ++j)
            {
                float partial = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k)
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

        // ── Online softmax 更新 + 输出累加 ──
        if (active)
        {
            // 找 chunk 内最大 score
            float chunk_max = -FLT_MAX;
            for (int j = 0; j < bn; ++j)
                chunk_max = fmaxf(chunk_max, sScores[warp_id * BN_PAD + j]);

            float m_new = fmaxf(m, chunk_max);
            float alpha = __expf(m - m_new);

            // 缩放旧累加器
            l *= alpha;
#pragma unroll
            for (int k = 0; k < 4; ++k)
                o[k] *= alpha;

            // 累加新贡献
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

    // ── 最终归一化并写出 ──
    if (active)
    {
        float inv_l = __frcp_rn(l);
#pragma unroll
        for (int k = 0; k < 4; ++k)
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