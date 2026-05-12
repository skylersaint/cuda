#include <cuda_runtime.h>
#include <cfloat>

constexpr int BLOCK_SIZE = 512;
constexpr int ELEMS_PER_BLOCK = 4096;

// ============================================================
// Warp 内 online softmax 合并:5 步 shuffle
// ============================================================
__device__ __forceinline__ void warpReduceMS(float &m, float &s)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        float m2 = __shfl_down_sync(0xffffffff, m, off);
        float s2 = __shfl_down_sync(0xffffffff, s, off);
        float nm = fmaxf(m, m2);
        s = s * __expf(m - nm) + s2 * __expf(m2 - nm);
        m = nm;
    }
}

// ============================================================
// Block 内 online softmax 合并:lane 0 of warp 0 拿到结果
// 注意:用 -FLT_MAX 而非 -INFINITY 作为单位元,避免 inf - inf = NaN
// ============================================================
__device__ __forceinline__ void blockReduceMS(float &m, float &s)
{
    __shared__ float sm[32], ss[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    warpReduceMS(m, s);
    if (lane == 0)
    {
        sm[wid] = m;
        ss[wid] = s;
    }
    __syncthreads();

    if (wid == 0)
    {
        constexpr int N_WARPS = BLOCK_SIZE / 32;
        m = (lane < N_WARPS) ? sm[lane] : -FLT_MAX;
        s = (lane < N_WARPS) ? ss[lane] : 0.0f;
        warpReduceMS(m, s);
    }
}

// ============================================================
// Kernel 1: 每个 block 算自己 tile 的 (m, s) partial
// ============================================================
__global__ void softmax_partial(const float *__restrict__ input,
                                float *__restrict__ partial_m,
                                float *__restrict__ partial_s,
                                int N)
{
    int blk_start = blockIdx.x * ELEMS_PER_BLOCK;
    int blk_end = min(blk_start + ELEMS_PER_BLOCK, N);
    if (blk_start >= N)
        return;
    int seg_len = blk_end - blk_start;

    float m = -FLT_MAX;
    float s = 0.0f;

    // Float4 向量化路径(完整 tile 总是 4 的倍数)
    if ((seg_len & 3) == 0)
    {
        const float4 *in4 = reinterpret_cast<const float4 *>(input + blk_start);
        int n4 = seg_len >> 2;
        for (int i = threadIdx.x; i < n4; i += BLOCK_SIZE)
        {
            float4 v = in4[i];
// 串行 online merge 4 个元素
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                float x = (&v.x)[k];
                float nm = fmaxf(m, x);
                s = s * __expf(m - nm) + __expf(x - nm);
                m = nm;
            }
        }
    }
    else
    {
        // 标量 fallback(最后一个 tile 长度不是 4 的倍数时)
        for (int i = blk_start + threadIdx.x; i < blk_end; i += BLOCK_SIZE)
        {
            float x = input[i];
            float nm = fmaxf(m, x);
            s = s * __expf(m - nm) + __expf(x - nm);
            m = nm;
        }
    }

    blockReduceMS(m, s);

    if (threadIdx.x == 0)
    {
        partial_m[blockIdx.x] = m;
        partial_s[blockIdx.x] = s;
    }
}

// ============================================================
// Kernel 2: 每个 block redundantly 合并所有 partial,然后写自己 tile 的输出
// ============================================================
__global__ void softmax_output(const float *__restrict__ input,
                               const float *__restrict__ partial_m,
                               const float *__restrict__ partial_s,
                               float *__restrict__ output,
                               int N, int num_partials)
{
    int blk_start = blockIdx.x * ELEMS_PER_BLOCK;
    int blk_end = min(blk_start + ELEMS_PER_BLOCK, N);
    if (blk_start >= N)
        return;
    int seg_len = blk_end - blk_start;

    // ── 每个 block 独立合并所有 partials,得到全局 (m, s) ──
    float m = -FLT_MAX;
    float s = 0.0f;
    for (int i = threadIdx.x; i < num_partials; i += BLOCK_SIZE)
    {
        float m2 = partial_m[i];
        float s2 = partial_s[i];
        float nm = fmaxf(m, m2);
        s = s * __expf(m - nm) + s2 * __expf(m2 - nm);
        m = nm;
    }
    blockReduceMS(m, s);

    // ── 广播 (m, 1/s) 到整个 block ──
    __shared__ float g_m, g_inv_s;
    if (threadIdx.x == 0)
    {
        g_m = m;
        g_inv_s = 1.0f / s;
    }
    __syncthreads();
    float final_m = g_m;
    float inv_s = g_inv_s;

    // ── 写出归一化结果(向量化) ──
    if ((seg_len & 3) == 0)
    {
        const float4 *in4 = reinterpret_cast<const float4 *>(input + blk_start);
        float4 *out4 = reinterpret_cast<float4 *>(output + blk_start);
        int n4 = seg_len >> 2;
        for (int i = threadIdx.x; i < n4; i += BLOCK_SIZE)
        {
            float4 v = in4[i];
            v.x = __expf(v.x - final_m) * inv_s;
            v.y = __expf(v.y - final_m) * inv_s;
            v.z = __expf(v.z - final_m) * inv_s;
            v.w = __expf(v.w - final_m) * inv_s;
            out4[i] = v;
        }
    }
    else
    {
        for (int i = blk_start + threadIdx.x; i < blk_end; i += BLOCK_SIZE)
        {
            output[i] = __expf(input[i] - final_m) * inv_s;
        }
    }
}

// ============================================================
// Host 端:用 static workspace 避免重复 cudaMalloc 的开销
// ============================================================
static float *d_workspace = nullptr;
static int d_workspace_size = 0;

extern "C" void solve(const float *input, float *output, int N)
{
    int num_blocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    int needed = 2 * num_blocks;

    if (needed > d_workspace_size)
    {
        if (d_workspace)
            cudaFree(d_workspace);
        cudaMalloc(&d_workspace, needed * sizeof(float));
        d_workspace_size = needed;
    }

    float *partial_m = d_workspace;
    float *partial_s = d_workspace + num_blocks;

    softmax_partial<<<num_blocks, BLOCK_SIZE>>>(input, partial_m, partial_s, N);
    softmax_output<<<num_blocks, BLOCK_SIZE>>>(input, partial_m, partial_s,
                                               output, N, num_blocks);
}