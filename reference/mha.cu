#include <cuda_runtime.h>
#include <float.h>

constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, o);
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, o));
    return val;
}

__device__ __forceinline__ float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

__global__ void mha_kernel(const float* __restrict__ Q,
                           const float* __restrict__ K,
                           const float* __restrict__ V,
                           float* __restrict__ output,
                           int N, int d_model, int h, int d_k) {
    int query_row = blockIdx.x;   // 0..N-1
    int head      = blockIdx.y;   // 0..h-1
    int tid       = threadIdx.x;

    if (query_row >= N || head >= h) return;

    int head_offset = head * d_k;   // 该 head 在 d_model 中的列偏移

    // Q 的当前 query row + head 段
    const float* q_ptr = Q + query_row * d_model + head_offset;

    extern __shared__ float smem[];
    float* s_q     = smem;             // [d_k] 缓存 query
    float* s_scores = smem + d_k;      // [N] 缓存 scores

    // 加载 query 到 shared
    for (int i = tid; i < d_k; i += BLOCK_SIZE) {
        s_q[i] = q_ptr[i];
    }
    __syncthreads();

    float inv_sqrt_dk = rsqrtf((float)d_k);

    // === 阶段 1: scores[j] = (Q_row · K_j) / √d_k ===
    for (int j = tid; j < N; j += BLOCK_SIZE) {
        const float* k_ptr = K + j * d_model + head_offset;
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) {
            dot = fmaf(s_q[d], k_ptr[d], dot);
        }
        s_scores[j] = dot * inv_sqrt_dk;
    }
    __syncthreads();

    // === 阶段 2: softmax ===
    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += BLOCK_SIZE) {
        local_max = fmaxf(local_max, s_scores[j]);
    }
    float row_max = blockReduceMax(local_max);

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += BLOCK_SIZE) {
        float e = __expf(s_scores[j] - row_max);
        s_scores[j] = e;
        local_sum += e;
    }
    float row_sum = blockReduceSum(local_sum);
    float inv_sum = 1.0f / row_sum;
    __syncthreads();

    // === 阶段 3: output[d] = Σ_j softmax[j] · V[j][d] ===
    float* out_ptr = output + query_row * d_model + head_offset;
    for (int d = tid; d < d_k; d += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) {
            float w = s_scores[j] * inv_sum;
            float v_val = V[j * d_model + head_offset + d];
            acc = fmaf(w, v_val, acc);
        }
        out_ptr[d] = acc;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                       float* output, int N, int d_model, int h) {
    int d_k = d_model / h;

    dim3 grid(N, h);
    dim3 block(BLOCK_SIZE);
    size_t smem = (d_k + N) * sizeof(float);

    mha_kernel<<<grid, block, smem>>>(Q, K, V, output, N, d_model, h, d_k);
}