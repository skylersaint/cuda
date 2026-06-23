/*
Kernel 1: RMSNorm1 → normed1 [T, 512]
Kernel 2: QKV projection → Q[T,512], K[T,128], V[T,128]
Kernel 3: RoPE on Q, K
Kernel 4: Causal GQA attention → attn_out [T, 512]
Kernel 5: Output projection + residual → x' [T, 512]
Kernel 6: RMSNorm2 → normed2 [T, 512]
Kernel 7: Gate & Up projection → gate[T,1408], up[T,1408]
Kernel 8: SiLU(gate) ⊙ up → ffn_mid [T, 1408]
Kernel 9: Down projection + residual → output [T, 512]
*/
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// ============ 维度常量 ============
constexpr int D_MODEL = 512;
constexpr int N_Q_HEADS = 8;
constexpr int N_KV_HEADS = 2;
constexpr int HEAD_DIM = 64;
constexpr int FFN_HIDDEN = 1408;
constexpr int Q_DIM = N_Q_HEADS * HEAD_DIM;   // 512
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; // 128
constexpr float EPS = 1e-5f;

// ============ 权重偏移(按题目给的 layout)============
constexpr int OFF_W1 = 0;
constexpr int OFF_WQ = 512;
constexpr int OFF_WK = 262656;
constexpr int OFF_WV = 328192;
constexpr int OFF_WO = 393728;
constexpr int OFF_W2 = 655872;
constexpr int OFF_WGATE = 656384;
constexpr int OFF_WUP = 1377280;
constexpr int OFF_WDOWN = 2098176;

constexpr int BLOCK_SIZE = 256;

// ============================================================
// RMSNorm: 每个 block 处理一行(token)
// out[t][i] = x[t][i] / rms(x[t]) * w[i]
// ============================================================
__global__ void rmsnorm_kernel(const float *__restrict__ x,
                               const float *__restrict__ w,
                               float *__restrict__ out,
                               int T, int dim)
{
    int t = blockIdx.x;
    if (t >= T)
        return;
    int tid = threadIdx.x;
    const float *x_row = x + t * dim;

    // 算平方和
    float local = 0.0f;
    for (int i = tid; i < dim; i += BLOCK_SIZE)
    {
        float v = x_row[i];
        local += v * v;
    }
    // block reduce sum
    __shared__ float shared[32];
    int lane = tid & 31, wid = tid >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        local += __shfl_xor_sync(0xffffffff, local, o);
    if (lane == 0)
        shared[wid] = local;
    __syncthreads();
    local = (tid < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        local += __shfl_xor_sync(0xffffffff, local, o);
    if (tid == 0)
        shared[0] = local;
    __syncthreads();
    float ms = shared[0] / dim;
    float inv_rms = 1.0f / sqrtf(ms + EPS);

    // 归一化 + 缩放
    float *out_row = out + t * dim;
    for (int i = tid; i < dim; i += BLOCK_SIZE)
    {
        out_row[i] = x_row[i] * inv_rms * w[i];
    }
}

// ============================================================
// 通用矩阵向量乘(每个 token 独立):out = in @ W^T
// W shape (out_dim, in_dim) row-major
// 每个 block 处理一个 (token, 若干 out_dim)
// ============================================================
__global__ void linear_kernel(const float *__restrict__ in,
                              const float *__restrict__ W,
                              float *__restrict__ out,
                              int T, int in_dim, int out_dim)
{
    int t = blockIdx.x;                             // token
    int od = blockIdx.y * BLOCK_SIZE + threadIdx.x; // output dim
    if (t >= T || od >= out_dim)
        return;

    const float *in_row = in + t * in_dim;
    const float *w_row = W + od * in_dim;
    float acc = 0.0f;
    for (int i = 0; i < in_dim; i++)
    {
        acc = fmaf(in_row[i], w_row[i], acc);
    }
    out[t * out_dim + od] = acc;
}

// ============================================================
// RoPE: 对 Q 或 K 应用旋转
// x shape (T, n_heads, head_dim), cos/sin shape (T, head_dim/2=32)
// RoPE: [q1|q2] -> [q1*cos - q2*sin | q1*sin + q2*cos]
//   q1 = x[:32], q2 = x[32:]
// ============================================================
__global__ void rope_kernel(float *__restrict__ x,
                            const float *__restrict__ cos_,
                            const float *__restrict__ sin_,
                            int T, int n_heads, int head_dim)
{
    int half = head_dim / 2; // 32
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = T * n_heads * half;
    if (idx >= total)
        return;

    // 解析 idx: (t, head, i) where i in [0, half)
    int i = idx % half;
    int tmp = idx / half;
    int head = tmp % n_heads;
    int t = tmp / n_heads;

    // x 的位置: [t][head][i] 和 [t][head][i+half]
    int base = (t * n_heads + head) * head_dim;
    float q1 = x[base + i];
    float q2 = x[base + i + half];

    float c = cos_[t * half + i];
    float s = sin_[t * half + i];

    x[base + i] = q1 * c - q2 * s;
    x[base + i + half] = q1 * s + q2 * c;
}

// ============================================================
// Causal GQA Attention
// Q (T, 8, 64), K (T, 2, 64), V (T, 2, 64)
// GQA: Q head h 用 KV head (h / 4)
// 每个 block 处理一个 (query_token, q_head)
// ============================================================
__global__ void gqa_attention_kernel(const float *__restrict__ Q,
                                     const float *__restrict__ K,
                                     const float *__restrict__ V,
                                     float *__restrict__ attn_out,
                                     int T)
{
    int qt = blockIdx.x; // query token
    int qh = blockIdx.y; // query head (0..7)
    if (qt >= T || qh >= N_Q_HEADS)
        return;

    int kvh = qh / (N_Q_HEADS / N_KV_HEADS); // GQA: which KV head (qh/4)
    int tid = threadIdx.x;

    const float *q_ptr = Q + (qt * N_Q_HEADS + qh) * HEAD_DIM;

    extern __shared__ float smem[];
    float *s_q = smem;                 // [HEAD_DIM]
    float *s_scores = smem + HEAD_DIM; // [qt+1]

    // 加载 query
    for (int i = tid; i < HEAD_DIM; i += BLOCK_SIZE)
    {
        s_q[i] = q_ptr[i];
    }
    __syncthreads();

    float inv_sqrt = 1.0f / sqrtf((float)HEAD_DIM);
    int n_keys = qt + 1; // causal

    // 阶段 1: scores
    for (int j = tid; j < n_keys; j += BLOCK_SIZE)
    {
        const float *k_ptr = K + (j * N_KV_HEADS + kvh) * HEAD_DIM;
        float dot = 0.0f;
        for (int dd = 0; dd < HEAD_DIM; dd++)
        {
            dot = fmaf(s_q[dd], k_ptr[dd], dot);
        }
        s_scores[j] = dot * inv_sqrt;
    }
    __syncthreads();

    // 阶段 2: softmax (用 block reduce)
    __shared__ float sh_max, sh_sum;
    float local_max = -FLT_MAX;
    for (int j = tid; j < n_keys; j += BLOCK_SIZE)
        local_max = fmaxf(local_max, s_scores[j]);
    // block reduce max
    __shared__ float red[32];
    int lane = tid & 31, wid = tid >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    if (lane == 0)
        red[wid] = local_max;
    __syncthreads();
    local_max = (tid < BLOCK_SIZE / 32) ? red[lane] : -FLT_MAX;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    if (tid == 0)
        sh_max = local_max;
    __syncthreads();
    float row_max = sh_max;

    float local_sum = 0.0f;
    for (int j = tid; j < n_keys; j += BLOCK_SIZE)
    {
        float e = expf(s_scores[j] - row_max);
        s_scores[j] = e;
        local_sum += e;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, o);
    if (lane == 0)
        red[wid] = local_sum;
    __syncthreads();
    local_sum = (tid < BLOCK_SIZE / 32) ? red[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, o);
    if (tid == 0)
        sh_sum = local_sum;
    __syncthreads();
    float inv_sum = 1.0f / sh_sum;

    // 阶段 3: output = Σ softmax * V
    float *out_ptr = attn_out + (qt * N_Q_HEADS + qh) * HEAD_DIM;
    for (int dd = tid; dd < HEAD_DIM; dd += BLOCK_SIZE)
    {
        float acc = 0.0f;
        for (int j = 0; j < n_keys; j++)
        {
            float v = V[(j * N_KV_HEADS + kvh) * HEAD_DIM + dd];
            acc = fmaf(s_scores[j] * inv_sum, v, acc);
        }
        out_ptr[dd] = acc;
    }
}

// ============================================================
// 残差加法: out = a + b
// ============================================================
__global__ void add_residual_kernel(const float *a, const float *b,
                                    float *out, int total)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < total)
        out[idx] = a[idx] + b[idx];
}

// ============================================================
// SwiGLU: out = SiLU(gate) * up
// SiLU(x) = x * sigmoid(x)
// ============================================================
__global__ void swiglu_kernel(const float *__restrict__ gate,
                              const float *__restrict__ up,
                              float *__restrict__ out, int total)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total)
        return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g)); // g * sigmoid(g)
    out[idx] = silu * up[idx];
}

// ============================================================
// Host 编排
// ============================================================
extern "C" void solve(const float *x, float *output, const float *weights,
                      const float *cos, const float *sin, int seq_len)
{
    int T = seq_len;

    // 中间 buffer
    float *normed1, *Q, *K, *V, *attn_out, *proj_out, *x_prime;
    float *normed2, *gate, *up, *ffn_mid;
    cudaMalloc(&normed1, T * D_MODEL * sizeof(float));
    cudaMalloc(&Q, T * Q_DIM * sizeof(float));
    cudaMalloc(&K, T * KV_DIM * sizeof(float));
    cudaMalloc(&V, T * KV_DIM * sizeof(float));
    cudaMalloc(&attn_out, T * Q_DIM * sizeof(float));
    cudaMalloc(&proj_out, T * D_MODEL * sizeof(float));
    cudaMalloc(&x_prime, T * D_MODEL * sizeof(float));
    cudaMalloc(&normed2, T * D_MODEL * sizeof(float));
    cudaMalloc(&gate, T * FFN_HIDDEN * sizeof(float));
    cudaMalloc(&up, T * FFN_HIDDEN * sizeof(float));
    cudaMalloc(&ffn_mid, T * FFN_HIDDEN * sizeof(float));

    const float *Wq = weights + OFF_WQ;
    const float *Wk = weights + OFF_WK;
    const float *Wv = weights + OFF_WV;
    const float *Wo = weights + OFF_WO;
    const float *w1 = weights + OFF_W1;
    const float *w2 = weights + OFF_W2;
    const float *Wgate = weights + OFF_WGATE;
    const float *Wup = weights + OFF_WUP;
    const float *Wdown = weights + OFF_WDOWN;

    // === Attention 部分 ===
    // RMSNorm1
    rmsnorm_kernel<<<T, BLOCK_SIZE>>>(x, w1, normed1, T, D_MODEL);

    // QKV projection
    linear_kernel<<<dim3(T, (Q_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        normed1, Wq, Q, T, D_MODEL, Q_DIM);
    linear_kernel<<<dim3(T, (KV_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        normed1, Wk, K, T, D_MODEL, KV_DIM);
    linear_kernel<<<dim3(T, (KV_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        normed1, Wv, V, T, D_MODEL, KV_DIM);

    // RoPE on Q and K
    int q_rope_total = T * N_Q_HEADS * (HEAD_DIM / 2);
    rope_kernel<<<(q_rope_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        Q, cos, sin, T, N_Q_HEADS, HEAD_DIM);
    int k_rope_total = T * N_KV_HEADS * (HEAD_DIM / 2);
    rope_kernel<<<(k_rope_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        K, cos, sin, T, N_KV_HEADS, HEAD_DIM);

    // Causal GQA attention
    size_t attn_smem = (HEAD_DIM + T) * sizeof(float);
    cudaFuncSetAttribute(gqa_attention_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem);
    gqa_attention_kernel<<<dim3(T, N_Q_HEADS), BLOCK_SIZE, attn_smem>>>(
        Q, K, V, attn_out, T);

    // Output projection
    linear_kernel<<<dim3(T, (D_MODEL + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        attn_out, Wo, proj_out, T, Q_DIM, D_MODEL);

    // residual: x' = x + proj_out
    add_residual_kernel<<<(T * D_MODEL + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        x, proj_out, x_prime, T * D_MODEL);

    // === FFN 部分 ===
    // RMSNorm2
    rmsnorm_kernel<<<T, BLOCK_SIZE>>>(x_prime, w2, normed2, T, D_MODEL);

    // Gate & Up projection
    linear_kernel<<<dim3(T, (FFN_HIDDEN + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        normed2, Wgate, gate, T, D_MODEL, FFN_HIDDEN);
    linear_kernel<<<dim3(T, (FFN_HIDDEN + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        normed2, Wup, up, T, D_MODEL, FFN_HIDDEN);

    // SwiGLU
    swiglu_kernel<<<(T * FFN_HIDDEN + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        gate, up, ffn_mid, T * FFN_HIDDEN);

    // Down projection
    linear_kernel<<<dim3(T, (D_MODEL + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        ffn_mid, Wdown, proj_out, T, FFN_HIDDEN, D_MODEL);

    // residual: output = x' + down_out
    add_residual_kernel<<<(T * D_MODEL + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        x_prime, proj_out, output, T * D_MODEL);

    // 清理
    cudaFree(normed1);
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(attn_out);
    cudaFree(proj_out);
    cudaFree(x_prime);
    cudaFree(normed2);
    cudaFree(gate);
    cudaFree(up);
    cudaFree(ffn_mid);
}