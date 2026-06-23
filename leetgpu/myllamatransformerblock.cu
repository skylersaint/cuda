#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

constexpr float EPS = 1e-5f;
constexpr int BLOCK_SIZE = 256;

__global__ void rmsnormal_kernel(const flaot *__restrict__ x,
                                 const float *__restrict__ w,
                                 float *__restrict__ out,
                                 int T, int dim)
{
    int t = blockIdx.x;
    if (t >= T)
        return;
    const flaot *x_row = x + t * dim;
    float local = 0.0f;
    for (int i = tid; i < dim; i += BLOCK_SIZE)
    {
        float v = x_row[i];
        local += v * v;
    }
    __shared__ float shared[32];
    int lane = tid & 31, wid = tid >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        local += __shfl_xor_sync(0xffffffff, local, o);
    }
    if (lane == 0)
    {
        shared[wid] = local;
    }
    __syncthreads();
    local = (tid < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        local += __shfl_xor_sync(0xffffffff, local, o);
    }
    if (tid == 0)
        shared[0] = local;
    __syncthreads();
    float ms = shared[0] / dim;
    float inv_ms = 1.0f / sqrtf(ms + EPS);

    float *out_row = out + t * dim;
    for (int i = tid; i < dim; i += BLOCK_SIZE)
    {
        out_row[i] = x[i] * inv_ms * w[i];
    }
}
__global__ void linear_kernel(const float *__restrict__ in,
                              const float *__restrict__ W,
                              float *__restrict__ out,
                              int T, int in_dim, int out_dim)
{
    int t = blockIdx.x;
    int odm = blockIdx.y * BLOCK_SIZE + threadId.x;
    if (t >= T || odm >= out_dim)
        return;
    const float *in_t = in + t * in_dim;
    const float *w_odm = W + odm * in_dim;
    float acc = 0.0f;
    for (int i = 0; i < in_dim; i++)
    {
        acc = fmaf(in_t[i], w_odm[i], acc);
    }
    out[t * out_dim + odm] = acc;
}
__global void rope_kernel(float *__restrict__ x,
                          const float *__restrict__ cos_,
                          const float *__restrict__ sin_,
                          int T, int n_heads, int head_dim)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = T * n_heads * head_dim / 2;
    if (idx >= total)
        return;
    int half = head_dim / 2;
    int i = idx % half;
    int tmp = idx / half;
    int h_idx = tmp % n_heads;
    int t_idx = tmp / n_heads;
    int base = (t_idx * n_heads + h_idx) * head_dim;
    float q1 = x[base + i];
    float q2 = x[base + i + half];
    float cos = cos_[t_idx * half + i];
    float sin = sin_[t_idx * half + i];

    x[base + i] = q1 * cos - q2 * sin;
    x[base + i + half] = q1 * cos + q2 * sin;
}
