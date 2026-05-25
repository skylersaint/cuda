#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void gemm_wmma_kernel(const half *__restrict__ A,
                                 const half *__restrict__ B,
                                 half *__restrict__ C,
                                 int M, int N, int K,
                                 float alpha, float beta)
{
    // 每 block 处理 C 的一个 16x16 tile
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    int row_start = tile_row * WMMA_M;
    int col_start = tile_col * WMMA_N;

    if (row_start >= M || col_start >= N)
        return;

    // 累加器(FP32 fragment)
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    // 沿 K 维度遍历
    for (int k_tile = 0; k_tile < K; k_tile += WMMA_K)
    {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;

        // 加载 A 的 16x16 fragment(从 [row_start, k_tile])
        load_matrix_sync(a_frag, A + row_start * K + k_tile, K);
        // 加载 B 的 16x16 fragment(从 [k_tile, col_start])
        load_matrix_sync(b_frag, B + k_tile * N + col_start, N);

        // tensor core: acc += A × B
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // 加载 C_initial,做 alpha * acc + beta * C
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 把 half 的 C 读出来,转 FP32 累加
    // WMMA 没有直接 load half 到 fp32 accumulator 的 API
    // 简单做法:用 shared memory 中转
    __shared__ half c_tile[WMMA_M * WMMA_N];
    __shared__ float c_tile_fp32[WMMA_M * WMMA_N];

    int tid = threadIdx.x;
// 16x16 = 256,一个 warp(32 线程)每线程加载 8 个
#pragma unroll
    for (int i = tid; i < WMMA_M * WMMA_N; i += 32)
    {
        int local_row = i / WMMA_N;
        int local_col = i % WMMA_N;
        int gr = row_start + local_row;
        int gc = col_start + local_col;
        c_tile_fp32[i] = (gr < M && gc < N) ? __half2float(C[gr * N + gc]) : 0.0f;
    }
    __syncwarp();

    load_matrix_sync(c_frag, c_tile_fp32, WMMA_N, mem_row_major);

// alpha * acc + beta * c
#pragma unroll
    for (int i = 0; i < acc_frag.num_elements; ++i)
    {
        acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // 写回 C(FP16)
    store_matrix_sync(c_tile_fp32, acc_frag, WMMA_N, mem_row_major);
    __syncwarp();

#pragma unroll
    for (int i = tid; i < WMMA_M * WMMA_N; i += 32)
    {
        int local_row = i / WMMA_N;
        int local_col = i % WMMA_N;
        int gr = row_start + local_row;
        int gc = col_start + local_col;
        if (gr < M && gc < N)
        {
            C[gr * N + gc] = __float2half(c_tile_fp32[i]);
        }
    }
}

extern "C" void solve(const half *A, const half *B, half *C,
                      int M, int N, int K,
                      float alpha, float beta)
{
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(32); // 1 warp per block

    gemm_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}