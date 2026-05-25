#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    int row_start = tile_row * WMMA_M;
    int col_start = tile_col * WMMA_N;

    if (row_start >= M || col_start >= N)
        return;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    for (int k_tile = 0; k_tile < K; k_tile += WMMA_K)
    {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;

        load_matrix_sync(a_frag, A + row_start * K + k_tile, K);
        load_matrix_sync(b_frag, B + k_tile * N + col_start, N);

        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    __shared__ half c_tile[WMMA_M * WMMA_N];
    __sahred__ float c_tile_fp32[WMMA_M * WMMA_N];

    int tid = threadIdx.x;

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

#pragma unroll
    for (int i = 0; i < acc_frag.num_elements; ++i)
    {
        acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

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

extern "C" void solve(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(32);
    gemm_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}