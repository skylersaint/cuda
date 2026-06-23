#include <cuda_runtime.h>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 8; // 每 thread 的 M 方向 tile
constexpr int TN = 8; // 每 thread 的 N 方向 tile

// BM/TM = 8, BN/TN = 8, 每 block 64 threads
constexpr int BLOCK_SIZE = (BM / TM) * (BN / TN); // 64

__global__ void batched_gemm_kernel(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C,
                                    int M, int N, int K)
{
    int batch = blockIdx.z;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // 当前 batch 的 A, B, C 指针
    const float *A_batch = A + batch * M * K;
    const float *B_batch = B + batch * K * N;
    float *C_batch = C + batch * M * N;

    int tid = threadIdx.y * (BN / TN) + threadIdx.x;
    int thread_row = threadIdx.y; // 0..7
    int thread_col = threadIdx.x; // 0..7

    // Shared memory
    __shared__ float sA[BM][BK]; // 64×16 = 4 KB
    __shared__ float sB[BK][BN]; // 16×64 = 4 KB

    // 每 thread 的累加器(寄存器)
    float acc[TM][TN] = {0.0f};

    // 沿 K 方向迭代
    for (int kt = 0; kt < K; kt += BK)
    {
// === 加载 A 到 sA ===
// sA 是 64×16 = 1024 个元素, 64 thread → 每 thread 16 个
#pragma unroll
        for (int i = 0; i < (BM * BK + BLOCK_SIZE - 1) / BLOCK_SIZE; i++)
        {
            int idx = i * BLOCK_SIZE + tid;
            if (idx < BM * BK)
            {
                int r = idx / BK;
                int c = idx % BK;
                int gr = block_row * BM + r;
                int gc = kt + c;
                sA[r][c] = (gr < M && gc < K) ? A_batch[gr * K + gc] : 0.0f;
            }
        }

// === 加载 B 到 sB ===
#pragma unroll
        for (int i = 0; i < (BK * BN + BLOCK_SIZE - 1) / BLOCK_SIZE; i++)
        {
            int idx = i * BLOCK_SIZE + tid;
            if (idx < BK * BN)
            {
                int r = idx / BN;
                int c = idx % BN;
                int gr = kt + r;
                int gc = block_col * BN + c;
                sB[r][c] = (gr < K && gc < N) ? B_batch[gr * N + gc] : 0.0f;
            }
        }
        __syncthreads();

// === 计算 thread tile: 每 thread 算 TM × TN = 8×8 ===
#pragma unroll
        for (int k = 0; k < BK; k++)
        {
            // 把 sA 的一列、sB 的一行加载到寄存器
            float a_reg[TM];
            float b_reg[TN];
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
                a_reg[i] = sA[thread_row * TM + i][k];
            }
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                b_reg[j] = sB[k][thread_col * TN + j];
            }
// 8×8 = 64 FMA
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
#pragma unroll
                for (int j = 0; j < TN; j++)
                {
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // === 写回 C ===
    int c_row_base = block_row * BM + thread_row * TM;
    int c_col_base = block_col * BN + thread_col * TN;
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
            int r = c_row_base + i;
            int c = c_col_base + j;
            if (r < M && c < N)
            {
                C_batch[r * N + c] = acc[i][j];
            }
        }
    }
}

extern "C" void solve(const float *A, const float *B, float *C,
                      int BATCH, int M, int N, int K)
{
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    dim3 block(BN / TN, BM / TM); // 8×8 = 64 threads

    batched_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}