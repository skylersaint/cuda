#include <cuda_runtime.h>

// ============================================================
// Tile 配置
//   Block tile:   128 × 128(每 block 算 C 的 128×128 子块)
//   Thread tile:  8 × 8(每 thread 算 8×8 个输出)
//   K-tile:       8(每次沿 K 加载 8 列)
//   Block size:   256 线程(16×16)
// ============================================================
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
// (BM/TM) × (BN/TN) = 16 × 16 = 256 线程
constexpr int BLOCK_SIZE = (BM / TM) * (BN / TN);

__global__ void __launch_bounds__(BLOCK_SIZE)
    sgemm_kernel(const float *__restrict__ A,
                 const float *__restrict__ B,
                 float *__restrict__ C,
                 int M, int N, int K)
{
    // Block 在 C 中的位置
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // 线程在 block 内的 2D 坐标(16×16)
    int tid = threadIdx.x;
    int thread_row = tid / (BN / TN); // 0..15
    int thread_col = tid % (BN / TN); // 0..15

    // Shared memory
    // sA 转置存储:[BK][BM],方便后续按列读(coalesced)
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    // 寄存器累加器:每 thread 8×8
    float acc[TM][TN] = {0.0f};
    // 寄存器缓存:从 shared 读出的一列 A 和一行 B
    float regA[TM];
    float regB[TN];

    // A, B 的起始位置(当前 block)
    const float *A_block = A + block_row * BM * K;
    const float *B_block = B + block_col * BN;

    // ── 加载索引预计算 ──
    // 用 float4 加载 A:每线程加载 A 的若干元素
    // A tile 是 BM×BK = 128×8 = 1024 元素,256 线程 → 每线程 4 个 = 1 个 float4
    int a_load_row = tid / (BK / 4);       // tid / 2,范围 0..127
    int a_load_col = (tid % (BK / 4)) * 4; // (tid%2)*4,0 或 4

    // B tile 是 BK×BN = 8×128 = 1024 元素,256 线程 → 每线程 1 个 float4
    int b_load_row = tid / (BN / 4);       // tid / 32,范围 0..7
    int b_load_col = (tid % (BN / 4)) * 4; // (tid%32)*4

    // ── 沿 K 方向遍历 ──
    for (int kt = 0; kt < K; kt += BK)
    {
        // === 加载 A tile 到 sA(转置存储)===
        // A[block_row*BM + a_load_row][kt + a_load_col .. +3]
        {
            int gr = block_row * BM + a_load_row;
            int gc = kt + a_load_col;
            float4 tmp = make_float4(0, 0, 0, 0);
            if (gr < M && gc + 3 < K)
            {
                tmp = *reinterpret_cast<const float4 *>(&A[gr * K + gc]);
            }
            else
            {
                // 边界标量加载
                if (gr < M)
                {
                    if (gc + 0 < K)
                        tmp.x = A[gr * K + gc + 0];
                    if (gc + 1 < K)
                        tmp.y = A[gr * K + gc + 1];
                    if (gc + 2 < K)
                        tmp.z = A[gr * K + gc + 2];
                    if (gc + 3 < K)
                        tmp.w = A[gr * K + gc + 3];
                }
            }
            // 转置写入 sA[col][row]
            sA[a_load_col + 0][a_load_row] = tmp.x;
            sA[a_load_col + 1][a_load_row] = tmp.y;
            sA[a_load_col + 2][a_load_row] = tmp.z;
            sA[a_load_col + 3][a_load_row] = tmp.w;
        }

        // === 加载 B tile 到 sB(直接存储)===
        {
            int gr = kt + b_load_row;
            int gc = block_col * BN + b_load_col;
            float4 tmp = make_float4(0, 0, 0, 0);
            if (gr < K && gc + 3 < N)
            {
                tmp = *reinterpret_cast<const float4 *>(&B[gr * N + gc]);
            }
            else
            {
                if (gr < K)
                {
                    if (gc + 0 < N)
                        tmp.x = B[gr * N + gc + 0];
                    if (gc + 1 < N)
                        tmp.y = B[gr * N + gc + 1];
                    if (gc + 2 < N)
                        tmp.z = B[gr * N + gc + 2];
                    if (gc + 3 < N)
                        tmp.w = B[gr * N + gc + 3];
                }
            }
            *reinterpret_cast<float4 *>(&sB[b_load_row][b_load_col]) = tmp;
        }

        __syncthreads();

// === 计算:沿 BK 做 outer product 累加 ===
#pragma unroll
        for (int k = 0; k < BK; k++)
        {
// 从 sA 读这个 thread 负责的 8 行(sA 已转置,sA[k][row] 连续)
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
                regA[i] = sA[k][thread_row * TM + i];
            }
// 从 sB 读这个 thread 负责的 8 列
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                regB[j] = sB[k][thread_col * TN + j];
            }
// 8×8 outer product 累加
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
#pragma unroll
                for (int j = 0; j < TN; j++)
                {
                    acc[i][j] = fmaf(regA[i], regB[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

// === 写回 C ===
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int gr = block_row * BM + thread_row * TM + i;
        if (gr >= M)
            continue;
#pragma unroll
        for (int j = 0; j < TN; j += 4)
        {
            int gc = block_col * BN + thread_col * TN + j;
            if (gc + 3 < N)
            {
                float4 out = make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
                *reinterpret_cast<float4 *>(&C[gr * N + gc]) = out;
            }
            else
            {
                if (gc + 0 < N)
                    C[gr * N + gc + 0] = acc[i][j + 0];
                if (gc + 1 < N)
                    C[gr * N + gc + 1] = acc[i][j + 1];
                if (gc + 2 < N)
                    C[gr * N + gc + 2] = acc[i][j + 2];
                if (gc + 3 < N)
                    C[gr * N + gc + 3] = acc[i][j + 3];
            }
        }
    }
}

extern "C" void solve(const float *A, const float *B, float *C,
                      int M, int N, int K)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}