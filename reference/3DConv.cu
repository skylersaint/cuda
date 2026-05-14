#include <cuda_runtime.h>

constexpr int BD = 4;                 // block tile depth
constexpr int BR = 8;                 // block tile rows
constexpr int BC = 8;                 // block tile cols
constexpr int BLOCK_N = BD * BR * BC; // 256 线程
constexpr int MAX_KD = 5;
constexpr int MAX_KR = 5;
constexpr int MAX_KC = 5;
constexpr int TILE_D = BD + MAX_KD - 1;           // 8
constexpr int TILE_R = BR + MAX_KR - 1;           // 12
constexpr int TILE_C = BC + MAX_KC - 1;           // 12
constexpr int MAX_KSZ = MAX_KD * MAX_KR * MAX_KC; // 125

__global__ void conv3d_kernel(
    const float *__restrict__ input,
    const float *__restrict__ kernel,
    float *__restrict__ output,
    int ID, int IR, int IC,
    int KD, int KR, int KC,
    int OD, int OR, int OC)
{

    __shared__ float sIn[TILE_D][TILE_R][TILE_C];
    __shared__ float sK[MAX_KSZ];

    // 线程的 3D 坐标(block 内)
    int tid = threadIdx.x;
    int tz = tid / (BR * BC);
    int ty = (tid / BC) % BR;
    int tx = tid % BC;

    // 当前 block 处理的输出 tile 起点
    int out_z0 = blockIdx.z * BD;
    int out_y0 = blockIdx.y * BR;
    int out_x0 = blockIdx.x * BC;

    // input tile 起点 = output tile 起点(valid 卷积下两者重合)
    int in_z0 = out_z0;
    int in_y0 = out_y0;
    int in_x0 = out_x0;

    // 实际需要的 tile 尺寸(运行时,因为 KD/KR/KC 是变量)
    int tile_d = BD + KD - 1;
    int tile_r = BR + KR - 1;
    int tile_c = BC + KC - 1;
    int tile_sz = tile_d * tile_r * tile_c;

    // ── 1. 加载 input tile,256 线程协作 ──
    for (int i = tid; i < tile_sz; i += BLOCK_N)
    {
        int lz = i / (tile_r * tile_c);
        int ly = (i / tile_c) % tile_r;
        int lx = i % tile_c;
        int gz = in_z0 + lz;
        int gy = in_y0 + ly;
        int gx = in_x0 + lx;
        float v = 0.0f;
        if (gz < ID && gy < IR && gx < IC)
            v = input[(gz * IR + gy) * IC + gx];
        sIn[lz][ly][lx] = v;
    }

    // ── 2. 加载 kernel ──
    int ksz = KD * KR * KC;
    for (int i = tid; i < ksz; i += BLOCK_N)
    {
        sK[i] = kernel[i];
    }

    __syncthreads();

    // ── 3. 计算输出 ──
    int out_z = out_z0 + tz;
    int out_y = out_y0 + ty;
    int out_x = out_x0 + tx;
    if (out_z < OD && out_y < OR && out_x < OC)
    {
        float sum = 0.0f;
#pragma unroll 4
        for (int d = 0; d < KD; ++d)
        {
#pragma unroll 4
            for (int r = 0; r < KR; ++r)
            {
#pragma unroll 4
                for (int c = 0; c < KC; ++c)
                {
                    sum = fmaf(
                        sIn[tz + d][ty + r][tx + c],
                        sK[(d * KR + r) * KC + c],
                        sum);
                }
            }
        }
        output[(out_z * OR + out_y) * OC + out_x] = sum;
    }
}

extern "C" void solve(
    const float *input, const float *kernel, float *output,
    int input_depth, int input_rows, int input_cols,
    int kernel_depth, int kernel_rows, int kernel_cols)
{

    int OD = input_depth - kernel_depth + 1;
    int OR = input_rows - kernel_rows + 1;
    int OC = input_cols - kernel_cols + 1;

    dim3 block(BLOCK_N);
    dim3 grid((OC + BC - 1) / BC,
              (OR + BR - 1) / BR,
              (OD + BD - 1) / BD);

    conv3d_kernel<<<grid, block>>>(
        input, kernel, output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        OD, OR, OC);
}