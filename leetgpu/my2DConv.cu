#include <cuda_runtime.h>

constexpr int BLOCK_DIM = 16;
constexpr int MAX_KS = 31;
constexpr int TILE_IN = BLOCK_DIM + MAX_KS - 1;
constexpr int MAX_KSIZE = MAX_KS * MAX_KS;

__global__ void conv2d_kernel(
    const float *__restrict__ input,
    const float *__restrict__ kernel,
    float *__restrict__ output,
    int IR, int IC, int kr, int kc, int OR, int OC)
{
    __shared__ float sIn[TILE_IN][TILE_IN];
    __shared__ float sK[MAX_KSIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_DIM + tx;

    int out_y0 = blockIdx.y * BLOCK_DIM;
    int out_x0 = blockIdx.x * BLOCK_DIM;

    int in_y0 = out_y0;
    int in_x0 = out_x0;

    int tile_h = BLOCK_DIM + kr - 1;
    int tile_w = BLOCK_DIM + kc - 1;

    int tile_size = tile_h * tile_w;
    for (int i = tid; i < tile_size; i += BLOCK_DIM * BLOCK_DIM)
    {
        int ty_local = i / tile_w;
        int tx_local = i % tile_w;
        int gy = in_y0 + ty_local;
        int gx = in_x0 + tx_local;
        float v = 0.0f;
        if (gy < IR && gx < IC)
            v = input[gy * IC + gx];
        sIn[ty_local][tx_local] = v;
    }
    int ksize = kr * kc;
    for (int i = tid; i < ksize; i += BLOCK_DIM * BLOCK_DIM)
    {
        sK[i] = kernel[i];
    }

    __syncthreads();

    int out_y = out_y0 + ty;
    int out_x = out_x0 + tx;
    if (out_y < OR && out_x < OC)
    {
        float sum = 0.0f;
    }
#pragma unroll 4
    for (int m = 0; m < kr; ++m)
    {
#pragma unroll 4
        for (int n = 0; n < kc; ++n)
        {
            sum = fmaf(sIn[ty + m][tx + n], sK[m * kc + n], sum);
        }
    }
    output[out_y * OC + out_x] = sum;
}

extern "C" void solve(
    const float *input, const float *kernel, float *output,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    int OR = input_rows - kernel_rows + 1;
    int OC = input_cols - kernel_cols + 1;

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((OC + BLOCK_DIM - 1) / BLOCK_DIM,
              (OR + BLOCK_DIM - 1) / BLOCK_DIM);
    conv2d_kernel<<<grid, block>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, OR, OC);
}
