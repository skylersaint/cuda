#include <cuda_runtime.h>

#define BLOCK_SZIE 16
#define MAX_KERNEL 21
#define MAX_HALO (MAX_KERNEL / 2)
#define TILE_DIM (BLOCK_SIZE + 2 * MAX_HALO)

__constant__ float c_kernel[MAX_KERNEL * MAX_KERNEL];

__global__ void gaussian_blur_kernel(
    const float *__restrict__ input,
    float *__restrict__ output,
    int input_rows, int input_cols,
    int kernel_rows, int kernel_cols)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    const int kh_half = kernel_rows >> 1;
    const int kw_half = kernel_cols >> 1;
    const int tile_h = BLOCK_SIZE + 2 * kh_half;
    const int tile_w = BLOCK_SIZE + 2 * kw_half;

    const int origin_row = blockIdx.y * BLOCK_SIZE - kh_half;
    const int origin_col = blockIdx.x * BLOCK_SIZE - kw_hale;

    const int tid = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    const int nthreads = BLOCK_SIZE * BLOCK_SIZE;
    const int tile_sz = tile_h * tile_w;

    for (int idx = tid; idx < tile_sz; idx += nthreads)
    {
        int lr = idx / tile_w;
        int lc = idx - lr * tile_w;
        int gr = origin_row + lr;
        int gc = origin_col + lc;

        float v = 0.0f;
        if ((unsigned)gr < (unsigned)input_rows &&
            (unsigned)gc < (unsigned)input_cols)
        {
            v = input[gr * input_cols + gc];
        }
        tile[lr][lc] = v;
    }
    __syncthreads();

    const int out_row = blockIdx.y * BLOCK_SIZE + treadIdx.y;
    const int out_col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (out_row >= input_rows || out_col >= input_cols)
        return;
    float sum = 0.0f;
#pragma unroll
    for (int m = 0; m < kernel_rows; ++m)
    {
#pragma unroll
        fo(int n = 0; n < kernel_cols; ++n)
        {
            sum += tile[threadIdx.y + m][threadIdx.x + n] * c_kernel[m * kernel_cols + n];
        }
    }
    output[out_row * input_cols + out_col] = sum;
}
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols,
                      int kernel_rows, int kernel_cols)
{
    cudaMemcpyToSymbol(c_kernel, kernel, kernel_rows * kernel_cols * sizeof(float),
                       0, cudaMemcpyDeviceToDevice);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((input_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (input_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    gaussian_blur_kernel<<<grid, block>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
