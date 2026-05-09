constexpr int BLOCK_SIZE = 256;
constexpr int VEC_SIZE = 4;
constexpr int TILE = 4096;

__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val)
{
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
    if (wid == 0)
        val = warpReduceSum(val);
    return val;
}

__global__ void reduce_kernel(const float *input, float *output, int N)
{
    int blk_start = blockIdx.x * TILE;
    int blk_end = min(blk_start + TILE, N);
    if (blk_end <= blk_start)
        return;

    float val = 0.0f;

    // 向量化路径(主路径,所有完整 tile 走这里)
    if ((blk_end - blk_start) % VEC_SIZE == 0)
    {
        for (int i = blk_start + threadIdx.x * VEC_SIZE;
             i < blk_end;
             i += BLOCK_SIZE * VEC_SIZE)
        {
            float4 v = *reinterpret_cast<const float4 *>(input + i);
            val += (v.x + v.y) + (v.z + v.w);
        }
    }
    else
    {
        // 标量 fallback(只有最后一个 block 可能走这里)
        for (int i = blk_start + threadIdx.x; i < blk_end; i += BLOCK_SIZE)
            val += input[i];
    }

    val = blockReduceSum(val);

    if (threadIdx.x == 0)
        atomicAdd(output, val);
}

extern "C" void solve(const float *input, float *output, int N)
{
    int grid = (N + TILE - 1) / TILE;
    reduce_kernel<<<grid, BLOCK_SIZE>>>(input, output, N);
}