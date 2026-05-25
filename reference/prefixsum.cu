#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int ELEMS_PER_THREAD = 4;
constexpr int TILE = BLOCK_SIZE * ELEMS_PER_THREAD; // 1024

// ============================================================
// Warp 内 inclusive scan(5 步 shuffle)
// ============================================================
__device__ __forceinline__ float warpInclusiveScan(float val)
{
    int lane = threadIdx.x & 31;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset)
            val += n;
    }
    return val;
}

// ============================================================
// Block 内 inclusive scan
// 输入:每线程持有 val
// 输出:每线程持有"block 内 inclusive scan 后的对应值"
// 副产出:thread 0 通过 *block_sum 返回 block 总和(若非 nullptr)
// ============================================================
__device__ __forceinline__ float blockInclusiveScan(float val, float *block_sum)
{
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    constexpr int N_WARPS = BLOCK_SIZE / 32; // 8

    // Step 1: 每 warp 内 scan
    val = warpInclusiveScan(val);

    // Step 2: 每 warp 的总和写到 shared
    if (lane == 31)
        warp_sums[wid] = val;
    __syncthreads();

    // Step 3: 第一个 warp 对 N_WARPS 个 warp 总和做 scan
    if (wid == 0)
    {
        float v = (lane < N_WARPS) ? warp_sums[lane] : 0.0f;
        v = warpInclusiveScan(v);
        if (lane < N_WARPS)
            warp_sums[lane] = v;
    }
    __syncthreads();

    // Step 4: 加上前面所有 warp 的总和(exclusive 偏移)
    float warp_offset = (wid > 0) ? warp_sums[wid - 1] : 0.0f;
    val += warp_offset;

    // 副产出:block 总和
    if (block_sum && threadIdx.x == BLOCK_SIZE - 1)
    {
        *block_sum = val;
    }
    return val;
}

// ============================================================
// Kernel 1: 每 block 处理一个 tile,做 inclusive scan
// 同时输出 block_sums[blockIdx.x] = 这段的总和
// ============================================================
__global__ void scan_block_kernel(const float *__restrict__ input,
                                  float *__restrict__ output,
                                  float *__restrict__ block_sums,
                                  int N)
{
    __shared__ float s_block_sum;

    int base = blockIdx.x * TILE;

    // ── 加载 tile 到寄存器 + 每线程串行 inclusive scan ──
    float vals[ELEMS_PER_THREAD];
#pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k)
    {
        int idx = base + threadIdx.x * ELEMS_PER_THREAD + k;
        vals[k] = (idx < N) ? input[idx] : 0.0f;
    }
// 串行前缀和:vals[k] = sum of original vals[0..k]
#pragma unroll
    for (int k = 1; k < ELEMS_PER_THREAD; ++k)
    {
        vals[k] += vals[k - 1];
    }

    // 此时 vals[ELEMS_PER_THREAD - 1] 是该线程的 local sum
    // 用 block scan 得到"前面所有线程 local sum 之和"作为该线程的偏移
    float thread_sum = vals[ELEMS_PER_THREAD - 1];
    float scanned = blockInclusiveScan(thread_sum,
                                       (threadIdx.x == BLOCK_SIZE - 1) ? &s_block_sum : nullptr);
    // scanned 是包含 thread_sum 自己的前缀和,exclusive 偏移 = scanned - thread_sum
    float thread_offset = scanned - thread_sum;

// ── 写出 ──
#pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k)
    {
        int idx = base + threadIdx.x * ELEMS_PER_THREAD + k;
        if (idx < N)
        {
            output[idx] = vals[k] + thread_offset;
        }
    }

    // ── thread 0 写出 block sum ──
    if (block_sums && threadIdx.x == 0)
    {
        __syncthreads(); // 等 s_block_sum 写好
        block_sums[blockIdx.x] = s_block_sum;
    }
}

// ============================================================
// Kernel 2: 对 block_sums 做 exclusive scan,得到每个 block 的起始偏移
// 假设 num_blocks <= BLOCK_SIZE * 多次迭代,用单 block 处理
// 对 N=25万,num_blocks ≈ 245,单 block 足够
// ============================================================
__global__ void scan_block_sums_kernel(float *__restrict__ block_sums,
                                       float *__restrict__ block_offsets,
                                       int num_blocks)
{
    // 把 block_sums 转成 exclusive scan,存到 block_offsets
    __shared__ float carry;
    if (threadIdx.x == 0)
        carry = 0.0f;
    __syncthreads();

    // 多次迭代处理大于 BLOCK_SIZE 的情况
    for (int base = 0; base < num_blocks; base += BLOCK_SIZE)
    {
        int i = base + threadIdx.x;
        float v = (i < num_blocks) ? block_sums[i] : 0.0f;
        float scanned = blockInclusiveScan(v, nullptr);
        // exclusive = inclusive - 自身 + carry
        if (i < num_blocks)
        {
            block_offsets[i] = scanned - v + carry;
        }
        // 更新 carry
        __syncthreads();
        if (threadIdx.x == BLOCK_SIZE - 1 || i == num_blocks - 1)
        {
            // 取 scanned 的最后一个值作为新 carry
            // 但只有 BLOCK_SIZE - 1 那个线程有正确值
        }
        // 简单做法:再写一个 shared 存 last scanned
        __shared__ float last_scanned;
        int last_thread_in_range = min(BLOCK_SIZE - 1, num_blocks - 1 - base);
        if (threadIdx.x == last_thread_in_range)
        {
            last_scanned = scanned;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            carry += last_scanned;
        }
        __syncthreads();
    }
}

// ============================================================
// Kernel 3: 把每个 block 的 offset 加到 output 上
// ============================================================
__global__ void add_offset_kernel(float *__restrict__ output,
                                  const float *__restrict__ block_offsets,
                                  int N)
{
    int base = blockIdx.x * TILE;
    float offset = block_offsets[blockIdx.x];

#pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k)
    {
        int idx = base + threadIdx.x * ELEMS_PER_THREAD + k;
        if (idx < N)
        {
            output[idx] += offset;
        }
    }
}

// ============================================================
// Host 端
// ============================================================
static float *d_block_sums = nullptr;
static float *d_block_offsets = nullptr;
static int d_workspace_size = 0;

extern "C" void solve(const float *input, float *output, int N)
{
    int num_blocks = (N + TILE - 1) / TILE;

    if (num_blocks > d_workspace_size)
    {
        if (d_block_sums)
            cudaFree(d_block_sums);
        if (d_block_offsets)
            cudaFree(d_block_offsets);
        cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
        cudaMalloc(&d_block_offsets, num_blocks * sizeof(float));
        d_workspace_size = num_blocks;
    }

    // Kernel 1
    scan_block_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, d_block_sums, N);

    // Kernel 2:对 block_sums 做 exclusive scan
    if (num_blocks > 1)
    {
        scan_block_sums_kernel<<<1, BLOCK_SIZE>>>(d_block_sums, d_block_offsets, num_blocks);

        // Kernel 3:加 offset
        add_offset_kernel<<<num_blocks, BLOCK_SIZE>>>(output, d_block_offsets, N);
    }
    // num_blocks == 1 时 Kernel 1 已经完成全部工作
}