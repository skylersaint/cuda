// reduce.cu
// 编译: nvcc -O3 -arch=sm_80 reduce.cu -o reduce
// 运行: ./reduce

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// ============================================================
// 错误检查宏:每次 CUDA API 调用都应该包一下
// ============================================================
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// 启动 kernel 后检查的辅助宏
#define CUDA_CHECK_KERNEL()                  \
    do                                       \
    {                                        \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)

// ============================================================
// 配置常量
// ============================================================
constexpr int BLOCK_SIZE = 256;                          // 每个 block 的线程数
constexpr int ITEMS_PER_THREAD = 8;                      // 每个线程处理多少个元素
constexpr int TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD; // 2048

// ============================================================
// Warp 内归约:5 步 shuffle,完全在寄存器里
// ============================================================
__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val; // 线程 0 持有 warp 内总和
}

// ============================================================
// Block 内归约:warp_reductions 算法
// ============================================================
__device__ __forceinline__ float blockReduceSum(float val)
{
    static __shared__ float shared[32]; // 最多 32 个 warp 的部分和
    int lane = threadIdx.x & 31;        // warp 内 id (0..31)
    int wid = threadIdx.x >> 5;         // warp id (0..7 for BLOCK_SIZE=256)

    // 第一步:每个 warp 内归约
    val = warpReduceSum(val);

    // 第二步:每个 warp 的代表(lane 0)写到 shared
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    // 第三步:第一个 warp 把所有 warp 的部分和再归约一次
    val = (threadIdx.x < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
    if (wid == 0)
        val = warpReduceSum(val);

    return val; // 线程 0 持有 block 总和
}

// ============================================================
// 第一阶段 kernel:每个 block 处理一段输入,输出一个部分和
// ============================================================
__global__ void reduceStage1(const float *__restrict__ input,
                             float *__restrict__ partial_sums,
                             int n)
{
    // 当前 block 处理的起点
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * TILE_SIZE;

    // ── 阶段 A:每个线程串行加 ITEMS_PER_THREAD 个元素到寄存器 ──
    float thread_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        int idx = offset + i * BLOCK_SIZE + tid; // 跨步访问,保证 coalescing
        if (idx < n)
            thread_sum += input[idx];
    }

    // ── 阶段 B:block 内归约 ──
    float block_sum = blockReduceSum(thread_sum);

    // ── 阶段 C:线程 0 写出 block 部分和 ──
    if (tid == 0)
        partial_sums[bid] = block_sum;
}

// ============================================================
// 第二阶段 kernel:用一个 block 把所有 partial sums 加起来
// 这里假设 num_blocks <= TILE_SIZE,实战里如果数组超大需要再分一层
// ============================================================
__global__ void reduceStage2(const float *__restrict__ partial_sums,
                             float *__restrict__ result,
                             int num_blocks)
{
    int tid = threadIdx.x;

    // 每个线程加自己负责的部分(可能多个,如果 num_blocks > BLOCK_SIZE)
    float thread_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += BLOCK_SIZE)
    {
        thread_sum += partial_sums[i];
    }

    // block 内归约
    float total = blockReduceSum(thread_sum);

    if (tid == 0)
        *result = total;
}

// ============================================================
// Host 端封装:管理两阶段 kernel + 中间 buffer
// ============================================================
float deviceReduceSum(const float *d_input, int n)
{
    int num_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;

    // 分配中间 buffer 存储每个 block 的部分和
    float *d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));

    // 分配最终结果(放 device 上,避免最后一步 device→host 同步)
    float *d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    // ── 第一阶段:n 元素 → num_blocks 个部分和 ──
    reduceStage1<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    // ── 第二阶段:num_blocks 个部分和 → 1 个最终值 ──
    reduceStage2<<<1, BLOCK_SIZE>>>(d_partial, d_result, num_blocks);
    CUDA_CHECK(cudaGetLastError());

    // 拷回 host
    float h_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // 清理中间 buffer
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));

    return h_result;
}

// ============================================================
// CPU 参考实现,用于验证正确性
// ============================================================
float cpuReduceSum(const float *h_input, int n)
{
    // 用 double 累加避免 float 误差累积
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += h_input[i];
    return (float)sum;
}

// ============================================================
// main:数据准备、调用、计时、验证
// ============================================================
int main(int argc, char **argv)
{
    // 输入大小,默认 16M 元素 (~64 MB)
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    printf("Reducing %d elements (%.2f MB)\n", n, n * sizeof(float) / 1e6);

    // ── Host 端分配并初始化 ──
    float *h_input = (float *)malloc(n * sizeof(float));
    if (!h_input)
    {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // 用一个简单的可预测输入便于验证:全部填 1.0,期望结果就是 n
    for (int i = 0; i < n; ++i)
        h_input[i] = 1.0f;

    // ── Device 端分配 ──
    float *d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));

    // ── Warmup(避免第一次 launch 的额外开销影响计时)──
    for (int i = 0; i < 3; ++i)
        deviceReduceSum(d_input, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── 计时:用 CUDA event 测 GPU 时间 ──
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int n_iter = 100;
    float gpu_result = 0.0f;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < n_iter; ++i)
    {
        gpu_result = deviceReduceSum(d_input, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    float ms_per_iter = ms_total / n_iter;

    // ── CPU 参考结果 ──
    auto t0 = std::chrono::high_resolution_clock::now();
    float cpu_result = cpuReduceSum(h_input, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── 报告 ──
    printf("CPU result: %.4f (%.2f ms)\n", cpu_result, cpu_ms);
    printf("GPU result: %.4f (%.4f ms/iter)\n", gpu_result, ms_per_iter);

    float rel_err = fabsf(gpu_result - cpu_result) / fabsf(cpu_result);
    printf("Relative error: %.6e %s\n", rel_err,
           (rel_err < 1e-4f) ? "[PASS]" : "[FAIL]");

    // 带宽计算:reduce 是访存受限,关注有效带宽
    double bytes = (double)n * sizeof(float);
    double gb_per_s = bytes / (ms_per_iter * 1e-3) / 1e9;
    printf("Effective bandwidth: %.2f GB/s\n", gb_per_s);

    // ── 清理 ──
    CUDA_CHECK(cudaFree(d_input));
    free(h_input);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}